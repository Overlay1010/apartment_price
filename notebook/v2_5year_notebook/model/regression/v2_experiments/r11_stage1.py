import os
import sys, time, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 5, '')
D = ROOT + "data/v2_5year_seoul/processed/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')

reg = pd.read_csv(D + "v2_feature_registry.csv")
reg = reg[reg.dataset.eq("v2_trade")]
t = pd.read_parquet(D + "v2_trade.parquet")
clus = pd.read_parquet(ROOT + "models/clustering/v2/trade_clusters.parquet")
t = t.merge(clus, on="transaction_id", how="left", validate="one_to_one")
t["단지일련번호"] = t["단지일련번호"].astype(str)
y = np.log(t["거래금액"].to_numpy(float))
split = t["데이터구분"].to_numpy()
tr, va, te = split == "train", split == "valid", split == "test"


def cols(sources, roles=("model",)):
    return reg[reg.source.isin(sources) & reg.role.isin(roles)].column.tolist()


V1 = cols(["V1"])
MARKET = cols(["06_market(확정월)"]); RENT = cols(["06_rent(확정월)"]); RATE = cols(["06_rate"])
LOC = cols(["06_location(시점)"])
RENT_FLAGS = [c for c in reg.column if c.startswith("전세피처출처_확정")]
GONGSI = [c for c in cols(["05b_complex"]) if c.startswith("공시")]
COMPLEX = [c for c in cols(["05b_complex"]) if not c.startswith("공시")]
SCHOOL = cols(["02_school"])
REGION = cols(["01_market", "03_policy", "03_policy_trade", "04_supply", "05_demographics", "06_supply_ratio"])
CLUSTER = ["location_cluster", "education_cluster"]
V1RAW = V1 + [c for c in reg.column if c.endswith("_v1")] + ["응급의료기관거리_m"]
CORE = V1 + MARKET + RENT + RATE + LOC

CONFIGS = {
    "V1raw (V1 시점 그대로)": V1RAW,
    "F0 V1정의+V2시점 (출처·응급·유치원·학교거리 포함)": CORE + RENT_FLAGS + ["응급의료기관거리_m", "유치원거리_m",
                                                           "초등학교거리_m_v1", "중학교거리_m_v1", "고등학교거리_m_v1"],
    "F1 핵심 (레지스트리 기본)": CORE,
    "F1+공시": CORE + GONGSI,
    "F2 +단지속성": CORE + COMPLEX,
    "F3 +단지속성+공시": CORE + COMPLEX + GONGSI,
    "F4 +학교·학원": CORE + COMPLEX + GONGSI + SCHOOL,
    "F5 +구×월 (전체 196)": CORE + COMPLEX + GONGSI + SCHOOL + REGION,
    "F6 +군집 라벨": CORE + COMPLEX + GONGSI + SCHOOL + REGION + CLUSTER,
}
assert set(CONFIGS["F5 +구×월 (전체 196)"]) == set(reg[reg.role.eq("model")].column), \
    set(reg[reg.role.eq("model")].column) ^ set(CONFIGS["F5 +구×월 (전체 196)"])


def frame(features):
    X = t[features].copy()
    for c in features:
        if X[c].dtype == object or c in CLUSTER:
            X[c] = X[c].astype("category")
    return X


def fit(X, tr_mask, ev_mask, seed=42):
    m = XGBRegressor(n_estimators=6000, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                     tree_method="hist", enable_categorical=True, early_stopping_rounds=100, n_jobs=8, random_state=seed)
    m.fit(X[tr_mask], y[tr_mask], eval_set=[(X[ev_mask], y[ev_mask])], verbose=False)
    return m


def metrics(mask, pred_log):
    a = np.exp(y[mask]); p = np.exp(pred_log)
    ape = np.abs(a - p) / a * 100
    return {"MAPE": ape.mean(), "중앙APE": np.median(ape), "10%이내": (ape <= 10).mean() * 100, "n": int(mask.sum())}


train_complexes = set(t.loc[tr, "단지일련번호"])
seen_va = va & t["단지일련번호"].isin(train_complexes).to_numpy()
unseen_va = va & ~t["단지일련번호"].isin(train_complexes).to_numpy()
rng = np.random.default_rng(0)
holdouts = []
uniq = np.array(sorted(train_complexes))
for s in range(3):
    held = set(rng.choice(uniq, int(0.2 * len(uniq)), replace=False))
    in_held = t["단지일련번호"].isin(held).to_numpy()
    holdouts.append((tr & ~in_held, va & in_held, va & ~in_held))

rows = []
for name, feats in CONFIGS.items():
    st = time.time()
    X = frame(feats)
    m = fit(X, tr, va)
    p = m.predict(X[va]); pv = pd.Series(p, index=np.where(va)[0])
    row = {"구성": name, "Feature수": len(feats), "best_iter": m.best_iteration}
    row.update({f"Valid_{k}": v for k, v in metrics(va, p).items() if k != "n"})
    row["Seen_MAPE"] = metrics(seen_va, pv.loc[np.where(seen_va)[0]].to_numpy())["MAPE"]
    row["Unseen_MAPE(Valid 신규)"] = metrics(unseen_va, pv.loc[np.where(unseen_va)[0]].to_numpy())["MAPE"]
    hm = []
    for tr_h, ev_h, es_h in holdouts:
        mh = fit(X, tr_h, es_h, seed=1)  # 조기 종료는 홀드아웃 아닌 단지의 Valid로
        hm.append(metrics(ev_h, mh.predict(X[ev_h]))["MAPE"])
    row["홀드아웃_MAPE"] = float(np.mean(hm)); row["홀드아웃_MAPE_범위"] = f"{min(hm):.2f}~{max(hm):.2f}"
    row["초"] = round(time.time() - st)
    rows.append(row)
    print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT + "r11_stage1.csv", index=False, encoding="utf-8-sig")

# 단순 규칙: 같은 단지 확정 12개월(m-13~m-2) 중위 평당가 × 평 → 동 → 구
t["월"] = t["거래월일자"].dt.to_period("M")
base = t[["단지일련번호", "법정동", "시군구명", "월", "평당가"]]
def ref_price(level):
    g = base.groupby([level, "월"])["평당가"].agg(list)
    return g
out = np.full(len(t), np.nan)
src = np.array([""] * len(t), dtype=object)
months = sorted(t.loc[va | te, "월"].unique())
for mth in months:
    window = base[(base["월"] >= mth - 13) & (base["월"] <= mth - 2)]
    rows_m = np.where(((va | te)) & (t["월"] == mth).to_numpy())[0]
    for level in ["단지일련번호", "법정동", "시군구명"]:
        med = window.groupby(level)["평당가"].median()
        need = rows_m[np.isnan(out[rows_m])]
        vals = t.iloc[need][level].map(med).to_numpy(float)
        ok = ~np.isnan(vals)
        out[need[ok]] = vals[ok]; src[need[ok]] = level
naive = np.log(out * t["전용면적"].to_numpy(float) * 0.3025)
nrow = {"구성": "단순 규칙 (단지 확정 12개월 중위 평당가 × 평)"}
for nm, mk in [("Valid", va), ("Test", te), ("Seen", seen_va), ("Unseen(Valid 신규)", unseen_va)]:
    nrow[f"{nm}_MAPE"] = metrics(mk, naive[mk])["MAPE"]
nrow["단지값 사용 비율(Valid)"] = float((src[va] == "단지일련번호").mean())
print(nrow)
json.dump(nrow, open(OUT + "r11_naive.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
