# 31 점검: 20년치 구별 분기 실거래가격지수(부동산원, KOSIS 408/DT_KAB_11672_S5, 2006Q1~)로 "다음 분기 많이 오를 구" 순위를 모델이 최근 흐름보다 잘 맞히는가
# - 분기 k 지수는 분기 종료 약 7~8주 뒤 공표 → 공표 시점에 아는 값은 k까지, 정답은 공표 뒤 시작하는 분기 성장률 log I(k+2) - log I(k+1)
# - 정답: 같은 분기 25개 구 중 상위 8개(1/3). 학습은 정답이 확정된 분기만(k' + 2 ≤ k) 써서 매 분기 다시 학습
# - 갭 메우기 가설: 지난 3년 서울보다 덜 오른 구가 다음에 따라 오르는가 → '상대_3년변화' Feature
import os, sys, warnings
import numpy as np, pandas as pd, requests
from dotenv import dotenv_values
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = os.path.join(HERE, *['..'] * 5, '')
CACHE = HERE + 'kab_gu_quarterly.parquet'
if not os.path.exists(CACHE):
    K = dotenv_values(ROOT + '.env')['KSKILL_KOSIS_API_KEY']
    d = requests.get('https://kosis.kr/openapi/Param/statisticsParameterData.do', timeout=60, params={
        'apiKey': K, 'format': 'json', 'jsonVD': 'Y', 'method': 'getList', 'orgId': '408', 'tblId': 'DT_KAB_11672_S5',
        'prdSe': 'Q', 'startPrdDe': '200601', 'endPrdDe': '202704', 'itmId': 'ALL', 'objL1': 'ALL'}).json()
    q = pd.DataFrame(d)[['C1', 'C1_NM', 'PRD_DE', 'DT']]
    q = q[q['C1'].str.match(r'^100\d\d$')].rename(columns={'C1_NM': '자치구', 'PRD_DE': '분기', 'DT': '지수'})  # 서울 25개 구 (코드 10001~10025)
    q['지수'] = q['지수'].astype(float); q.to_parquet(CACHE, index=False)
q = pd.read_parquet(CACHE)
w = np.log(q.pivot(index='분기', columns='자치구', values='지수').sort_index())
assert w.shape[1] == 25, w.shape
print(f'구별 분기 지수 {w.index[0]} ~ {w.index[-1]} ({len(w)}개 분기)')
seoul = w.mean(axis=1)
rows = []
for k in range(12, len(w) - 2):
    for gu in w.columns:
        s = w[gu]
        rows.append({'k': k, '분기': w.index[k], '자치구': gu,
                     '1분기변화': s.iloc[k] - s.iloc[k - 1], '2분기변화': s.iloc[k] - s.iloc[k - 2], '4분기변화': s.iloc[k] - s.iloc[k - 4],
                     '상대_1년변화': (s.iloc[k] - s.iloc[k - 4]) - (seoul.iloc[k] - seoul.iloc[k - 4]),
                     '상대_3년변화': (s.iloc[k] - s.iloc[k - 12]) - (seoul.iloc[k] - seoul.iloc[k - 12]),
                     '서울_1분기변화': seoul.iloc[k] - seoul.iloc[k - 1], '서울_4분기변화': seoul.iloc[k] - seoul.iloc[k - 4],
                     '정답_성장률': w[gu].iloc[k + 2] - w[gu].iloc[k + 1]})
f = pd.DataFrame(rows)
f['y'] = (f.groupby('k')['정답_성장률'].rank(ascending=False, method='first') <= 8).astype(int)
for c in ['1분기변화', '2분기변화', '4분기변화']:  # 같은 분기 안 순위(구끼리 비교하는 문제)
    f[f'{c}_순위'] = f.groupby('k')[c].rank(pct=True)
FEATS = {'최근 흐름만 (1·2·4분기 순위)': ['1분기변화_순위', '2분기변화_순위', '4분기변화_순위'],
         '최근 흐름 + 갭 메우기': ['1분기변화_순위', '2분기변화_순위', '4분기변화_순위', '상대_1년변화', '상대_3년변화'],
         '최근 흐름 + 갭 메우기 + 서울 국면': ['1분기변화_순위', '2분기변화_순위', '4분기변화_순위', '상대_1년변화', '상대_3년변화', '서울_1분기변화', '서울_4분기변화']}
def logit(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
def lgbm(): return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=5, min_child_samples=40, subsample=0.8, subsample_freq=1, verbose=-1, random_state=42)
START = 24  # 앞 24개 분기(2006~2011)는 학습만
preds = {f'기준선: {c}': f.loc[f.k >= START, c].to_numpy() for c in ['1분기변화', '2분기변화', '4분기변화']}
for sname, feats in FEATS.items():
    for mname, mk in [('로지스틱', logit), ('LightGBM', lgbm)]:
        out = []
        for k in range(START, f.k.max() + 1):
            trm = f.k <= k - 2; m = mk().fit(f.loc[trm, feats], f.loc[trm, 'y'])
            out.append(m.predict_proba(f.loc[f.k == k, feats])[:, 1])
        preds[f'{mname} · {sname}'] = np.concatenate(out)
ev = f[f.k >= START].copy(); ev['연'] = ev['분기'].str[:4].astype(int)
PERIODS = {'전체 2012~2026': ev['연'] >= 0, '2012~2016': ev['연'].between(2012, 2016), '2017~2021': ev['연'].between(2017, 2021),
           '2022~2024': ev['연'].between(2022, 2024), '2025~2026': ev['연'] >= 2025}
def qauc(mask, p):
    g = ev[mask].assign(p=p[mask.to_numpy()])
    return np.mean([roc_auc_score(x['y'], x['p']) for _, x in g.groupby('k')])
res = pd.DataFrame({name: {pn: qauc(pm, p) for pn, pm in PERIODS.items()} for name, p in preds.items()}).T.round(3)
# 상위 8개 적중률 (전체 기간)
def hit(p):
    g = ev.assign(p=p); return np.mean([x.nlargest(8, 'p')['y'].mean() for _, x in g.groupby('k')])
res['상위8 적중률(전체)'] = [round(hit(p), 3) for p in preds.values()]
print(res.to_string())
res.to_csv(HERE + 'r31_long_quarterly.csv', encoding='utf-8-sig')
# 갭 메우기 단독 신호: 지난 3년 덜 오른 구(상대_3년변화 낮음)가 다음 분기 상위에 드는가
print('\n갭 메우기 단독(지난 3년 덜 오른 구 = 높은 점수) 분기내 AUC:', {pn: round(qauc(pm, -ev['상대_3년변화'].to_numpy()), 3) for pn, pm in PERIODS.items()})
