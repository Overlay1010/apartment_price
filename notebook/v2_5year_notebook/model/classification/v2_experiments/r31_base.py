# 31 공통: 구 × 예측시점 데이터 · Feature 묶음 · 라벨 시점 규칙 · 백테스트 · 지표
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score, precision_score, recall_score
sys.stdout.reconfigure(encoding='utf-8'); warnings.filterwarnings('ignore')
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = os.path.join(HERE, *['..'] * 5, ''); D = ROOT + 'data/v2_5year_seoul/processed/'
reg = pd.read_csv(D + 'v2_feature_registry.csv'); reg = reg[reg.dataset.eq('v2_region_month')]
r = pd.read_parquet(D + 'v2_region_month.parquet')
r['p'] = r['예측시점'].dt.to_period('M')
TARGET = os.environ.get('R31_TARGET', '상승세')   # 상승세(+1%) · 강한상승(+3%) · 상승여부(V1 중위 평당가, 비교용)
r = r[r[TARGET].notna()].copy(); r['상승여부_v1'] = r['상승여부']; r['상승여부'] = r[TARGET].astype(int)
def src(*s): return reg[reg.role.eq('model') & reg.source.isin(s)].column.tolist()
G = {'시장(확정월)': src('06_market(확정월)'), '전월세(확정월)': src('06_rent(확정월)'), '금리': src('06_rate'),
     '시장지표(01)': [c for c in src('01_market') if c != '권역'], '정책(03)': src('03_policy'),
     '공급(04)': src('04_supply', '06_supply_ratio'), '인구(05)': src('05_demographics'), '공식지수': src('06_official_index')}
# 회귀 OOF Gap: 거래월 M의 Gap은 M+2월 1일부터 사용 가능 → 예측시점 p에는 M ≤ p-2
gap = pd.read_parquet(ROOT + 'models/regression/v2/oof_gap_gu_month.parquet'); gap['M'] = gap['월'].dt.to_period('M')
g = gap.pivot(index='M', columns='시군구명', values='중개_gap_중위')
rows = []
for p in r['p'].unique():
    last = g.loc[g.index <= p - 2]
    if len(last) == 0: continue
    for gu in g.columns:
        s = last[gu].dropna()
        rows.append({'p': p, '자치구': gu, 'gap_최근월': s.iloc[-1] if len(s) else np.nan, 'gap_3개월': s.iloc[-3:].mean() if len(s) else np.nan,
                     'gap_3개월변화': s.iloc[-3:].mean() - s.iloc[-6:-3].mean() if len(s) >= 6 else np.nan,
                     'gap_서울_3개월': last.iloc[-3:].stack().mean()})
r = r.merge(pd.DataFrame(rows), on=['p', '자치구'], how='left')
G['회귀 OOF Gap'] = ['gap_최근월', 'gap_3개월', 'gap_3개월변화', 'gap_서울_3개월']
# V1 분류 Feature (기준월 m = p-1 당월 거래 사용) — V1 노트북 재현값
v1 = pd.read_parquet(HERE + 'v1_cls_df.parquet'); v1['p'] = v1['기준월'] + 1
V1_FEATS = [c for c in v1.columns if c not in ('구', '기준월', 'p', '상승여부', '향후3개월_변화율', '향후3개월_중위평당가', '월')]
v1r = v1.rename(columns={'구': '자치구'})[['자치구', 'p', '상승여부'] + [c for c in V1_FEATS if c != '구']].rename(columns={c: f'v1_{c}' for c in V1_FEATS} | {'상승여부': 'v1_상승여부'})
r = r.merge(v1r, on=['자치구', 'p'], how='left')
V1COLS = [f'v1_{c}' for c in V1_FEATS if c != '구']
# 예측시점 p에 정답을 아는 행: 공식 정답은 p'+3월 1일에 공표된 지수로 정해지므로 p' + 3 ≤ p,
# 중위 평당가 정답(V1)은 p'+2월 거래가 확정돼야 함 (p' ≤ p-4)
wk = pd.read_csv(ROOT + 'data/external/rone_weekly/apt_sale_index_weekly_gu.csv', dtype={'주': str}); wk['주'] = pd.to_datetime(wk['주'])
def label_known(p_prime, p):
    return p_prime <= p - 4 if TARGET == '상승여부' else p_prime + 3 <= p
LABEL_AVAIL = {pp: min([q for q in pd.period_range('2021-01', '2027-12', freq='M') if label_known(pp, q)]) for pp in r['p'].unique()}
r['정답공개'] = r['p'].map(LABEL_AVAIL)
LABEL_LAG = 4
y = r['상승여부'].to_numpy()

def metrics(yt, pr, months=None):
    out = {'n': len(yt), '상승비율': yt.mean(), 'ROC_AUC': roc_auc_score(yt, pr) if len(set(yt)) > 1 else np.nan,
           'PR_AUC': average_precision_score(yt, pr), 'Brier': brier_score_loss(yt, pr), 'LogLoss': log_loss(yt, np.clip(pr, 1e-4, 1 - 1e-4)),
           'F1@0.5': f1_score(yt, pr >= 0.5, zero_division=0), 'Precision@0.5': precision_score(yt, pr >= 0.5, zero_division=0),
           'Recall@0.5': recall_score(yt, pr >= 0.5, zero_division=0)}
    if months is not None:  # 같은 달 안에서 구 순위를 맞추는지 (월별 AUC 평균)
        a = [roc_auc_score(yt[months == m], pr[months == m]) for m in np.unique(months) if len(set(yt[months == m])) > 1]
        out['월내_AUC'] = float(np.mean(a)) if a else np.nan
    return out

def walk_forward(make_model, feats, test_months, window=None, frame=None):
    # 매달 p 시작 시점에 정답을 아는 행(p' ≤ p-4)으로 다시 학습해 p월 25개 구를 예측
    frame = r if frame is None else frame
    preds = []
    for p in test_months:
        trm = frame['정답공개'] <= p
        if window: trm &= frame['p'] > p - LABEL_LAG - window
        tem = frame['p'] == p
        if tem.sum() == 0: continue
        m = make_model(); m.fit(frame.loc[trm, feats], frame.loc[trm, '상승여부'])
        preds.append(pd.DataFrame({'p': p, '자치구': frame.loc[tem, '자치구'].to_numpy(), 'y': frame.loc[tem, '상승여부'].to_numpy(),
                                   'prob': m.predict_proba(frame.loc[tem, feats])[:, 1]}))
    return pd.concat(preds, ignore_index=True)
DEV = pd.period_range('2023-11', '2025-12', freq='M')
TEST = pd.period_range('2026-01', '2026-06' if TARGET != '상승여부' else '2026-04', freq='M')
