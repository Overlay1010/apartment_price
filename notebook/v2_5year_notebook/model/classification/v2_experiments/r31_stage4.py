# 31 4단계: 상대 정답 — 같은 달 25개 구 중 향후 3개월 상승률 상위 1/3(많이 오를 구)을 맞히는가 (DEV · Test)
# 시장 전체가 오르든 내리든 매달 정답 비율이 1/3로 같아 2026년처럼 거의 모든 구가 오른 기간도 평가할 수 있다
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
TOP = 1 / 3
chg = r['공식_향후3개월_변화율']
r['상승여부'] = (chg.groupby(r['p']).rank(ascending=False, method='first') <= round(25 * TOP)).astype(int)
# 구끼리 비교하는 문제라 Feature도 같은 달 서울 평균과의 차이로 바꾼 묶음을 함께 본다 (서울 공통값은 0이 돼 빠진다)
BASE = G['공식지수'] + G['시장(확정월)'] + G['전월세(확정월)'] + G['금리'] + G['시장지표(01)'] + G['공급(04)'] + G['인구(05)']
num = [c for c in dict.fromkeys(BASE + G['회귀 OOF Gap']) if pd.api.types.is_numeric_dtype(r[c])]
for c in num:
    r[f'상대_{c}'] = r[c] - r.groupby('p')[c].transform('mean')
rel = lambda cols: [f'상대_{c}' for c in cols if c in num and r[f'상대_{c}'].std() > 0]
def logit(C=1.0): return lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def lgbm(): return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=42)

def rank_metrics(pr):
    # 월마다: 순위 상관(예측 점수 ↔ 실제 상승률), 모델 상위 8개 중 실제 상위 8개 적중 수, 월내 AUC
    pr = pr.merge(r[['p', '자치구', '공식_향후3개월_변화율']], on=['p', '자치구'])
    sp, hit, auc, spread = [], [], [], []
    for _, g in pr.groupby('p'):
        sp.append(spearmanr(g['prob'], g['공식_향후3개월_변화율'])[0])
        top = g.nlargest(8, 'prob'); bot = g.nsmallest(8, 'prob')
        hit.append(top['y'].sum() / 8)
        spread.append((top['공식_향후3개월_변화율'].mean() - bot['공식_향후3개월_변화율'].mean()) * 100)
        auc.append(roc_auc_score(g['y'], g['prob']))
    return {'월내_AUC': np.mean(auc), '순위상관': np.mean(sp), '상위8_적중률': np.mean(hit), '상위8-하위8_상승률차(%p)': np.mean(spread),
            '순위상관_양수월비율': np.mean(np.array(sp) > 0)}

def baseline(col, months):
    t = r[r['p'].isin(months)]
    return pd.DataFrame({'p': t['p'], '자치구': t['자치구'], 'y': t['상승여부'], 'prob': t[col]})

SETS = {'공식지수': rel(G['공식지수']), '공식지수 + 확정월 시장·전월세': rel(G['공식지수'] + G['시장(확정월)'] + G['전월세(확정월)']),
        '공식지수 + 확정월 + 수급(01) + 공급 + 인구': rel(BASE), '위 + 회귀 OOF Gap': rel(BASE + G['회귀 OOF Gap'])}
rows = []
for period, months in [('DEV', DEV), ('Test', TEST)]:
    cands = [(f'기준선: {c}', None, baseline(c, months)) for c in ['공식지수_4주변화', '공식지수_13주변화', '공식지수_26주변화']]
    for sname, feats in SETS.items():
        for mname, mk, win in [('로지스틱', logit(0.1), None), ('로지스틱', logit(0.1), 36), ('LightGBM', lgbm, None)]:
            cands.append((f'{mname} 창 {win or "전체"} · {sname}', len(feats), walk_forward(mk, feats, months, window=win)))
    for name, nf, pr in cands:
        m = rank_metrics(pr); m.update({'기간': period, '방법': name, 'Feature수': nf}); rows.append(m)
        print(period, name, *[f'{k} {v:.3f}' for k, v in m.items() if isinstance(v, float)], flush=True)
pd.DataFrame(rows).to_csv(HERE + 'r31_stage4_상대상위.csv', index=False, encoding='utf-8-sig')
