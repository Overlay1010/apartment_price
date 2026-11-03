# 31 1단계: V1 재현 · 기준선 · Feature 묶음 · 모델 · 학습 창 비교 (매달 재학습 백테스트)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
rows = []
def add(name, pred, note=''):
    m = metrics(pred['y'].to_numpy(), pred['prob'].to_numpy(), pred['p'].astype(str).to_numpy()); m.update({'구성': name, '비고': note}); rows.append(m)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)

# A. V1 재현: V1 Feature(당월 거래) · V1 XGB · 18개월 rolling · V1 fold 4개 (학습 후 6개월 고정 예측)
v1d = pd.read_parquet(HERE + 'v1_cls_df.parquet'); V1F = [c for c in v1d.columns if c not in ('기준월', '상승여부', '향후3개월_변화율', '향후3개월_중위평당가', '월')]
def v1_model():
    pre = ColumnTransformer([('gu', OneHotEncoder(handle_unknown='ignore'), ['구'])], remainder='passthrough')
    return make_pipeline(pre, XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, tree_method='hist', eval_metric='logloss', random_state=42, n_jobs=8))
FOLDS = [('2023-06', '2023-10', '2024-03'), ('2023-12', '2024-04', '2024-09'), ('2024-06', '2024-10', '2025-03'), ('2024-12', '2025-04', '2025-09')]
aucs = []
for te, vs, ve in FOLDS:
    te, vs, ve = pd.Period(te, 'M'), pd.Period(vs, 'M'), pd.Period(ve, 'M')
    trd = v1d[v1d['기준월'].between(te - 17, te)]; vad = v1d[v1d['기준월'].between(vs, ve)]
    m = v1_model().fit(trd[V1F], trd['상승여부']); aucs.append(roc_auc_score(vad['상승여부'], m.predict_proba(vad[V1F])[:, 1]))
print('V1 재현 (fold 4개 평균 ROC-AUC, V1 기록 0.613):', np.round(aucs, 3), round(np.mean(aucs), 3), flush=True)
rows.append({'구성': 'V1 재현 (V1 fold 4개, 18개월)', 'ROC_AUC': np.mean(aucs), '비고': str(np.round(aucs, 3).tolist())})

# B. 매달 재학습 백테스트 (예측시점 2023-11 ~ 2025-12, 27개월 × 25구)
class Const:
    def fit(self, X, yy): self.p = yy.mean(); return self
    def predict_proba(self, X): return np.c_[1 - np.full(len(X), self.p), np.full(len(X), self.p)]
add('기준선: 학습 기간 상승 비율', walk_forward(Const, G['시장(확정월)'][:1], DEV))
for nm, col in [('기준선: 최근 확정 3개월 변화율', '구_확정월기준_실현_3개월변화율'), ('기준선: 서울 상승 구 비율', '서울_확정월기준_상승구비율')]:
    pr = r[r['p'].isin(DEV)][['p', '자치구', '상승여부', col]].rename(columns={'상승여부': 'y', col: 'prob'})
    pr['prob'] = pr['prob'].rank(pct=True)  # 순위 점수 (AUC용, 확률 아님)
    m = metrics(pr['y'].to_numpy(), pr['prob'].to_numpy(), pr['p'].astype(str).to_numpy()); m.update({'구성': nm, '비고': '순위 점수 — Brier/F1 의미 없음'}); rows.append(m); print(nm, round(m['ROC_AUC'], 3), round(m['월내_AUC'], 3))
last = r[['자치구', 'p', '상승여부']].copy(); last['p'] = last['p'] + LABEL_LAG
pr = r[r['p'].isin(DEV)][['p', '자치구', '상승여부']].merge(last.rename(columns={'상승여부': 'prob'}), on=['자치구', 'p']).rename(columns={'상승여부': 'y'})
add('기준선: 같은 구 최근 확정 정답 (p-4)', pr.assign(prob=pr['prob'] * 0.6 + 0.2))

def logit(): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
def lgbm(): return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=42)
def xgb(): return XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, tree_method='hist', eval_metric='logloss', random_state=42, n_jobs=8)
CORE = G['시장(확정월)'] + G['전월세(확정월)'] + G['금리']
ALL = sum(G.values(), [])
LADDER = {'V1 Feature (당월 거래)': V1COLS, 'V1 정의 + 확정월 (시장·전월세·금리)': CORE, '+ 시장지표(01)': CORE + G['시장지표(01)'],
          '+ 정책(03)': CORE + G['시장지표(01)'] + G['정책(03)'], '+ 공급(04)': CORE + G['시장지표(01)'] + G['정책(03)'] + G['공급(04)'],
          '+ 인구(05) = 전체': CORE + G['시장지표(01)'] + G['정책(03)'] + G['공급(04)'] + G['인구(05)'], '+ 회귀 OOF Gap': ALL}
for win in (None, 18, 36):
    for mname, mk in [('XGB(V1 설정)', xgb), ('LightGBM', lgbm), ('로지스틱', logit)]:
        for fname, feats in LADDER.items():
            if win == 36 and mname != 'LightGBM': continue
            add(f'{fname} | {mname} | 창 {win or "전체"}', walk_forward(mk, feats, DEV, window=win))
pd.DataFrame(rows).to_csv(HERE + 'r31_stage1.csv', index=False, encoding='utf-8-sig')
