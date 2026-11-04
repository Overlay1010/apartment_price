# 31 2단계: 공식 지수 정답(상승세 +1% / 강한상승 +3%) — 기준선 · Feature 묶음 · 모델 · 학습 창 (매달 재학습 백테스트)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
rows = []
def add(name, pred, note=''):
    m = metrics(pred['y'].to_numpy(), pred['prob'].to_numpy(), pred['p'].astype(str).to_numpy()); m.update({'정답': TARGET, '구성': name, '비고': note}); rows.append(m)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in ('구성', 'ROC_AUC', '월내_AUC', 'PR_AUC', 'Brier', 'F1@0.5')}, flush=True)
    return pred
class Const:
    def fit(self, X, yy): self.p = yy.mean(); return self
    def predict_proba(self, X): return np.c_[1 - np.full(len(X), self.p), np.full(len(X), self.p)]
dev = r[r['p'].isin(DEV)]
add('기준선: 학습 기간 상승 비율', walk_forward(Const, G['공식지수'][:1], DEV))
for col in ['공식지수_4주변화', '공식지수_13주변화']:
    pr = dev[['p', '상승여부', col]].rename(columns={'상승여부': 'y', col: 'prob'}); pr['prob'] = pr['prob'].rank(pct=True)
    m = metrics(pr['y'].to_numpy(), pr['prob'].to_numpy(), pr['p'].astype(str).to_numpy()); m.update({'정답': TARGET, '구성': f'기준선: {col} 순위', '비고': '순위 점수 — Brier/F1 의미 없음'}); rows.append(m)
    print(m['구성'], round(m['ROC_AUC'], 3), round(m['월내_AUC'], 3))
def logit(C=0.1): return lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def lgbm(): return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=42)
add('기준선: 공식지수만 로지스틱', walk_forward(logit(), G['공식지수'], DEV))
CORE = G['시장(확정월)'] + G['전월세(확정월)'] + G['금리']
O = G['공식지수']
LADDER = {'V1 Feature (당월 거래)': V1COLS, 'V1 정의 + 확정월': CORE, '공식지수 + 확정월': O + CORE, '+ 시장지표(01)': O + CORE + G['시장지표(01)'],
          '+ 정책(03)': O + CORE + G['시장지표(01)'] + G['정책(03)'], '+ 공급(04)': O + CORE + G['시장지표(01)'] + G['정책(03)'] + G['공급(04)'],
          '+ 인구(05)': O + CORE + G['시장지표(01)'] + G['정책(03)'] + G['공급(04)'] + G['인구(05)'],
          '+ 회귀 OOF Gap (전체)': O + CORE + G['시장지표(01)'] + G['정책(03)'] + G['공급(04)'] + G['인구(05)'] + G['회귀 OOF Gap'],
          '공식지수 + 회귀 OOF Gap': O + G['회귀 OOF Gap']}
for win in (None, 36, 24):
    for mname, mk in [('LightGBM', lgbm), ('로지스틱', logit())]:
        for fname, feats in LADDER.items():
            add(f'{fname} | {mname} | 창 {win or "전체"}', walk_forward(mk, feats, DEV, window=win))
pd.DataFrame(rows).to_csv(HERE + f'r31_stage2_{TARGET}.csv', index=False, encoding='utf-8-sig')
