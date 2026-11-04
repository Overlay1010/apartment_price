# 31 3단계: 단순 모델 — 최근 흐름(공식 4주 변화율)만으로 만든 확률 vs 작은 Feature 조합 (DEV · Test)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
def logit(C=1.0): return lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
def lgbm(): return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=42)
SETS = {'4주 변화율만': ['공식지수_4주변화'], '4주 + 13주': ['공식지수_4주변화', '공식지수_13주변화'],
        '4주 + 서울 흐름': ['공식지수_4주변화', '공식지수_서울_13주변화', '공식지수_4주상승구비율'],
        '4주 + 확정월 구 시장': ['공식지수_4주변화'] + G['시장(확정월)'],
        '4주 + 확정월 구 시장 + 금리': ['공식지수_4주변화'] + G['시장(확정월)'] + G['금리'],
        '4주 + 수급·심리(01)': ['공식지수_4주변화'] + G['시장지표(01)'],
        '공식지수 + 확정월 (2단계 최고 묶음)': G['공식지수'] + G['시장(확정월)'] + G['전월세(확정월)'] + G['금리']}
rows = []
for period, months in [('DEV', DEV), ('Test', TEST)]:
    for sname, feats in SETS.items():
        for mname, mk, win in [('로지스틱', logit(1.0), None), ('로지스틱', logit(0.1), 36), ('LightGBM', lgbm, None)]:
            if mname == 'LightGBM' and len(feats) < 3: continue
            pr = walk_forward(mk, feats, months, window=win)
            m = metrics(pr['y'].to_numpy(), pr['prob'].to_numpy(), pr['p'].astype(str).to_numpy())
            m.update({'정답': TARGET, '기간': period, 'Feature': sname, '모델': f'{mname} 창 {win or "전체"}'}); rows.append(m)
            print(period, sname, m['모델'], round(m['ROC_AUC'], 3), round(m['월내_AUC'], 3), round(m['Brier'], 3), round(m['F1@0.5'], 3), flush=True)
pd.DataFrame(rows).to_csv(HERE + f'r31_stage3_{TARGET}.csv', index=False, encoding='utf-8-sig')
