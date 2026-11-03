# 31 점검: 구 고정 효과(구 이름 원-핫)를 더하면 나아지는가 — 상권 · 학군처럼 천천히 변하는 구 특성이 줄 수 있는 효과의 상한
# 구 이름은 "그 구가 늘 더(덜) 오른다"를 통째로 담으므로, 이게 안 통하면 느리게 변하는 구 특성 Feature도 기대하기 어렵다
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
GU = pd.get_dummies(r['자치구'], prefix='구').astype(float); r = pd.concat([r, GU], axis=1); GUCOLS = GU.columns.tolist()
chg = r['공식_향후3개월_변화율']
LABELS = {'상승세(+1%)': r['상승여부'].copy(),
          '상대 상위 1/3': (chg.groupby(r['p']).rank(ascending=False, method='first') <= 8).astype(int)}
def logit(C): return lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, max_iter=3000))
OFF = G['공식지수']
SETS = {'공식지수': OFF, '공식지수 + 구 이름': OFF + GUCOLS, '구 이름만': GUCOLS}
rows = []
for lname, lab in LABELS.items():
    r['상승여부'] = lab
    for period, months in [('DEV', DEV), ('Test', TEST)]:
        for sname, feats in SETS.items():
            for C in (0.1, 1.0):
                pr = walk_forward(logit(C), feats, months)
                m = metrics(pr['y'].to_numpy(), pr['prob'].to_numpy(), pr['p'].astype(str).to_numpy())
                rows.append({'정답': lname, '기간': period, 'Feature': sname, 'C': C, 'ROC_AUC': m['ROC_AUC'], '월내_AUC': m['월내_AUC']})
                print(lname, period, sname, C, round(m['ROC_AUC'], 3), round(m['월내_AUC'], 3), flush=True)
pd.DataFrame(rows).to_csv(HERE + 'r31_gu_effect.csv', index=False, encoding='utf-8-sig')
