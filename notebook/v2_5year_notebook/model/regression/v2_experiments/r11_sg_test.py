# 11 보완 후보 점검: 상권 Feature 7개를 더하면 신규 단지(학습에 없던 단지) 오차가 줄어드는가
# - Valid(2025-01~06): 전체 · 학습에 있던 단지 · 처음 보는 단지
# - 단지 홀드아웃 3회: 학습 단지 20%를 통째로 빼고 학습 → 그 단지들의 Valid 거래로 평가 (신규 단지 상황 재현)
import os, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor, early_stopping
sg = pd.read_parquet(S + 'sg_feat.parquet'); SG = [c for c in sg.columns if c.startswith('상권_')]
X = X.join(t[['transaction_id']].merge(sg, on='transaction_id', how='left')[SG].set_axis(X.index))
assert X[SG].notna().all().all()
P = json.load(open(S + 'r11_best_params_lasttrade.json', encoding='utf-8'))
def fit(feats, trm, esm, seed=42):
    m = LGBMRegressor(**P['lgb'], n_estimators=6000, learning_rate=0.06, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    m.fit(X.loc[trm, feats], y[trm], eval_set=[(X.loc[esm, feats], y[esm])], callbacks=[early_stopping(100, verbose=False)])
    return m
def score(mask, p):
    a = ape(mask, p); return {'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(mask.sum())}
broker = (t['거래유형'].eq('중개거래') if '거래유형' in t else pd.Series(True, index=t.index)).to_numpy()
rows = []
for name, feats in [('현재 11 (199)', FEATS), ('+ 상권 7개', FEATS + SG)]:
    st = time.time(); m = fit(feats, tr, va); p = m.predict(X.loc[va, feats]); pv = pd.Series(p, index=np.where(va)[0])
    r = {'구성': name}
    for lab, mk in [('Valid 전체', va & broker), ('Valid 기존 단지', va & broker & in_train), ('Valid 처음 보는 단지', va & broker & ~in_train)]:
        s = score(mk, pv.loc[np.where(mk)[0]].to_numpy()); r[f'{lab} MAPE'] = s['MAPE']; r[f'{lab} n'] = s['n']
    hs = []
    for k, (trh, evh, esh) in HOLD.items():
        mh = fit(feats, trh, esh, seed=1); evb = evh & broker; hs.append(score(evb, mh.predict(X.loc[evb, feats])))
    r['홀드아웃 단지 MAPE'] = np.mean([h['MAPE'] for h in hs]); r['홀드아웃 단지 10%이내'] = np.mean([h['10%이내'] for h in hs])
    r['홀드아웃 회차별'] = [round(h['MAPE'], 3) for h in hs]; r['초'] = round(time.time() - st)
    if name.startswith('+'):
        imp = pd.Series(m.booster_.feature_importance('gain'), index=feats); r['상권 gain 순위'] = imp.rank(ascending=False)[SG].astype(int).to_dict()
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(S + 'r11_sg_test.csv', index=False, encoding='utf-8-sig')
