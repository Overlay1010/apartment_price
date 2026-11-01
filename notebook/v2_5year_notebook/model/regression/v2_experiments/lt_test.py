import os
import json, time
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
BP = json.load(open(S + 'r11_best_params.json'))['lgb']
f = pd.read_parquet(S + 'lt_feat.parquet')  # lt_feature.py 프로토타입 결과 (06 반영 전); NEW = [c for c in f.columns if c != 'transaction_id']
t2 = t[['transaction_id']].merge(f, on='transaction_id', how='left', validate='one_to_one')
for c in NEW: X[c] = t2[c].to_numpy(float)
mk = (t['거래유형'] == '중개거래').to_numpy()
for name, feats in [('기존 194', FEATS), ('+ 직전 거래 5개', FEATS + NEW)]:
    st = time.time()
    m = LGBMRegressor(n_estimators=20000, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP)
    m.fit(X.loc[tr, feats], y[tr], eval_set=[(X.loc[va, feats], y[va])], callbacks=[lgb_es(200, verbose=False), log_evaluation(0)])
    p = m.predict(X.loc[va, feats]); a = pd.Series(ape(va, p)); nv = (~in_train)[va]; mm = mk[va]
    print(name, 'iter', m.best_iteration_, 'Valid', round(a.mean(), 3), '중개', round(a[mm].mean(), 3), '중개 중앙', round(a[mm].median(), 3),
          '신규', round(a[nv].mean(), 2), round(time.time() - st), '초', flush=True)
