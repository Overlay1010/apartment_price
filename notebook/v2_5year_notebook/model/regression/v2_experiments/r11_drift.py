# Test 과소추정 진단: (1) Train+Valid 재학습  (2) 시장 수준 대비 상대 정답  (3) 둘 다
import os
import json, time
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read())
BP = json.load(open(S + 'r11_best_params.json'))['lgb']
OFF = {'abs': np.zeros(len(t)),
       '구평당가': np.log(t['구_확정월_중위평당가'].to_numpy(float)) + PY,
       '단지예상가': np.log(t['예상가격_단지만'].fillna(t['구_확정월_중위평당가'] * t['전용면적'] * 0.3025).to_numpy(float))}
def model(n=20000): return LGBMRegressor(n_estimators=n, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP)
def report(name, p_va, p_te):
    r = {'구성': name}
    for nm, m, p in [('Valid', va, p_va), ('Test', te, p_te)]:
        if p is None: continue
        r[f'{nm}_MAPE'] = mape(m, p); r[f'{nm}_편향'] = float(((np.exp(p) / np.exp(y[m])) - 1).mean() * 100)
    print(r, flush=True); return r
rows = []
for k, off in OFF.items():
    if np.isnan(off).any(): print(k, 'NaN', np.isnan(off).sum())
    yy = y - off; st = time.time()
    m = model(); m.fit(X[tr], yy[tr], eval_set=[(X[va], yy[va])], callbacks=[lgb_es(200, verbose=False), log_evaluation(0)])
    it = m.best_iteration_
    rows.append(report(f'{k} | Train 학습', m.predict(X[va]) + off[va], m.predict(X[te]) + off[te]))
    m2 = model(int(it * 1.15)); m2.fit(X[tr | va], yy[tr | va])
    rows.append(report(f'{k} | Train+Valid 재학습 ({int(it*1.15)}회)', None, m2.predict(X[te]) + off[te]))
    print(k, round(time.time() - st), flush=True)
    pd.DataFrame(rows).to_csv(S + 'r11_drift.csv', index=False, encoding='utf-8-sig')
