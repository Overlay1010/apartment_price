# 3단계: 최적 설정 재학습 (최종 학습률) → fold · 홀드아웃 예측 저장 (앙상블 비교용)
import os, time, json, optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
from catboost import CatBoostRegressor
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read())
P = S + 'r11_preds/'; os.makedirs(P, exist_ok=True)
BEST = {m: optuna.load_study(study_name=f'r11_{m}', storage=f'sqlite:///{S}r11_optuna.db').best_params for m in ['xgb', 'lgb', 'cat']}
json.dump(BEST, open(S + 'r11_best_params.json', 'w'), indent=1)
Xc = X.copy()
for c in CAT: Xc[c] = Xc[c].astype(str)
LR = {'xgb': 0.03, 'lgb': 0.03, 'cat': 0.1}

def fit(model, trm, esm, seed, target):
    yy = y - PY if target == 'py' else y
    if model == 'xgb':
        m = XGBRegressor(n_estimators=20000, learning_rate=LR[model], tree_method='hist', enable_categorical=True,
                         early_stopping_rounds=200, n_jobs=8, random_state=seed, **BEST[model])
        m.fit(X[trm], yy[trm], eval_set=[(X[esm], yy[esm])], verbose=False); it = m.best_iteration; pr = lambda M: m.predict(X[M])
    elif model == 'lgb':
        m = LGBMRegressor(n_estimators=20000, learning_rate=LR[model], n_jobs=8, random_state=seed, verbose=-1, bagging_freq=1, **BEST[model])
        m.fit(X[trm], yy[trm], eval_set=[(X[esm], yy[esm])], callbacks=[lgb_es(200, verbose=False), log_evaluation(0)])
        it = m.best_iteration_; pr = lambda M: m.predict(X[M])
    else:
        m = CatBoostRegressor(iterations=20000, learning_rate=LR[model], thread_count=8, random_seed=seed, verbose=0,
                              od_type='Iter', od_wait=200, cat_features=CAT, **BEST[model])
        m.fit(Xc[trm], yy[trm], eval_set=(Xc[esm], yy[esm]), use_best_model=True); it = m.get_best_iteration(); pr = lambda M: m.predict(Xc[M])
    off = PY if target == 'py' else 0
    return it, lambda M: pr(M) + (off[M] if target == 'py' else 0)

JOBS = []  # (이름, 모델, 시드, 정답, 분할 목록)
JOBS += [('lgb_s42', 'lgb', 42, 'log', ['f1', 'f2', 'h1', 'h2', 'h3'])]
JOBS += [('xgb_s42', 'xgb', 42, 'log', ['f1', 'f2', 'h1', 'h2', 'h3'])]
JOBS += [(f'lgb_s{s}', 'lgb', s, 'log', ['f1', 'f2', 'h1', 'h2', 'h3']) for s in (43, 44)]
JOBS += [('lgb_py_s42', 'lgb', 42, 'py', ['f1', 'f2', 'h1', 'h2', 'h3'])]
JOBS += [('cat_s42', 'cat', 42, 'log', ['f1', 'f2', 'h1', 'h2', 'h3'])]
log = []
for name, model, seed, target, splits in JOBS:
    for sp in splits:
        f = P + f'{name}_{sp}.npy'
        if os.path.exists(f): continue
        st = time.time()
        if sp in FOLDS: trm, evm = FOLDS[sp]; esm = evm
        else: trm, evm, esm = HOLD[sp]
        it, pr = fit(model, trm, esm, seed, target)
        np.save(f, pr(evm))
        if sp == 'f2': np.save(P + f'{name}_test.npy', pr(te))  # Test는 최종 선택 뒤에만 채점
        r = dict(job=name, split=sp, iter=int(it), mape=mape(evm, np.load(f)), sec=round(time.time() - st)); print(r, flush=True); log.append(r)
        with open(S + 'r11_refit_log.jsonl', 'a', encoding='utf-8') as fh: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
