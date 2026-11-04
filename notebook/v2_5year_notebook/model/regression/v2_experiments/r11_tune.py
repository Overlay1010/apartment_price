# 2단계: Optuna 하이퍼파라미터 탐색 (시간 fold 2개, lr 0.1 고정)
import os
import sys, time, json, numpy as np, pandas as pd, optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
from catboost import CatBoostRegressor, Pool
sys.stdout.reconfigure(encoding='utf-8')
S=os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
exec(open(S+'r11_stage1.py',encoding='utf-8').read().split('def fit(')[0])
MODEL = sys.argv[1]; N_TRIALS = int(sys.argv[2])
FEATS = json.load(open(S+'r11_features.json',encoding='utf-8'))
t['예상가격_단지만'] = t['예상가격_만원'].where(t['예상가격_출처'].eq('단지'))
X = frame(FEATS)
CAT = [c for c in FEATS if str(X[c].dtype) == 'category']
mon = t['거래월일자']
FOLDS = [((mon < '2025-01-01').to_numpy(), (tr & (mon >= '2025-01-01')).to_numpy()),
         (tr, va)]
if MODEL == 'cat':
    Xc = X.copy()
    for c in CAT: Xc[c] = Xc[c].astype(str).fillna('NA')

def mape(mask, p): a = np.exp(y[mask]); return float((np.abs(a - np.exp(p)) / a).mean() * 100)

def fit_one(params, trm, evm, lr=0.1, seed=42):
    if MODEL == 'xgb':
        m = XGBRegressor(n_estimators=8000, learning_rate=lr, tree_method='hist', enable_categorical=True,
                         early_stopping_rounds=100, n_jobs=8, random_state=seed, **params)
        m.fit(X[trm], y[trm], eval_set=[(X[evm], y[evm])], verbose=False)
        return m, m.predict(X[evm]), m.best_iteration
    if MODEL == 'lgb':
        m = LGBMRegressor(n_estimators=8000, learning_rate=lr, n_jobs=8, random_state=seed, verbose=-1,
                          bagging_freq=1, **params)
        m.fit(X[trm], y[trm], eval_set=[(X[evm], y[evm])], callbacks=[lgb_es(100, verbose=False), log_evaluation(0)])
        return m, m.predict(X[evm]), m.best_iteration_
    m = CatBoostRegressor(iterations=8000, learning_rate=lr, thread_count=8, random_seed=seed, verbose=0,
                          od_type='Iter', od_wait=100, cat_features=CAT, **params)
    m.fit(Xc[trm], y[trm], eval_set=(Xc[evm], y[evm]), use_best_model=True)
    return m, m.predict(Xc[evm]), m.get_best_iteration()

def space(tr_):
    if MODEL == 'xgb':
        return dict(max_depth=tr_.suggest_int('max_depth', 4, 12),
                    min_child_weight=tr_.suggest_float('min_child_weight', 1, 100, log=True),
                    subsample=tr_.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=tr_.suggest_float('colsample_bytree', 0.3, 1.0),
                    reg_lambda=tr_.suggest_float('reg_lambda', 1e-3, 50, log=True),
                    reg_alpha=tr_.suggest_float('reg_alpha', 1e-3, 10, log=True),
                    max_cat_to_onehot=tr_.suggest_int('max_cat_to_onehot', 1, 32, log=True))
    if MODEL == 'lgb':
        return dict(num_leaves=tr_.suggest_int('num_leaves', 31, 1023, log=True),
                    min_child_samples=tr_.suggest_int('min_child_samples', 5, 300, log=True),
                    colsample_bytree=tr_.suggest_float('colsample_bytree', 0.3, 1.0),
                    subsample=tr_.suggest_float('subsample', 0.5, 1.0),
                    reg_lambda=tr_.suggest_float('reg_lambda', 1e-3, 50, log=True),
                    reg_alpha=tr_.suggest_float('reg_alpha', 1e-3, 10, log=True),
                    cat_smooth=tr_.suggest_float('cat_smooth', 1, 100, log=True),
                    max_bin=tr_.suggest_categorical('max_bin', [255, 511]))
    return dict(depth=tr_.suggest_int('depth', 6, 10),
                l2_leaf_reg=tr_.suggest_float('l2_leaf_reg', 1, 50, log=True),
                random_strength=tr_.suggest_float('random_strength', 0.1, 5, log=True),
                bagging_temperature=tr_.suggest_float('bagging_temperature', 0, 1),
                rsm=tr_.suggest_float('rsm', 0.3, 1.0),
                one_hot_max_size=tr_.suggest_int('one_hot_max_size', 2, 32, log=True))

def objective(trial):
    params = space(trial); scores = []
    for k, (trm, evm) in enumerate(FOLDS):
        st = time.time(); _, p, it = fit_one(params, trm, evm)
        scores.append(mape(evm, p)); trial.set_user_attr(f'fold{k+1}_iter', int(it))
        trial.set_user_attr(f'fold{k+1}_sec', round(time.time() - st))
        trial.report(scores[-1], k)
        if k == 0 and trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr('fold_mape', scores)
    return float(np.mean(scores))

study = optuna.create_study(study_name=f'r11_{MODEL}', storage=f'sqlite:///{S}r11_optuna.db', load_if_exists=True,
                            direction='minimize', sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=6))
if MODEL == 'xgb' and len(study.trials) == 0:  # 1단계 기본 설정을 출발점으로
    study.enqueue_trial(dict(max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1, reg_alpha=1e-3, max_cat_to_onehot=1))
remaining = N_TRIALS - len([x for x in study.trials if x.state.is_finished()])
study.optimize(objective, n_trials=max(remaining, 0),
               callbacks=[lambda s, tr_: print(tr_.number, tr_.state.name, tr_.value, tr_.params, tr_.user_attrs, flush=True)])
print('BEST', study.best_value, study.best_params, study.best_trial.user_attrs)
