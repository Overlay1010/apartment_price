# 보완 후 LightGBM 재탐색: 직전 거래 Feature 포함 기본 Feature(레지스트리), 시간 fold 2개, lr 0.1
import os
import sys, time, json, optuna
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
N_TRIALS = int(sys.argv[1])
FOLDS2 = [FOLDS['f1'], FOLDS['f2']]
def objective(trial):
    params = dict(num_leaves=trial.suggest_int('num_leaves', 31, 1023, log=True),
                  min_child_samples=trial.suggest_int('min_child_samples', 5, 300, log=True),
                  colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                  subsample=trial.suggest_float('subsample', 0.5, 1.0),
                  reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 50, log=True),
                  reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                  cat_smooth=trial.suggest_float('cat_smooth', 1, 100, log=True),
                  max_bin=trial.suggest_categorical('max_bin', [255, 511]))
    scores = []
    for k, (trm, evm) in enumerate(FOLDS2):
        st = time.time()
        m = LGBMRegressor(n_estimators=8000, learning_rate=0.1, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **params)
        m.fit(X[trm], y[trm], eval_set=[(X[evm], y[evm])], callbacks=[lgb_es(100, verbose=False), log_evaluation(0)])
        scores.append(mape(evm, m.predict(X[evm])))
        trial.set_user_attr(f'fold{k+1}_iter', int(m.best_iteration_)); trial.set_user_attr(f'fold{k+1}_sec', round(time.time() - st))
        trial.report(scores[-1], k)
        if k == 0 and trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr('fold_mape', scores)
    return float(np.mean(scores))
study = optuna.create_study(study_name='r11_lgb_lasttrade', storage=f'sqlite:///{S}r11_optuna.db', load_if_exists=True, direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=8), pruner=optuna.pruners.MedianPruner(n_startup_trials=6))
if len(study.trials) == 0:  # 보완 전 최적 설정을 첫 시도로
    study.enqueue_trial(json.load(open(S + 'r11_best_params.json'))['lgb'])
left = N_TRIALS - len([x for x in study.trials if x.state.is_finished()])
study.optimize(objective, n_trials=max(left, 0), callbacks=[lambda s, t_: print(t_.number, t_.state.name, t_.value, t_.user_attrs, flush=True)])
print('BEST', study.best_value, study.best_params, study.best_trial.user_attrs)
