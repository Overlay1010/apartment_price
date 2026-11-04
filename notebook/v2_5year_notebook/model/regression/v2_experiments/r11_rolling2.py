# 보완 후 매달 재학습 평가: 직전 거래 Feature 포함, 재탐색한 LightGBM 설정
import os, json, time, optuna
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
BP = optuna.load_study(study_name='r11_lgb_lasttrade', storage=f'sqlite:///{S}r11_optuna.db').best_params
# 반복 수: Train 학습 · Valid 조기 종료(lr 0.03) 최적 반복 × 1.15 (학습 기간이 길어지는 만큼)
m = LGBMRegressor(n_estimators=20000, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP)
m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb_es(200, verbose=False), log_evaluation(0)])
N_EST = int(round(m.best_iteration_ * 1.15, -1))
print('Train→Valid', round(mape(va, m.predict(X[va])), 3), 'best_iter', m.best_iteration_, '→ N_EST', N_EST, flush=True)
json.dump({'lgb': BP, 'n_estimators': N_EST, 'learning_rate': 0.03, 'train_valid_mape': mape(va, m.predict(X[va])),
           'best_iteration': int(m.best_iteration_)}, open(S + 'r11_best_params_lasttrade.json', 'w'), ensure_ascii=False, indent=1)
P = S + 'r11_roll2/'; os.makedirs(P, exist_ok=True)
per = t['거래월일자'].dt.to_period('M')
for mth in pd.period_range('2022-01', '2026-06', freq='M'):
    f = P + f'{mth}.npy'
    if os.path.exists(f): continue
    st = time.time(); trm = (per <= mth - 2).to_numpy(); evm = (per == mth).to_numpy()
    mdl = LGBMRegressor(n_estimators=N_EST, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP).fit(X[trm], y[trm])
    np.save(f, mdl.predict(X[evm]))
    print(mth, trm.sum(), round(mape(evm, np.load(f)), 3), round(time.time() - st), flush=True)
