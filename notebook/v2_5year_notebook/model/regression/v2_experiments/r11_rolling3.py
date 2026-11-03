# 보완 2 매달 재학습 평가: 직전 거래가 시세 보정 2개 포함 (기본 Feature 201개), LightGBM 설정은 r11_lgb_lasttrade 재탐색 결과 그대로
# - 반복 수는 r11_rolling2와 같은 방식으로 다시 정한다: Train 학습 · Valid 조기 종료(lr 0.03) 최적 반복 × 1.15
# - 2022-01~2026-06 각 월 m: m-2월까지 확정된 거래로 학습 → m월 예측 (r11_roll3/*.npy), 끝나면 OOF 표(r11_oof3.parquet)를 만든다
import os, json, time, optuna
from lightgbm import LGBMRegressor, early_stopping as lgb_es, log_evaluation
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
assert {'직전거래후_구지수변화', '직전거래가_시세보정'} <= set(FEATS), '06 재실행 전 레지스트리'
print('기본 Feature', len(FEATS), flush=True)
BP = optuna.load_study(study_name='r11_lgb_lasttrade', storage=f'sqlite:///{S}r11_optuna.db').best_params
PARAMS = S + 'r11_best_params_idx.json'
if not os.path.exists(PARAMS):
    m = LGBMRegressor(n_estimators=20000, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb_es(200, verbose=False), log_evaluation(0)])
    json.dump({'lgb': BP, 'n_estimators': int(round(m.best_iteration_ * 1.15, -1)), 'learning_rate': 0.03,
               'train_valid_mape': mape(va, m.predict(X[va])), 'best_iteration': int(m.best_iteration_)},
              open(PARAMS, 'w'), ensure_ascii=False, indent=1)
cfg = json.load(open(PARAMS, encoding='utf-8')); N_EST = cfg['n_estimators']
print('Train→Valid', round(cfg['train_valid_mape'], 3), 'best_iter', cfg['best_iteration'], '→ N_EST', N_EST, flush=True)
P = S + 'r11_roll3/'; os.makedirs(P, exist_ok=True)
per = t['거래월일자'].dt.to_period('M'); months = pd.period_range('2022-01', '2026-06', freq='M')
for mth in months:
    f = P + f'{mth}.npy'
    if os.path.exists(f): continue
    st = time.time(); trm = (per <= mth - 2).to_numpy(); evm = (per == mth).to_numpy()
    mdl = LGBMRegressor(n_estimators=N_EST, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP).fit(X[trm], y[trm])
    np.save(f, mdl.predict(X[evm]))
    print(mth, trm.sum(), round(mape(evm, np.load(f)), 3), round(time.time() - st), flush=True)

# OOF 표 (r11_interval4와 같은 열)
parts = []
for m in months:
    evm = (per == m).to_numpy(); p = np.load(P + f'{m}.npy')
    seen = t.loc[evm, '단지일련번호'].isin(set(t.loc[(per <= m - 2).to_numpy(), '단지일련번호'])).to_numpy()
    parts.append(pd.DataFrame({'transaction_id': t.loc[evm, 'transaction_id'].to_numpy(), '월': m, 'p': p, 'y': y[evm],
                               '단지': np.where(seen, '기존', '신규'), '거래유형': t.loc[evm, '거래유형'].to_numpy()}))
d = pd.concat(parts, ignore_index=True); d['r'] = d['y'] - d['p']
d['가격대'] = pd.cut(np.exp(d['p']) / 1e4, [0, 6, 9, 15, np.inf], labels=['6억 미만', '6~9억', '9~15억', '15억 이상'], right=False).astype(str)
d['ape'] = np.abs(1 - np.exp(d['p'] - d['y'])) * 100
d.to_parquet(S + 'r11_oof3.parquet')
old = pd.read_parquet(S + 'r11_oof2.parquet')  # v2.2 OOF (r11_compare_idx.py 주석 참고)
cmp = d.merge(old[['transaction_id', 'ape']], on='transaction_id', suffixes=('', '_보완전'))
cmp['구간'] = np.select([cmp['월'].astype(str).between('2025-07', '2025-12'), cmp['월'].astype(str) >= '2026-01'], ['Valid', 'Test'], '그 이전')
mk = cmp['거래유형'].eq('중개거래')
print(cmp[mk].groupby('구간')[['ape_보완전', 'ape']].mean().round(3).rename(columns={'ape_보완전': 'v2.2 MAPE', 'ape': '보완 2 MAPE'}).to_string())
