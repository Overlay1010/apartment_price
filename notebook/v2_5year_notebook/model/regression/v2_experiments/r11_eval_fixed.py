# 11 보완 후보 재평가: 본 모델과 같은 설정(학습률 0.03, 트리 수 고정 = 튜닝 best_iteration)으로 상권 · 배정초 · K-apt를 다시 비교
# - 이전 점검(r11_sg_test · r11_hakgu_test · r11_kapt_test)은 학습률 0.06 + 평가 구간 조기 종료라 트리 360~400개에서 멈췄고,
#   시드만 바꿔도 f1 0.4%p · 홀드아웃 0.9%p가 흔들려(r11_noise_check) 0.1%p 채택 규칙을 판정할 수 없었다
# - 평가 데이터로 조기 종료하지 않는다 (평가 구간을 보고 멈추는 것 자체가 낙관 편향)
# 실행: python r11_eval_fixed.py [시드 ...]   (기본 시드 42 43)
import os, sys, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor
P = json.load(open(S + 'r11_best_params_lasttrade.json', encoding='utf-8'))
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43]
broker = t['거래유형'].eq('중개거래').to_numpy()
EXTRA = {}
for name, path, prefix in [('상권', 'sg_feat.parquet', '상권_'), ('배정초', 'hakgu_feat.parquet', '배정초_'), ('K-apt', 'kapt_feat.parquet', 'K_'),
                           ('시세보정', 'idx_feat.parquet', '지수_')]:
    f = pd.read_parquet(S + path); cols = [c for c in f.columns if c.startswith(prefix) and c != '배정초_학교명']
    X = X.join(t[['transaction_id']].merge(f, on='transaction_id', how='left')[cols].set_axis(X.index))
    for c in cols:
        if X[c].dtype == object: X[c] = X[c].astype('category')
    EXTRA[name] = cols
CONFIGS = {'현재 11': FEATS, **{f'+ {k} {len(v)}개': FEATS + v for k, v in EXTRA.items()}}
EVALS = {'f1': FOLDS['f1'], 'f2': FOLDS['f2'], **{k: (trh, evh) for k, (trh, evh, _) in HOLD.items()}}
def fit(feats, trm, seed):
    m = LGBMRegressor(**P['lgb'], n_estimators=P['best_iteration'], learning_rate=P['learning_rate'], subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    return m.fit(X.loc[trm, feats], y[trm])
out_path = S + 'r11_eval_fixed.csv'
done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['평가', '구성', '시드'])
rows = done.to_dict('records')
for ev, (trm, evm) in EVALS.items():
    evb = evm & broker
    for seed in SEEDS:
        for name, feats in CONFIGS.items():
            if ((done['평가'] == ev) & (done['구성'] == name) & (done['시드'] == seed)).any(): continue
            st = time.time(); a = ape(evb, fit(feats, trm, seed).predict(X.loc[evb, feats]))
            rows.append({'평가': ev, '구성': name, '시드': seed, 'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(evb.sum()), '초': round(time.time() - st)})
            print(rows[-1], flush=True); pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
d = pd.DataFrame(rows)
print(d.pivot_table(index='구성', columns='평가', values='MAPE', aggfunc='mean').round(3).to_string())
print('시드 간 차이(최대-최소):'); print(d.groupby(['평가', '구성'])['MAPE'].agg(lambda s: s.max() - s.min()).unstack().round(3).to_string())
