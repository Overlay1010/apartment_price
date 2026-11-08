# 11 보완 3 후보 비교: 단지 모든 면적 시세 보정가(단지_) · 같은 구 유사 단지 시세 보정가(유사_) — r11_nb_build.py
# - r11_eval_fixed와 같은 방식: 본 모델 설정(r11_best_params_idx, 학습률 0.03, 트리 수 고정 = best_iteration), 평가 구간 조기 종료 없음
# - 기준은 지금 11 (레지스트리 기본 201개, 시세 보정 포함). 평가는 중개거래, f1 · f2 · 단지 홀드아웃 3개 (Test는 보지 않음)
# - 채택 규칙: 두 시드 모두 f1 · f2에서 0.1%p 이상 개선, 홀드아웃 평균이 나빠지지 않을 것
# - 오차가 몰린 구간(같은 면적 직전 거래 6개월+ · 없음, 100세대 미만)도 따로 기록
# 실행: python r11_nb_test.py [시드 ...]   (기본 시드 42 43)
import os, sys, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor
P = json.load(open(S + 'r11_best_params_idx.json', encoding='utf-8'))
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43]
broker = t['거래유형'].eq('중개거래').to_numpy()
f = pd.read_parquet(S + 'nb_feat.parquet')
EXTRA = {'단지': [c for c in f.columns if c.startswith('단지_')], '유사': [c for c in f.columns if c.startswith('유사_')]}
X = X.join(t[['transaction_id']].merge(f, on='transaction_id', how='left')[EXTRA['단지'] + EXTRA['유사']].set_axis(X.index))
CONFIGS = {'현재 11': FEATS, '+ 단지 3개': FEATS + EXTRA['단지'], '+ 유사 2개': FEATS + EXTRA['유사'],
           '+ 단지 · 유사 5개': FEATS + EXTRA['단지'] + EXTRA['유사']}
EVALS = {'f1': FOLDS['f1'], 'f2': FOLDS['f2'], **{k: (trh, evh) for k, (trh, evh, _) in HOLD.items()}}
SEG = {'직전 6개월+/없음': (t['직전거래_경과일'].isna() | (t['직전거래_경과일'] > 180)).to_numpy(),
       '100세대 미만': (t['세대수'] < 100).to_numpy()}
def fit(feats, trm, seed):
    m = LGBMRegressor(**P['lgb'], n_estimators=P['best_iteration'], learning_rate=P['learning_rate'], subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    return m.fit(X.loc[trm, feats], y[trm])
out_path = S + 'r11_nb_test.csv'
done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['평가', '구성', '시드'])
rows = done.to_dict('records')
for seed in SEEDS:
    for ev, (trm, evm) in EVALS.items():
        evb = evm & broker
        for name, feats in CONFIGS.items():
            if ((done['평가'] == ev) & (done['구성'] == name) & (done['시드'] == seed)).any(): continue
            st = time.time(); a = ape(evb, fit(feats, trm, seed).predict(X.loc[evb, feats]))
            row = {'평가': ev, '구성': name, '시드': seed, 'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(evb.sum())}
            for sname, smask in SEG.items():
                sm = smask[evb]; row[f'{sname} MAPE'] = a[sm].mean(); row[f'{sname} n'] = int(sm.sum())
            row['초'] = round(time.time() - st)
            rows.append(row); print(row, flush=True); pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
d = pd.DataFrame(rows)
d['평가묶음'] = np.where(d['평가'].str.startswith('h'), '홀드아웃 평균', d['평가'])
for col in ['MAPE', '직전 6개월+/없음 MAPE', '100세대 미만 MAPE']:
    print(f'\n[{col}] 시드 평균'); print(d.pivot_table(index='구성', columns='평가묶음', values=col, aggfunc='mean').round(3).to_string())
print('\n시드 간 차이(최대-최소):'); print(d.groupby(['평가', '구성'])['MAPE'].agg(lambda s: s.max() - s.min()).unstack().round(3).to_string())
base = d[d['구성'].eq('현재 11')].set_index(['평가', '시드'])['MAPE']
diff = d[~d['구성'].eq('현재 11')].assign(개선=lambda x: x.set_index(['평가', '시드']).index.map(base) - x['MAPE'])
print('\n현재 11 대비 개선 (%p, +가 좋아짐):'); print(diff.pivot_table(index=['구성', '시드'], columns='평가', values='개선').round(3).to_string())
