# 11 보완 4 후보 비교: 호별 공시가격 층별 Feature — r11_unit_build.py
# - r11_nb_test와 같은 설정 · 평가 · 채택 규칙 (두 시드 모두 f1 · f2에서 0.1%p 이상 개선, 홀드아웃 평균이 나빠지지 않을 것), '현재 11' 결과는 r11_nb_test.csv를 그대로 쓴다
# - Test는 보지 않는다
# 실행: python r11_unit_test.py [시드 ...]   (기본 시드 42 43)
import os, sys, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor
P = json.load(open(S + 'r11_best_params_idx.json', encoding='utf-8'))
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 43]
broker = t['거래유형'].eq('중개거래').to_numpy()
f = pd.read_parquet(S + 'unit_feat.parquet')
NEW = [c for c in f.columns if c != 'transaction_id']
X = X.join(t[['transaction_id']].merge(f, on='transaction_id', how='left', validate='one_to_one')[NEW].set_axis(X.index))
CONFIGS = {'+ 호공시 층값 2개': FEATS + ['호공시_층별m2당', '호공시_층프리미엄'], '+ 호공시 4개': FEATS + NEW}
EVALS = {'f1': FOLDS['f1'], 'f2': FOLDS['f2'], **{k: (trh, evh) for k, (trh, evh, _) in HOLD.items()}}
SEG = {'직전 6개월+/없음': (t['직전거래_경과일'].isna() | (t['직전거래_경과일'] > 180)).to_numpy(),
       '100세대 미만': (t['세대수'] < 100).to_numpy()}
def fit(feats, trm, seed):
    m = LGBMRegressor(**P['lgb'], n_estimators=P['best_iteration'], learning_rate=P['learning_rate'], subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    return m.fit(X.loc[trm, feats], y[trm])
out_path = S + 'r11_unit_test.csv'
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
b = pd.read_csv(S + 'r11_nb_test.csv'); d = pd.concat([b[b['구성'].eq('현재 11')], pd.DataFrame(rows)], ignore_index=True)
print('\n[MAPE] 시드 평균'); print(d.pivot_table(index='구성', columns='평가', values='MAPE').round(3).to_string())
base = d[d['구성'].eq('현재 11')].set_index(['평가', '시드'])['MAPE']
x = d[~d['구성'].eq('현재 11')].copy(); x['개선'] = base.loc[list(zip(x['평가'], x['시드']))].to_numpy() - x['MAPE'].to_numpy()
print('\n현재 11 대비 개선 (%p, +가 좋아짐):'); print(x.pivot_table(index=['구성', '시드'], columns='평가', values='개선').round(3).to_string())
print('\n시드 간 차이:'); print(d.groupby(['평가', '구성'])['MAPE'].agg(lambda s: s.max() - s.min()).unstack().round(3).to_string())
