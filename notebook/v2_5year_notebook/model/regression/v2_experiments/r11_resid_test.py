# V3 추가 시험: 단지 최근 모델 오차(OOF 잔차) Feature 2개 — r11_resid_build.py
# - V3 범위(설계서 Phase 4-3) 밖의 1회 추가 시험 (2026-11-12 사용자 결정). 통과하지 못하면 V3를 모델 유지로 닫는다
# - 설정 · 평가 · 채택 규칙은 보완 5(r11_method_test.py)와 같다 (결과를 보기 전에 고정, 2026-11-12):
#   시간 fold 4개(f0a 2024 상반기 · f0b 2024 하반기 · f1 2025 상반기 · f2 2025 하반기) + 단지 홀드아웃 3개, 시드 42 · 43 · 44, 중개거래, Test 미사용
#   (1) 시간 fold 4개 × 시드 3개 평균 개선 ≥ 0.1%p  (2) fold별 시드 평균 개선 > 0  (3) 시드별 fold 평균 개선 > 0  (4) 홀드아웃 평균 개선 ≥ 0
# - '현재 11' 결과는 r11_method_test.csv(같은 코드 · 설정)를 그대로 쓴다
# - 홀드아웃 주의: 잔차는 운영 OOF(그 단지를 학습한 이전 달 모델)에서 왔다. 홀드아웃 단지도 자기 과거 거래 정보(직전 거래가 등)를 쓰는 기존 Feature와 같은 성격이라 조건 (4)는 "나빠지지 않음"만 본다
# 실행: python r11_resid_test.py   (결과 CSV에 이어 쓰기)
# 결과 (2026-11-12): 시간 fold 평균 +0.084%p, f0a −0.007 · f2 +0.020 → 불채택 (r11_resid_test_output.txt)
import os, sys, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor
P = json.load(open(S + 'r11_best_params_idx.json', encoding='utf-8'))
SEEDS = [42, 43, 44]
broker = t['거래유형'].eq('중개거래').to_numpy()
f = pd.read_parquet(S + 'resid_feat.parquet')
NEW = ['단지_최근잔차', '단지_최근잔차_건수']
X = X.join(t[['transaction_id']].merge(f, on='transaction_id', how='left', validate='one_to_one')[NEW].set_axis(X.index))
CFG = '+ 단지 최근 잔차 2개'


def time_fold(start, end):
    return (tr & (mon < start)).to_numpy(), (tr & (mon >= start) & (mon < end)).to_numpy()


TIME = {'f0a': time_fold('2024-01-01', '2024-07-01'), 'f0b': time_fold('2024-07-01', '2025-01-01'), 'f1': FOLDS['f1'], 'f2': FOLDS['f2']}
EVALS = {**TIME, **{k: (trh, evh) for k, (trh, evh, _) in HOLD.items()}}
SEG = {'직전 6개월+/없음': (t['직전거래_경과일'].isna() | (t['직전거래_경과일'] > 180)).to_numpy(), '100세대 미만': (t['세대수'] < 100).to_numpy()}

out_path = S + 'r11_resid_test.csv'
done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['평가', '구성', '시드'])
rows = done.to_dict('records'); have = {(r['평가'], r['구성'], r['시드']) for r in rows}
for seed in SEEDS:
    for ev, (trm, evm) in EVALS.items():
        if (ev, CFG, seed) in have: continue
        evb = evm & broker; st = time.time()
        m = LGBMRegressor(**P['lgb'], n_estimators=P['best_iteration'], learning_rate=P['learning_rate'], subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
        a = ape(evb, m.fit(X.loc[trm, FEATS + NEW], y[trm]).predict(X.loc[evb, FEATS + NEW]))
        row = {'평가': ev, '구성': CFG, '시드': seed, 'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(evb.sum())}
        for sname, smask in SEG.items():
            sm = smask[evb]; row[f'{sname} MAPE'] = a[sm].mean(); row[f'{sname} n'] = int(sm.sum())
        row['초'] = round(time.time() - st)
        rows.append(row); print(row, flush=True); pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')

base = pd.read_csv(S + 'r11_method_test.csv'); base = base[base['구성'].eq('현재 11')]
d = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
print('\n[MAPE] 시드 평균'); print(d.pivot_table(index='구성', columns='평가', values='MAPE').round(3).to_string())
b = base.set_index(['평가', '시드'])['MAPE']
x = pd.DataFrame(rows); x['개선'] = b.loc[list(zip(x['평가'], x['시드']))].to_numpy() - x['MAPE'].to_numpy()
print('\n현재 11 대비 개선 (%p, +가 좋아짐):'); print(x.pivot_table(index='시드', columns='평가', values='개선').round(3).to_string())
tm = x[x['평가'].isin(TIME)]; ho = x[x['평가'].str.startswith('h')]
c1 = tm['개선'].mean(); c2 = tm.groupby('평가')['개선'].mean(); c3 = tm.groupby('시드')['개선'].mean(); c4 = ho['개선'].mean()
ok = [c1 >= 0.1, (c2 > 0).all(), (c3 > 0).all(), c4 >= 0]
print(f"\n{CFG}: (1) 시간 fold 평균 {c1:+.3f} {'O' if ok[0] else 'X'} | (2) fold별 {c2.round(3).to_dict()} {'O' if ok[1] else 'X'} | "
      f"(3) 시드별 {c3.round(3).to_dict()} {'O' if ok[2] else 'X'} | (4) 홀드아웃 {c4:+.3f} {'O' if ok[3] else 'X'} → {'채택' if all(ok) else '불채택'}")
for sname in SEG:
    bb = base.set_index(['평가', '시드'])[f'{sname} MAPE'] if f'{sname} MAPE' in base else None
    if bb is not None:
        xs = x.assign(seg_imp=bb.loc[list(zip(x['평가'], x['시드']))].to_numpy() - x[f'{sname} MAPE'].to_numpy())
        print(f"  구간 {sname}: 시간 fold 평균 개선 {xs[xs['평가'].isin(TIME)]['seg_imp'].mean():+.3f}%p")
