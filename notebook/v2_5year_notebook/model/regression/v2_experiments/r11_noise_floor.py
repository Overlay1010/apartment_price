# 11 보완 3 점검: 같은 단지 · 면적(전용 1㎡ 반올림) · 같은 달 다른 거래의 실제 가격 평균으로 맞혔을 때의 오차 vs 매달 재학습 OOF 모델
# - 예측 시점에는 알 수 없는 값을 쓰는 비교 기준이다. 비교 거래도 한두 건이라 그 자체 오차가 섞여 엄밀한 하한은 아니다
# - 중개거래만. 같은 달 비교 거래가 없는 거래 · 신규 단지의 오차도 따로 낸다
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), ''); ROOT = S + '../../../../../'
t = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '단지일련번호', '전용면적'])
o = pd.read_parquet(ROOT + 'models/regression/v2/oof_predictions.parquet')
d = o.merge(t, on='transaction_id', validate='one_to_one'); d['월'] = d['월'].astype(str); d = d[d['거래유형'].eq('중개거래')].copy()
d['grp'] = d['단지일련번호'].astype(str) + '|' + d['전용면적'].round(0).astype(str) + '|' + d['월']
g = d.groupby('grp')['y']; n = g.transform('size'); d['비교거래'] = n >= 2
d['loo_ape'] = np.abs(np.exp(d['y'] - (g.transform('sum') - d['y']) / (n - 1).where(n >= 2)) - 1) * 100
PERIOD = {'2023~2025': d['월'].between('2023-01', '2025-12'), 'Test 2026-01~06': d['월'] >= '2026-01'}
rows = []
for name, mask in PERIOD.items():
    x = d[mask]; p = x[x['비교거래']]
    rows.append({'기간': name, '구분': '비교 거래 있음: 같은 달 다른 거래 평균', 'n': len(p), 'MAPE': p['loo_ape'].mean(), '중앙': p['loo_ape'].median()})
    rows.append({'기간': name, '구분': '비교 거래 있음: 모델', 'n': len(p), 'MAPE': p['ape'].mean(), '중앙': p['ape'].median()})
    for lab, q in [('비교 거래 없음 · 기존 단지: 모델', x[~x['비교거래'] & x['단지'].eq('기존')]), ('신규 단지: 모델', x[x['단지'].eq('신규')])]:
        rows.append({'기간': name, '구분': lab, 'n': len(q), 'MAPE': q['ape'].mean(), '중앙': q['ape'].median()})
r = pd.DataFrame(rows); r.to_csv(S + 'r11_noise_floor.csv', index=False, encoding='utf-8-sig'); print(r.round(2).to_string(index=False))

# 비교 거래 자체의 잡음을 뺀 추정 (2023~25만, Test 제외): 비교 거래 m건 평균의 잡음이 독립이면 기준 오차 ≈ 하한 × sqrt(1 + 1/m)
# 같은 달 시세를 아는 값이라 예측 시점에는 쓸 수 없다 — 격차에는 정보 지연이 포함된다
x = d[d['월'].between('2023-01', '2025-12') & d['비교거래']].merge(pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '층']), on='transaction_id', validate='one_to_one')
x['m'] = x.groupby('grp')['y'].transform('size') - 1
rows = [{'구분': '같은 달 다른 거래 평균', 'n': len(x), '기준': x['loo_ape'].mean(), '잡음 제거 추정': (x['loo_ape'] / np.sqrt(1 + 1 / x['m'])).mean(), '모델': x['ape'].mean()}]
fl = []
for _, s in x[['grp', 'y', '층', 'ape']].groupby('grp'):
    Y, F, A = s['y'].to_numpy(), s['층'].to_numpy(float), s['ape'].to_numpy()
    for i in range(len(s)):
        k = np.abs(F - F[i]) <= 2; k[i] = False
        if k.any(): fl.append((abs(np.exp(Y[i] - Y[k].mean()) - 1) * 100, A[i], int(k.sum())))
z = pd.DataFrame(fl, columns=['e', 'ape', 'm'])
rows.append({'구분': '층 차이 2 이내 거래만', 'n': len(z), '기준': z['e'].mean(), '잡음 제거 추정': (z['e'] / np.sqrt(1 + 1 / z['m'])).mean(), '모델': z['ape'].mean()})
r2 = pd.DataFrame(rows); r2.to_csv(S + 'r11_noise_floor_adj.csv', index=False, encoding='utf-8-sig'); print(); print(r2.round(2).to_string(index=False))
