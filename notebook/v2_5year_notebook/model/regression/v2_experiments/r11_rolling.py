# 서비스 방식 평가: 매월 m 시작 시점에 확정된 거래(≤ m-2월)로 재학습 → m월 예측
import os
# + 최근 확정월 보정: m-2월 모델이 m-2월 거래에서 낸 잔차(구별 중위)를 m월 예측에 더함 (m 시작 시점에 알 수 있는 값)
import json, os, time
from lightgbm import LGBMRegressor
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read())
BP = json.load(open(S + 'r11_best_params.json'))['lgb']
P = S + 'r11_roll/'; os.makedirs(P, exist_ok=True)
per = t['거래월일자'].dt.to_period('M')
months = pd.period_range('2022-01', '2026-06', freq='M')
for m in months:
    f = P + f'{m}.npy'
    if os.path.exists(f): continue
    st = time.time(); trm = (per <= m - 2).to_numpy(); evm = (per == m).to_numpy()
    mdl = LGBMRegressor(n_estimators=3100, learning_rate=0.03, n_jobs=8, random_state=42, verbose=-1, bagging_freq=1, **BP)
    mdl.fit(X[trm], y[trm]); np.save(f, mdl.predict(X[evm]))
    print(m, '학습 행', trm.sum(), '예측 행', evm.sum(), 'MAPE', round(mape(evm, np.load(f)), 3), round(time.time() - st), '초', flush=True)
rows = []; res = {}
for m in months:
    evm = (per == m).to_numpy(); p = np.load(P + f'{m}.npy')
    res[m] = pd.DataFrame({'구': t.loc[evm, '시군구명'].to_numpy(), 'r': y[evm] - p})
    r = {'월': str(m), 'n': int(evm.sum()), 'MAPE': mape(evm, p), '편향%': float((np.exp(p - y[evm]) - 1).mean() * 100)}
    if m - 2 in res:
        g = res[m - 2].groupby('구')['r'].median(); s_all = res[m - 2]['r'].median()
        corr_gu = t.loc[evm, '시군구명'].map(g).fillna(s_all).to_numpy()
        r['MAPE_서울보정'] = mape(evm, p + s_all); r['MAPE_구보정'] = mape(evm, p + corr_gu)
        r['편향%_구보정'] = float((np.exp(p + corr_gu - y[evm]) - 1).mean() * 100)
    rows.append(r)
d = pd.DataFrame(rows); d.to_csv(S + 'r11_rolling.csv', index=False, encoding='utf-8-sig'); print(d.round(2).to_string())
tm = d['월'] >= '2026-01'
w = d[tm]['n'] / d[tm]['n'].sum()
print('Test 가중 평균', {c: round(float((d[tm][c] * w).sum()), 3) for c in ['MAPE', '편향%', 'MAPE_서울보정', 'MAPE_구보정', '편향%_구보정']})
vm = (d['월'] >= '2025-07') & (d['월'] <= '2025-12'); w = d[vm]['n'] / d[vm]['n'].sum()
print('Valid 가중 평균', {c: round(float((d[vm][c] * w).sum()), 3) for c in ['MAPE', '편향%', 'MAPE_서울보정', 'MAPE_구보정', '편향%_구보정']})
