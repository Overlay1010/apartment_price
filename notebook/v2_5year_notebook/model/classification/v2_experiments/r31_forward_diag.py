# 정답 정의 점검: 현재 공식 정답(향후 3개월 평균 ÷ 직전 3개월 평균)은 예측 시점에 이미 일어난 상승이 절반 가까이 섞인다.
# 순수 미래 변화(예측 시점에 공표된 최신 주 → 3개월 뒤 같은 시점)와 비교
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
wide = np.log(wk.pivot(index='주', columns='자치구', values='지수').sort_index())
rows = []
for p in pd.period_range('2022-01', '2026-06', freq='M'):
    ps = p.to_timestamp(); pe = (p + 3).to_timestamp()
    now = wide[wide.index + pd.Timedelta(days=3) < ps]       # p월 1일에 공표된 최신 주
    later = wide[wide.index + pd.Timedelta(days=3) < pe]     # p+3월 1일에 공표된 최신 주 (3개월 뒤 같은 시점)
    if len(now) < 14 or later.index[-1] <= now.index[-1] or later.index[-1] > wide.index[-1] or (pe > wide.index[-1] + pd.Timedelta(days=3)): continue
    for gu in wide.columns:
        rows.append({'p': p, '자치구': gu, '순수미래_3개월': later[gu].iloc[-1] - now[gu].iloc[-1],
                     '이미_일어난_부분': now[gu].iloc[-1] - wide.loc[(wide.index >= (p - 3).to_timestamp()) & (wide.index < ps), gu].mean(),
                     '모멘텀_4주': now[gu].iloc[-1] - now[gu].iloc[-5], '모멘텀_13주': now[gu].iloc[-1] - now[gu].iloc[-14]})
f = pd.DataFrame(rows).merge(r[['p', '자치구', '공식_향후3개월_변화율']], on=['p', '자치구'], how='left')
f['공식_log'] = np.log1p(f['공식_향후3개월_변화율'])
print('현재 정답 변화 중 예측 시점에 이미 일어난 부분 비중 (중앙):', round((f['이미_일어난_부분'] / f['공식_log']).where(f['공식_log'].abs() > 0.005).median(), 2))
print('상관: 현재 정답 vs 순수 미래', round(f['공식_log'].corr(f['순수미래_3개월']), 3))
d = f[f['p'].isin(DEV)]
for th, nm in [(0.01, '+1%'), (0.03, '+3%')]:
    for col, lab in [('공식_log', '현재 정답'), ('순수미래_3개월', '순수 미래')]:
        yt = (d[col] > np.log(1 + th)).astype(int)
        print(f'{nm} {lab}: 상승 비율 {yt.mean():.3f} | 4주 모멘텀 AUC {roc_auc_score(yt, d["모멘텀_4주"]):.3f} | 13주 {roc_auc_score(yt, d["모멘텀_13주"]):.3f}')
f.to_parquet(HERE + 'r31_forward_diag.parquet')
