# 공식 지수(부동산원 주간 아파트 매매가격지수, 구별) 기준 정답 진단 — 반복거래 지수 · 중위 평당가 정답과 비교
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
wk = pd.read_csv(ROOT + 'data/external/rone_weekly/apt_sale_index_weekly_gu.csv', dtype={'주': str})
wk['날짜'] = pd.to_datetime(wk['주']); wk['M'] = wk['날짜'].dt.to_period('M')
O = np.log(wk.groupby(['M', '자치구'])['지수'].mean().unstack())  # 월 평균 log 지수
fut = O.rolling(3).mean().shift(-2); past = O.rolling(3).mean().shift(1)
oc = (fut - past).stack().rename('공식_변화').reset_index().rename(columns={'M': 'p'})
x = pd.read_parquet(HERE + 'r31_target_diag.parquet').merge(oc, on=['자치구', 'p'])
print('행', len(x), '| p', x['p'].min(), '~', x['p'].max())
print('3개월 변화율 분포 (공식):', x['공식_변화'].describe(percentiles=[.25, .5, .75, .9]).round(4).to_dict())
for th in (0.03, 0.02, 0.015, 0.01, 0.005):
    print(f'임계 {th:.3f}: 상승 비율 공식 {(x["공식_변화"] > np.log(1 + th)).mean():.3f} | 반복거래 {(x["rs_변화"] > np.log(1 + th)).mean():.3f}')
print('상관 — 공식 vs 반복거래', round(x['공식_변화'].corr(x['rs_변화']), 3), '| 공식 vs 중위', round(x['공식_변화'].corr(x['향후3개월_변화율']), 3))
x = x.sort_values(['자치구', 'p'])
print('공식 자기상관 lag1', round(x.groupby('자치구')['공식_변화'].apply(lambda s: s.autocorr(1)).mean(), 3),
      'lag4', round(x.groupby('자치구')['공식_변화'].apply(lambda s: s.autocorr(4)).mean(), 3))
# 예측 시점 p 1일에 알 수 있는 공식 지수: 주간 지수는 그 주 목요일 공표 → p-1월까지의 주 대부분 공표됨 (보수적으로 p-1월 평균까지)
mom_o = (O.shift(1) - O.shift(4)).stack().rename('공식_모멘텀').reset_index().rename(columns={'M': 'p'})
x = x.merge(mom_o, on=['자치구', 'p'])
dev = x[x['p'].isin(DEV)]
for th in (0.03, 0.01):
    yt = (dev['공식_변화'] > np.log(1 + th)).astype(int)
    print(f'DEV 임계 {th}: 상승 비율 {yt.mean():.3f} | 공식 모멘텀(≤p-1) AUC {roc_auc_score(yt, dev["공식_모멘텀"]):.3f} | 반복거래 모멘텀(≤p-2) AUC {roc_auc_score(yt, dev["rs_모멘텀"]):.3f}')
x.to_parquet(HERE + 'r31_official_diag.parquet')
