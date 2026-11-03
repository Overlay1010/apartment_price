# 31 × 11 결합 점검: "많이 오를 구"에서 "적정가보다 싸게" 거래된 집이 실제로 그 뒤 더 올랐나
# 같은 단지 · 같은 면적(반올림 ㎡)의 다음 중개거래(90~400일 뒤)와 비교한다.
# - 초과상승 = 두 거래 사이 로그 가격 변화 - 같은 기간 그 구 공식 지수 로그 변화 (시장 흐름을 뺀 값)
# - 싸게/비싸게 = 첫 거래의 회귀 OOF 잔차 r = log(실거래) - log(예측) (매달 재학습 예측이라 그 시점에 알 수 있던 적정가)
# - 많이 오를 구 = 첫 거래 달 1일 기준 공식 지수 4주 변화가 25개 구 중 상위 8개 (4단계 기준선)
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = os.path.join(HERE, *['..'] * 5, '')
t = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '단지일련번호', '시군구명', '계약일자', '전용면적', '층'])
o = pd.read_parquet(ROOT + 'models/regression/v2/oof_predictions.parquet')
d = o[o['거래유형'].eq('중개거래')].merge(t, on='transaction_id')
d['계약일자'] = pd.to_datetime(d['계약일자']); d['면적키'] = d['전용면적'].round()
d = d.sort_values('계약일자').reset_index(drop=True)

# 구 공식 지수 (주간, 월요일 + 3일 공표) — 계약일 기준 그 날까지 공표된 최신값
wk = pd.read_csv(ROOT + 'data/external/rone_weekly/apt_sale_index_weekly_gu.csv', dtype={'주': str})
wk['공표일'] = pd.to_datetime(wk['주']) + pd.Timedelta(days=3); wk['log지수'] = np.log(wk['지수'])
wk = wk.sort_values('공표일')
d = pd.merge_asof(d, wk[['공표일', '자치구', 'log지수']].rename(columns={'자치구': '시군구명'}), left_on='계약일자', right_on='공표일', by='시군구명')

# 다음 거래 짝짓기: 같은 단지 · 면적에서 90일 뒤 이후 첫 거래 (400일 이내)
nxt = d[['단지일련번호', '면적키', '계약일자', 'y', 'log지수', 'r']].rename(columns=lambda c: c + '_2' if c not in ('단지일련번호', '면적키') else c)
nxt['기준일'] = nxt['계약일자_2'] - pd.Timedelta(days=90)
pairs = pd.merge_asof(d.sort_values('계약일자'), nxt.sort_values('기준일'), left_on='계약일자', right_on='기준일',
                      by=['단지일련번호', '면적키'], direction='forward')
pairs = pairs[pairs['계약일자_2'].notna() & ((pairs['계약일자_2'] - pairs['계약일자']).dt.days <= 400)].copy()
pairs['보유일'] = (pairs['계약일자_2'] - pairs['계약일자']).dt.days
pairs['가격변화'] = (pairs['y_2'] - pairs['y']) * 100
pairs['시장변화'] = (pairs['log지수_2'] - pairs['log지수']) * 100
pairs['초과상승'] = pairs['가격변화'] - pairs['시장변화']

# 첫 거래 달 1일 기준 4주 흐름 상위 8개 구
reg = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_region_month.parquet', columns=['예측시점', '자치구', '공식지수_4주변화'])
reg['상위8'] = reg.groupby('예측시점')['공식지수_4주변화'].rank(ascending=False, method='first') <= 8
pairs['예측시점'] = pairs['계약일자'].dt.to_period('M').dt.to_timestamp()
pairs = pairs.merge(reg.rename(columns={'자치구': '시군구명'}), on=['예측시점', '시군구명'], how='left')
pairs = pairs[pairs['공식지수_4주변화'].notna()]
pairs['가격'] = pd.cut(pairs['r'], [-9, -0.05, -0.02, 0.02, 0.05, 9], labels=['5% 넘게 쌈', '2~5% 쌈', '적정(±2%)', '2~5% 비쌈', '5% 넘게 비쌈'])
pairs['구 흐름'] = np.where(pairs['상위8'], '많이 오를 구(상위 8)', '나머지 구')
print(f'짝 {len(pairs):,}개, 첫 거래 {pairs["계약일자"].min():%Y-%m} ~ {pairs["계약일자"].max():%Y-%m}, 보유 중앙 {pairs["보유일"].median():.0f}일')

agg = lambda g: pd.Series({'짝수': len(g), '가격변화(%)': g['가격변화'].mean(), '시장변화(%)': g['시장변화'].mean(),
                           '초과상승(%p)': g['초과상승'].mean(), '초과상승>0 비율': (g['초과상승'] > 0).mean(), '다음거래도_쌈(r<0)': (g['r_2'] < 0).mean()})
print('\n[가격 수준별]'); print(pairs.groupby('가격', observed=True).apply(agg).round(2).to_string())
print('\n[구 흐름별]'); print(pairs.groupby('구 흐름').apply(agg).round(2).to_string())
tab = pairs.groupby(['구 흐름', '가격'], observed=True).apply(agg).round(2)
print('\n[구 흐름 × 가격]'); print(tab.to_string())
# 연도별로 같은 방향인지 (한 해에만 생긴 현상인지 점검)
pairs['연'] = pairs['계약일자'].dt.year
print('\n[연도별 초과상승(%p): 5% 넘게 쌈 - 적정]')
yr = pairs.groupby(['연', '가격'], observed=True)['초과상승'].mean().unstack()
print((yr['5% 넘게 쌈'] - yr['적정(±2%)']).round(2).to_string())
tab.reset_index().to_csv(HERE + 'r31_cheap_rising.csv', index=False, encoding='utf-8-sig')
