# 11 보완 4 점검: 오차가 어느 구간에 몰려 있는지 — 다음 후보의 우선순위를 정하는 근거
# - 매달 재학습 OOF(models/regression/v2/oof_predictions.parquet, 보완 2 모델), 중개거래, 2023-01~2025-12 (Test 제외)
# - 구간(위에서부터 먼저 해당): 처음 보는 단지 → 기존 단지지만 1년 내 단지 거래 없음(예상가격_단지만 결측) → 같은 면적 직전 거래 없음 · 6개월+ → 같은 면적 직전 거래 6개월 내
# - '20% 감소 시 개선'은 규모를 보기 위한 가정이지 예상치가 아니다
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 250)
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), ''); ROOT = os.path.join(S, *['..'] * 5, '')
o = pd.read_parquet(ROOT + 'models/regression/v2/oof_predictions.parquet')
t = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '예상가격_단지만', '직전거래_경과일'])
d = o.merge(t, on='transaction_id', how='left', validate='one_to_one')
d = d[d['거래유형'].eq('중개거래') & d['월'].astype(str).between('2023-01', '2025-12')].copy()
print(f"대상 {len(d):,}건, MAPE {d['ape'].mean():.3f}%, 중앙 {d['ape'].median():.3f}%")
d['구간'] = np.select([d['단지'].ne('기존'), d['예상가격_단지만'].isna(), d['직전거래_경과일'].isna() | (d['직전거래_경과일'] > 180)],
                     ['1 처음 보는 단지', '2 기존 단지 · 1년 내 단지 거래 없음', '3 같은 면적 직전 거래 없음 · 6개월+'], '4 같은 면적 직전 거래 6개월 내')
g = d.groupby('구간')['ape']
r = pd.DataFrame({'건수': g.size(), 'MAPE': g.mean(), '중앙': g.median(), '거래 비중%': g.size() / len(d) * 100,
                  '오차 비중%': g.sum() / d['ape'].sum() * 100, '20% 감소 시 전체 개선(%p)': g.sum() * 0.2 / len(d)})
print(r.round(2).to_string())
print(f"APE 20% 이상: 거래 {(d['ape'] >= 20).mean() * 100:.1f}%, 오차 비중 {d.loc[d['ape'] >= 20, 'ape'].sum() / d['ape'].sum() * 100:.1f}%")
r.round(4).to_csv(S + 'r11_error_share.csv', encoding='utf-8-sig')
