# 정답 진단: 구 중위 평당가 변화율(현재 정답)은 거래 단지 구성에 흔들린다 → 같은 단지 · 면적 반복거래 지수와 비교
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
t = pd.read_parquet(D + 'v2_trade.parquet', columns=['단지일련번호', '전용면적', '계약일자', '거래금액', '거래유형', '시군구명'])
w = pd.read_parquet(D + 'apt_trade_warmup_raw.parquet'); w = w[w['해제여부'].ne('O')].copy()
w['계약일자'] = pd.to_datetime(dict(year=w['계약년도'], month=w['계약월'], day=w['계약일']))
GU = {'11110': '종로구', '11140': '중구', '11170': '용산구', '11200': '성동구', '11215': '광진구', '11230': '동대문구', '11260': '중랑구', '11290': '성북구', '11305': '강북구', '11320': '도봉구', '11350': '노원구', '11380': '은평구', '11410': '서대문구', '11440': '마포구', '11470': '양천구', '11500': '강서구', '11530': '구로구', '11545': '금천구', '11560': '영등포구', '11590': '동작구', '11620': '관악구', '11650': '서초구', '11680': '강남구', '11710': '송파구', '11740': '강동구'}
w['시군구명'] = w['법정동시군구코드'].astype(str).str.zfill(5).map(GU)
a = pd.concat([w[['단지일련번호', '전용면적', '계약일자', '거래금액', '거래유형', '시군구명']], t], ignore_index=True)
a = a[a['거래유형'].ne('직거래') & a['시군구명'].notna()].copy()
a['거래금액'] = pd.to_numeric(a['거래금액'].astype(str).str.replace(',', ''), errors='coerce').astype(float)
a['unit'] = a['단지일련번호'].astype(str) + '_' + a['전용면적'].astype(float).round(0).astype(str)
a['M'] = a['계약일자'].dt.to_period('M'); a = a.sort_values('계약일자')
a['prevM'] = a.groupby('unit')['M'].shift(); a['prevP'] = a.groupby('unit')['거래금액'].shift()
pairs = a.dropna(subset=['prevM']).copy(); pairs = pairs[pairs['M'] != pairs['prevM']]
months = pd.period_range(a['M'].min(), a['M'].max(), freq='M'); mi = {m: i for i, m in enumerate(months)}
idx = {}
for gu, g in pairs.groupby('시군구명'):
    n = len(g); rows_ = np.repeat(np.arange(n), 2)
    cols = np.c_[g['prevM'].map(mi).to_numpy(), g['M'].map(mi).to_numpy()].ravel()
    vals = np.tile([-1.0, 1.0], n)
    A = csr_matrix((vals, (rows_, cols)), shape=(n, len(months)))[:, 1:]  # 첫 달 = 0 기준
    yv = np.log(g['거래금액'].to_numpy() / g['prevP'].to_numpy())
    beta = lsqr(A, yv, damp=0.05)[0]  # 약한 릿지로 거래 적은 달 안정화
    idx[gu] = pd.Series(np.r_[0, beta], index=months)
I = pd.DataFrame(idx)  # 구 × 월 log 반복거래 지수
# 반복거래 기준 정답: 향후 3개월(p..p+2) 평균 지수 − 직전 3개월(p-3..p-1) 평균 지수 > log(1.03)
fut = I.rolling(3).mean().shift(-2); past = I.rolling(3).mean().shift(1)
rs = (fut - past).stack().rename('rs_변화').reset_index().rename(columns={'level_0': 'p', 'level_1': '자치구'})
x = r[['자치구', 'p', '향후3개월_변화율', '상승여부']].merge(rs, on=['자치구', 'p'])
x['rs_상승'] = (x['rs_변화'] > np.log(1.03)).astype(int)
print('중위 기준 vs 반복거래 기준 변화율 상관', round(x['향후3개월_변화율'].corr(x['rs_변화']), 3), '| 정답 일치율', round((x['상승여부'] == x['rs_상승']).mean(), 3),
      '| 상승 비율 중위', round(x['상승여부'].mean(), 3), '반복거래', round(x['rs_상승'].mean(), 3))
x = x.sort_values(['자치구', 'p'])
for col in ['향후3개월_변화율', 'rs_변화']:
    print(col, '같은 구 자기상관 lag1', round(x.groupby('자치구')[col].apply(lambda s: s.autocorr(1)).mean(), 3),
          'lag4(예측 시점에 알 수 있는 최신)', round(x.groupby('자치구')[col].apply(lambda s: s.autocorr(4)).mean(), 3))
# 예측 시점에 알 수 있는 정보만으로 만든 단순 기준선: 확정월까지의 반복거래 지수 3개월 모멘텀
mom = (I.shift(2) - I.shift(5)).stack().rename('rs_모멘텀').reset_index().rename(columns={'level_0': 'p', 'level_1': '자치구'})
x = x.merge(mom, on=['자치구', 'p'])
dev = x[x['p'].isin(DEV)]
print('DEV 기준선 AUC — 반복거래 모멘텀(≤p-2)으로 반복거래 정답:', round(roc_auc_score(dev['rs_상승'], dev['rs_모멘텀']), 3),
      '| 같은 점수로 중위 정답:', round(roc_auc_score(dev['상승여부'], dev['rs_모멘텀']), 3))
I.to_parquet(HERE + 'rs_index_gu_month.parquet'); x.to_parquet(HERE + 'r31_target_diag.parquet')
print(I.iloc[[0, 12, 24, 36, 48, 60, -1]].round(3).iloc[:, :6])
