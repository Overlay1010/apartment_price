# 11 보완 후보: 같은 단지 · 면적 직전 거래가를 그 사이 구 시세 변화(부동산원 주간 아파트 매매가격지수, 01b 수집)만큼 보정 — 실험용 거래 단위 파일 생성
# - 지금 11은 직전 거래가와 경과일만 알고, 그 기간에 시세가 얼마나 움직였는지는 모른다
# - 지금 지수: 계약일 전에 공표된 최신 주 (월요일 기준 주 값은 목요일 +3일 공표)
# - 직전 거래 시점 지수: 직전 계약일이 속한 주 (경과일 ≥ 30일이라 계약일 전에 이미 공표됨)
# - 직전 계약일이 주간 지수 시작(2021-07-05) 전이면 비워 둔다
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'
RELEASE_LAG = pd.Timedelta(days=3)

w = pd.read_csv(ROOT / 'data/external/rone_weekly/apt_sale_index_weekly_gu.csv', dtype={'주': str})
w['주'] = pd.to_datetime(w['주']); w = w.sort_values('주')
t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '계약일자', '시군구명', '직전거래가_동일면적', '직전거래_경과일'])
t['계약일자'] = pd.to_datetime(t['계약일자'])
t['직전계약일'] = t['계약일자'] - pd.to_timedelta(t['직전거래_경과일'], unit='D')

def index_at(dates, gu, lag):
    # dates 시점에 쓸 수 있는 구 지수: (주 + lag) < date 인 마지막 주, lag=None이면 주 ≤ date 인 마지막 주
    q = pd.DataFrame({'i': np.arange(len(dates)), 'key': dates.to_numpy(), '자치구': gu.to_numpy()}).dropna(subset=['key']).sort_values('key')
    ref = w[['주', '자치구', '지수']].copy()
    if lag is None:
        ref['key'] = ref['주']; m = pd.merge_asof(q, ref.sort_values('key'), on='key', by='자치구', direction='backward', allow_exact_matches=True)
    else:
        ref['key'] = ref['주'] + lag; m = pd.merge_asof(q, ref.sort_values('key'), on='key', by='자치구', direction='backward', allow_exact_matches=False)
    out = np.full(len(dates), np.nan); out[m['i'].to_numpy()] = m['지수'].to_numpy()
    week = pd.Series(pd.NaT, index=range(len(dates))); week.iloc[m['i'].to_numpy()] = m['주'].to_numpy()
    return out, week

now, now_week = index_at(t['계약일자'], t['시군구명'], RELEASE_LAG)
then, then_week = index_at(t['직전계약일'], t['시군구명'], None)
then[(t['직전계약일'] < w['주'].min()).to_numpy()] = np.nan  # 지수 시작 전 직전 거래
# 누수 점검: 쓰는 두 주 값 모두 계약일 전에 공표된 것이어야 한다
ok = t['직전계약일'].notna() & ~np.isnan(then)
assert ((now_week + RELEASE_LAG) < t['계약일자'].reset_index(drop=True)).loc[~np.isnan(now)].all()
assert ((then_week + RELEASE_LAG)[ok.to_numpy()] < t['계약일자'].reset_index(drop=True)[ok.to_numpy()]).all()

ratio = now / then
out = pd.DataFrame({'transaction_id': t['transaction_id'],
                    '지수_직전거래이후변화': np.log(ratio),
                    '지수_직전거래가보정': t['직전거래가_동일면적'].to_numpy() * ratio})
print('결측률', out.isna().mean().round(4).to_dict())
print(out['지수_직전거래이후변화'].describe().round(4).to_string())
out.to_parquet(S + 'idx_feat.parquet', index=False)
print('저장 idx_feat.parquet', out.shape)
