# 11 보완 4 후보: 호별 공시가격에서 만든 층별 공시 Feature — 실험용 거래 단위 파일 생성
# - 지금 모델의 공시 Feature는 단지 · 면적대 중위수뿐이라 같은 단지 안 층 차이를 담지 못한다. 호별 공시가는 층마다 다른 값을 준다
# - 시점: Y년 공시는 Y년 7월 1일부터 사용 (05b 공시 Feature와 같은 규칙, 4월 말 공시 · 6월 말 조정 반영)
# - 연결: 필지(lot) · 공시연도 안에서 전용면적이 가장 가까운 면적(0.5㎡ 이내)을 고른 뒤 그 면적의 층별 중위수를 쓴다. 동 정보는 쓰지 않는다
#   (실거래 아파트동명은 2021~22년에 비어 있어 기간에 따라 뜻이 달라진다)
# - 호명 → 층: '지하' 제외, 'N층' 표기는 N, 그 외에는 마지막 숫자 묶음이 3~4자리일 때 100으로 나눈 몫. 나머지는 층 미상으로 뺀다
# - 후보(평가 전에 고정): 호공시_층별m2당, 호공시_층프리미엄(층별 / 같은 면적 중위), 호공시_직전층비(현재 층 / 직전 거래 층),
#   직전거래가_층보정(직전거래가_시세보정 × 직전층비)
import os, re, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'
AREA_TOL = 0.5


def parse_floor(s):
    s = str(s).strip()
    if '지하' in s: return np.nan
    m = re.search(r'(\d+)\s*층', s)
    if m: return int(m[1]) or np.nan
    ds = re.findall(r'\d+', s)
    if not ds or len(ds[-1]) not in (3, 4): return np.nan
    return int(ds[-1]) // 100 or np.nan


t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '단지일련번호', '계약일자', '전용면적', '층', '직전거래_층', '직전거래가_시세보정',
                                                    '최고층', '공시연도', '공시_m2당_면적대', '거래금액', '거래유형', '데이터구분'])
t['단지일련번호'] = t['단지일련번호'].astype(str); t['계약일자'] = pd.to_datetime(t['계약일자'])
assert (t['공시연도'].to_numpy() == np.where(t['계약일자'].dt.month >= 7, t['계약일자'].dt.year, t['계약일자'].dt.year - 1)).all()
a = pd.read_parquet(D / 'complex_attributes_v2.parquet', columns=['단지일련번호', 'lot']); a['단지일련번호'] = a['단지일련번호'].astype(str)
t = t.merge(a, on='단지일련번호', how='left', validate='many_to_one').reset_index(drop=True)
lots, years = set(t['lot']), sorted(t['공시연도'].unique())

frames, notice = [], {}
for y in years:
    g = pd.read_parquet(ROOT / f'data/external/official_price/apt_official_price_seoul_{y}.parquet',
                        columns=['기준연도', '공시일', '법정동코드', '본번', '부번', '호명', '전용면적', '공시가격'])
    notice[y] = pd.Timestamp(g['공시일'].iloc[0])
    g['lot'] = g['법정동코드'].astype(str).str.zfill(10) + g['본번'].astype(str).str.zfill(4) + g['부번'].astype(str).str.zfill(4)
    frames.append(g[g['lot'].isin(lots)])
u = pd.concat(frames, ignore_index=True)
assert (t['계약일자'] > t['공시연도'].map(notice)).all()  # 쓰는 공시는 모두 계약일 전에 공시됐다
u['기준연도'] = u['기준연도'].astype(int)
u['전용면적'] = pd.to_numeric(u['전용면적'], errors='coerce').round(2)
u['m2당'] = pd.to_numeric(u['공시가격'], errors='coerce').astype(float) / u['전용면적'] / 10_000  # 만원/㎡
u = u.dropna(subset=['m2당'])
names = pd.Series(u['호명'].astype(str).unique()); floor_of = dict(zip(names, names.map(parse_floor)))
u['층'] = u['호명'].astype(str).map(floor_of)
print(f"호 수 {len(u):,} | 층을 읽은 호 {u['층'].notna().mean():.2%}")

# 호명으로 읽은 층이 건축물대장 최고층과 맞는지 (필지별 최대 층)
mx = u[u['기준연도'].eq(max(years))].groupby('lot')['층'].max().rename('읽은최고층')
chk = t.groupby('lot')['최고층'].max().to_frame().join(mx, how='inner').query('최고층 > 0')
print(f"필지 {len(chk):,}곳: 읽은 최고층 = 대장 최고층 {(chk['읽은최고층'] == chk['최고층']).mean():.1%}, "
      f"±1층 {((chk['읽은최고층'] - chk['최고층']).abs() <= 1).mean():.1%}, 대장보다 2층 넘게 높음 {(chk['읽은최고층'] > chk['최고층'] + 2).mean():.1%}")

known = u.dropna(subset=['층'])
area_tbl = u.groupby(['lot', '기준연도', '전용면적'])['m2당'].median().rename('면적중위').reset_index()
floor_tbl = known.groupby(['lot', '기준연도', '전용면적', '층'])['m2당'].median().rename('층값').reset_index()

q = t[['lot', '공시연도', '전용면적']].assign(i=np.arange(len(t))).rename(columns={'공시연도': '기준연도', '전용면적': '거래면적'})
q['거래면적'] = q['거래면적'].astype(float); q['기준연도'] = q['기준연도'].astype('int64')
ref = area_tbl.sort_values('전용면적').rename(columns={'전용면적': '면적'}); ref['면적키'] = ref['면적'].astype(float); ref['기준연도'] = ref['기준연도'].astype('int64')
m = pd.merge_asof(q.sort_values('거래면적'), ref, left_on='거래면적', right_on='면적키', by=['lot', '기준연도'], direction='nearest', tolerance=AREA_TOL).sort_values('i')
assert m['i'].is_unique and len(m) == len(t)


def floor_value(floors):
    x = pd.DataFrame({'lot': m['lot'].to_numpy(), '기준연도': m['기준연도'].to_numpy(), '전용면적': m['면적'].to_numpy(), '층': np.asarray(floors, float)})
    return x.merge(floor_tbl, on=['lot', '기준연도', '전용면적', '층'], how='left', validate='many_to_one')['층값'].to_numpy()


now_v, prev_v = floor_value(t['층']), floor_value(t['직전거래_층'])
ratio = now_v / prev_v
out = pd.DataFrame({'transaction_id': t['transaction_id'], '호공시_층별m2당': now_v, '호공시_층프리미엄': now_v / m['면적중위'].to_numpy(),
                    '호공시_직전층비': ratio, '직전거래가_층보정': t['직전거래가_시세보정'].to_numpy(float) * ratio})

# ---------- 재계산 검증: 표본을 호별 원본에서 직접 다시 계산 ----------
rng = np.random.default_rng(0)
for i in rng.choice(np.flatnonzero(~np.isnan(now_v)), 300, replace=False):
    r = t.iloc[i]; p = u[u['lot'].eq(r['lot']) & u['기준연도'].eq(r['공시연도'])]
    d = (p['전용면적'].astype(float) - float(r['전용면적'])).abs(); assert d.min() <= AREA_TOL
    same = p[p['전용면적'].eq(m['면적'].iat[i])]
    assert np.isclose(same[same['층'].eq(r['층'])]['m2당'].median(), now_v[i]) and np.isclose(same['m2당'].median() * out['호공시_층프리미엄'].iat[i], now_v[i])
print('✅ 표본 300건 재계산 일치 · 공시연도는 Y년 7월 규칙과 같음 · 쓰는 공시는 모두 계약일 전 공시')

print('결측률', out.drop(columns='transaction_id').isna().mean().round(3).to_dict())
yr = t['계약일자'].dt.year; print(pd.DataFrame({'연도': yr, '층별 있음': ~np.isnan(now_v), '직전층비 있음': ~np.isnan(ratio)}).groupby('연도').mean().round(3).to_string())
print('층 프리미엄 분위', np.round(np.nanquantile(out['호공시_층프리미엄'], [.01, .1, .5, .9, .99]), 3))

# Test 제외 참고: 단독 기준가로 볼 때 층 보정이 직전 거래가의 오차를 줄이는지 (학습 · 검증 구간, 중개거래)
mask = t['데이터구분'].isin(['train', 'valid']).to_numpy() & t['거래유형'].eq('중개거래').to_numpy() & ~np.isnan(ratio) & t['직전거래가_시세보정'].notna().to_numpy()
a_ = t['거래금액'].to_numpy(float); ae = lambda c: (np.abs(c[mask] - a_[mask]) / a_[mask] * 100)
base, adj = ae(t['직전거래가_시세보정'].to_numpy(float)), ae(out['직전거래가_층보정'].to_numpy(float))
samef = (t['층'] == t['직전거래_층']).to_numpy()[mask]
print(f"직전거래가 시세보정 단독 MAPE {base.mean():.3f}% → 층 보정 {adj.mean():.3f}% (n={mask.sum():,})")
print(f"  직전 거래와 층이 다른 거래만 {base[~samef].mean():.3f}% → {adj[~samef].mean():.3f}% (n={(~samef).sum():,})")
out.to_parquet(S + 'unit_feat.parquet', index=False); print('저장 unit_feat.parquet', out.shape)
