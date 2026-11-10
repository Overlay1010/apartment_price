# 11 보완 3 후보: 비교 거래가 드문 거래용 시세 보정 기준가 — 실험용 거래 단위 파일 생성
# - 진단(수치는 2023~25 OOF): 같은 면적 직전 거래가 6개월보다 오래됐거나 없는 거래가 18%인데 오차의 27%,
#   100세대 미만 단지는 거래 7%에 오차 12%. 단, 이 구간을 살펴본 계기는 Test 구간별 오차(r11_noise_floor)였고 진단 표에도 Test 열을 함께 출력했다
# - 지금 단지 수준 기준가 `예상가격_단지만`은 [계약일-395, 계약일-30) 단지 거래 평당가 중위수로, 그 사이 시세 변화를 반영하지 않는다
# - 후보 A (단지_): 같은 단지 모든 면적 거래를 구 주간 지수로 지금 시세까지 보정한 평당가 중위수 × 전용면적, 창 안 거래 수, 최근 거래 경과일
# - 후보 B (유사_): 같은 구에서 좌표 · 사용승인연도 · 세대수가 가까운 단지 10곳의 같은 보정 평당가 중위수 × 전용면적, 창 안 거래 수
#   21 유사 단지 KNN(similar_complexes.parquet)은 2026-07 스냅샷 속성으로 이웃을 골라 과거 거래에 쓰면 미래 속성이 섞이고,
#   준공연도가 없어 신축의 이웃이 구축이 된다 → 바뀌지 않는 속성만으로 다시 고른다 (가격은 쓰지 않음)
# - 참고 거래: 직거래 제외, 계약일 30일 전보다 앞선 거래만 (신고 기한 30일, 03 예상가격과 같은 창)
# - 시세 보정: 참고 거래가 속한 주 지수 → 계약일 전에 공표된 최신 주 지수 (r11_idx_build와 같은 규칙, 주 + 3일 공표)
#   지수 시작(2021-07-05) 전 거래는 보정할 수 없어 참고 거래에서 뺀다
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'
RELEASE_LAG = pd.Timedelta(days=3)
LAG_DAYS, WINDOW_DAYS = 30, 365
K_NB = 10
PYEONG = 3.3058

w = pd.read_csv(ROOT / 'data/external/rone_weekly/apt_sale_index_weekly_gu.csv', dtype={'주': str})
w['주'] = pd.to_datetime(w['주']); w = w.sort_values('주')
t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '단지일련번호', '시군구명', '계약일자', '거래유형', '평당가',
                                                    '전용면적', '위도', '경도', '사용승인연도', '세대수'])
t['단지일련번호'] = t['단지일련번호'].astype(str); t['계약일자'] = pd.to_datetime(t['계약일자'])
t = t.reset_index(drop=True)


def index_at(dates, gu, lag):
    # r11_idx_build.index_at과 같은 규칙: lag가 있으면 (주 + lag) < date인 마지막 주, 없으면 주 ≤ date인 마지막 주
    q = pd.DataFrame({'i': np.arange(len(dates)), 'key': dates.to_numpy(), '자치구': gu.to_numpy()}).sort_values('key')
    ref = w[['주', '자치구', '지수']].copy()
    ref['key'] = ref['주'] + (lag if lag is not None else pd.Timedelta(0))
    m = pd.merge_asof(q, ref.sort_values('key'), on='key', by='자치구', direction='backward', allow_exact_matches=lag is None)
    out = np.full(len(dates), np.nan); out[m['i'].to_numpy()] = m['지수'].to_numpy()
    week = np.full(len(dates), np.datetime64('NaT'), 'datetime64[ns]'); week[m['i'].to_numpy()] = m['주'].to_numpy()
    return out, week


now_idx, now_week = index_at(t['계약일자'], t['시군구명'], RELEASE_LAG)
ref_idx, ref_week = index_at(t['계약일자'], t['시군구명'], None)
ref_idx[(t['계약일자'] < w['주'].min()).to_numpy()] = np.nan
assert (pd.Series(now_week)[~np.isnan(now_idx)] + RELEASE_LAG < t['계약일자'][~np.isnan(now_idx)]).all()

# 참고 거래: 직거래 제외(거래유형 미기재 2021년 초 거래는 포함, lt_feature와 같음), 보정 가능한 거래
is_ref = t['거래유형'].ne('직거래').to_numpy() & ~np.isnan(ref_idx)
v = t['평당가'].to_numpy(float) / ref_idx  # 지수 1 기준 평당가 → 같은 구 안에서는 × 지금 지수로 시세 보정
day = t['계약일자'].to_numpy().astype('datetime64[D]').astype(np.int64)
cx = t['단지일련번호'].to_numpy()


def window_stats(q_rows, ref_rows):
    # q_rows 각 거래에 대해 ref_rows 중 [d-395, d-30) 참고 거래의 v 중위수 · 건수 · 가장 최근 참고 거래까지 경과일(창 제한 없음)
    r = ref_rows[np.argsort(day[ref_rows], kind='stable')]
    rd, rv = day[r], v[r]
    qd = day[q_rows]
    lo = np.searchsorted(rd, qd - (WINDOW_DAYS + LAG_DAYS), side='left')
    hi = np.searchsorted(rd, qd - LAG_DAYS, side='left')
    med = np.full(len(q_rows), np.nan); cnt = hi - lo
    for j in np.flatnonzero(cnt > 0):
        med[j] = np.median(rv[lo[j]:hi[j]])
    last = np.where(hi > 0, qd - rd[np.maximum(hi - 1, 0)], np.nan)
    return med, cnt, last


# ---------- 후보 A: 같은 단지 모든 면적 ----------
a_med = np.full(len(t), np.nan); a_cnt = np.zeros(len(t), int); a_last = np.full(len(t), np.nan)
groups = pd.Series(np.arange(len(t))).groupby(cx).indices
for k, idx in groups.items():
    ref = idx[is_ref[idx]]
    if len(ref) == 0: continue
    m, c, l = window_stats(idx, ref)
    a_med[idx], a_cnt[idx], a_last[idx] = m, c, l

# ---------- 후보 B: 같은 구 유사 단지 10곳 (바뀌지 않는 속성만) ----------
cxt = t.groupby('단지일련번호').agg(구=('시군구명', 'first'), 위도=('위도', 'first'), 경도=('경도', 'first'),
                                  준공=('사용승인연도', 'first'), 세대=('세대수', 'first'))
cxt['준공'] = cxt['준공'].fillna(cxt.groupby('구')['준공'].transform('median'))
cxt['세대'] = np.log(cxt['세대'].fillna(cxt.groupby('구')['세대'].transform('median')).clip(lower=1))
# 거리 척도: 1km ≈ 준공 5년 ≈ 세대수 e배 (사전에 정한 값, 평가 결과로 고르지 않음)
lat0 = np.radians(37.55)
Z = np.c_[cxt['위도'] * 111.0, cxt['경도'] * 111.0 * np.cos(lat0), cxt['준공'] / 5.0, cxt['세대'] / 1.0]
nb = {}
for g, sub in cxt.groupby('구'):
    ix = np.flatnonzero(cxt['구'].eq(g).to_numpy()); Zg = Z[ix]
    dist = np.sqrt(((Zg[:, None, :] - Zg[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)[:, :K_NB]
    for a, row in enumerate(order):
        nb[cxt.index[ix[a]]] = cxt.index[ix[row]].tolist()
assert all(k not in v_ for k, v_ in nb.items())
b_med = np.full(len(t), np.nan); b_cnt = np.zeros(len(t), int)
for k, idx in groups.items():
    nidx = [groups[n] for n in nb[k] if n in groups]
    if not nidx: continue
    ref = np.concatenate(nidx); ref = ref[is_ref[ref]]
    if len(ref) == 0: continue
    m, c, _ = window_stats(idx, ref)
    b_med[idx], b_cnt[idx] = m, c

area_py = t['전용면적'].to_numpy(float) / PYEONG
out = pd.DataFrame({'transaction_id': t['transaction_id'],
                    '단지_시세보정가': a_med * now_idx * area_py, '단지_창내거래수': a_cnt, '단지_최근거래경과일': a_last,
                    '유사_시세보정가': b_med * now_idx * area_py, '유사_창내거래수': b_cnt})

# ---------- 누수 · 재계산 검증: 표본을 원본에서 직접 다시 계산 ----------
rng = np.random.default_rng(0)
tt = t.assign(v=v, is_ref=is_ref, now_idx=now_idx)
for i in rng.choice(np.flatnonzero(a_cnt > 0), 300, replace=False):
    r = tt.iloc[i]
    p = tt[tt['is_ref'] & tt['단지일련번호'].eq(r['단지일련번호'])
           & (tt['계약일자'] >= r['계약일자'] - pd.Timedelta(days=WINDOW_DAYS + LAG_DAYS)) & (tt['계약일자'] < r['계약일자'] - pd.Timedelta(days=LAG_DAYS))]
    assert len(p) == a_cnt[i] and np.isclose(p['v'].median() * r['now_idx'] * r['전용면적'] / PYEONG, out['단지_시세보정가'].iat[i])
    assert (p['계약일자'] < r['계약일자'] - pd.Timedelta(days=LAG_DAYS)).all()
for i in rng.choice(np.flatnonzero(b_cnt > 0), 300, replace=False):
    r = tt.iloc[i]
    p = tt[tt['is_ref'] & tt['단지일련번호'].isin(nb[r['단지일련번호']])
           & (tt['계약일자'] >= r['계약일자'] - pd.Timedelta(days=WINDOW_DAYS + LAG_DAYS)) & (tt['계약일자'] < r['계약일자'] - pd.Timedelta(days=LAG_DAYS))]
    assert len(p) == b_cnt[i] and np.isclose(p['v'].median() * r['now_idx'] * r['전용면적'] / PYEONG, out['유사_시세보정가'].iat[i])
    # 구 경계에 걸친 단지 5곳은 거래마다 시군구명이 달라, 이웃 범위는 단지 대표 구(첫 거래)로 확인한다
    assert not p['단지일련번호'].eq(r['단지일련번호']).any() and p['단지일련번호'].map(cxt['구']).eq(cxt.at[r['단지일련번호'], '구']).all()
assert (out['단지_최근거래경과일'].dropna() >= LAG_DAYS).all()
print('✅ 표본 600건 재계산 일치 · 참고 거래는 모두 계약일 30일 전보다 앞섬 · 유사 단지는 같은 구 · 자기 단지 제외')

print('결측률', out.drop(columns='transaction_id').isna().mean().round(3).to_dict())
yr = t['계약일자'].dt.year
print(pd.DataFrame({'연도': yr, 'A 있음': out['단지_시세보정가'].notna(), 'B 있음': out['유사_시세보정가'].notna()}).groupby('연도').mean().round(3).to_string())
lp = np.log(t['평당가'] * area_py)
for c in ['단지_시세보정가', '유사_시세보정가']:
    e = (np.abs(np.exp(lp - np.log(out[c])) - 1) * 100)[out[c].notna() & (yr >= 2023)]
    print(f'{c} 단독 기준가 오차(2023~, 참고): 평균 {e.mean():.2f}% 중앙 {e.median():.2f}%')
out.to_parquet(S + 'nb_feat.parquet', index=False)
pd.DataFrame([(k, n) for k, v_ in nb.items() for n in v_], columns=['단지일련번호', '유사단지일련번호']).to_parquet(S + 'nb_neighbors.parquet', index=False)
print('저장 nb_feat.parquet', out.shape)
