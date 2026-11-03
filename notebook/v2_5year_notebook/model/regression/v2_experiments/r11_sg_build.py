# 11 보완 후보: 단지 주변 상권 규모 Feature (소상공인 상가(상권)정보 분기 스냅샷) — 실험용 거래 단위 파일 생성
# 시점 규칙은 02c 학원 Feature와 같다: 스냅샷은 기준월 + 3개월부터 사용, 결손 분기(2023-03~2024-09)와 좌표 오류 분기(2025-03)는 건너뛰고 직전 정상 스냅샷
import os, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'
STORE_DIR = ROOT / 'data/external/상가정보'
EARTH_RADIUS_M = 6_371_000
RELEASE_LAG_MONTHS = 3
BAD = {'202303', '202306', '202309', '202312', '202403', '202406', '202409', '202503'}
# (Feature, 반경 m, 업종 조건) — 교육(P1)은 02c 학원 Feature, 부동산(L1)은 주거지 자체를 세는 셈이라 뺀다
LIFE = {'G2', 'I2', 'Q1', 'S2', 'R1'}  # 소매 · 음식 · 보건의료 · 수리·개인 · 예술·스포츠 = 주민이 직접 쓰는 업소
RULES = [('상권_500m_생활업소', 500, lambda d: d['대'].isin(LIFE)), ('상권_1km_생활업소', 1000, lambda d: d['대'].isin(LIFE)),
         ('상권_500m_음식점', 500, lambda d: d['대'].eq('I2')), ('상권_500m_카페', 500, lambda d: d['중'].eq('I212')),
         ('상권_500m_식료품종합소매', 500, lambda d: d['중'].isin({'G204', 'G205'})), ('상권_1km_의원', 1000, lambda d: d['중'].eq('Q102')),
         ('상권_1km_사무업소', 1000, lambda d: d['대'].eq('M1'))]
SG_COLS = [n for n, _, _ in RULES]

t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '위도', '경도', '거래월일자'])
t['거래월'] = pd.to_datetime(t['거래월일자']).dt.to_period('M')
locs = t[['위도', '경도']].drop_duplicates().reset_index(drop=True)
loc_rad = np.radians(locs[['위도', '경도']].to_numpy())

snaps = sorted(p.stem.split('_')[-1] for p in STORE_DIR.glob('*.csv'))
good = [s for s in snaps if s not in BAD]
avail = pd.DataFrame({'기준월': good, '사용가능': [pd.Period(f'{s[:4]}-{s[4:]}', 'M') + RELEASE_LAG_MONTHS for s in good]})
months = pd.DataFrame({'거래월': sorted(t['거래월'].unique())})
months['기준월'] = [avail.loc[avail['사용가능'] <= m, '기준월'].iloc[-1] for m in months['거래월']]
need = sorted(months['기준월'].unique()); print('사용 스냅샷:', need)

frames = []
for s in need:
    d = pd.read_csv(next(STORE_DIR.glob(f'*_{s}.csv')), encoding='utf-8-sig', dtype=str,
                    usecols=['상권업종대분류코드', '상권업종중분류코드', '위도', '경도']).rename(columns={'상권업종대분류코드': '대', '상권업종중분류코드': '중'})
    d[['위도', '경도']] = d[['위도', '경도']].astype(float); d = d.dropna(subset=['위도', '경도'])
    assert d['위도'].between(37.41, 37.72).mean() > 0.99, s  # 좌표 오류 분기 차단
    f = locs.copy(); f['기준월'] = s
    for name, radius, cond in RULES:
        sub = d[cond(d)]
        f[name] = BallTree(np.radians(sub[['위도', '경도']].to_numpy()), metric='haversine').query_radius(loc_rad, r=radius / EARTH_RADIUS_M, count_only=True)
    frames.append(f); print(s, len(d), f[SG_COLS].median().round(0).to_dict(), flush=True)
feat = pd.concat(frames, ignore_index=True)
out = t[['transaction_id', '위도', '경도', '거래월']].merge(months, on='거래월').merge(feat, on=['위도', '경도', '기준월'], how='left', validate='many_to_one')
assert len(out) == len(t) and out[SG_COLS].notna().all().all()
out.rename(columns={'기준월': '상권스냅샷기준월'})[['transaction_id', '상권스냅샷기준월'] + SG_COLS].to_parquet(S + 'sg_feat.parquet', index=False)
print('저장 sg_feat.parquet', out.shape)
