# 11 보완 후보: 학구도(한국교육시설안전원, 공공데이터포털 15159265 · 15159266)로 단지의 '배정 초등학교'를 찾고 그 학교 지표를 붙인다 — 실험용 거래 단위 파일 생성
# - 지금 학교 Feature는 '가장 가까운 초등학교' 기준이다. 실제 입학은 통학구역(학구)으로 정해지므로 배정 학교와 다를 수 있다
# - 학교 지표 시점은 02b와 같다: 거래의 학교기준연도(6월 이후 당해, 1~5월 전년) 값을 쓴다
# - 학구 경계는 2026-09 배포본 하나뿐이라 과거 경계 변경은 반영하지 못한다 (경계 수정일 분포를 함께 출력)
import os, sys, io, zipfile
import numpy as np, pandas as pd, shapefile
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import shape
from shapely import STRtree, points
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'; Z = ROOT / 'data/external/schoolzone'
EARTH_RADIUS_M = 6_371_000
VALUE_COLS = ['학생수', '학급당학생수', '교원1인당학생수', '학생수증감률', '학생순전입률']

# 1) 서울 초등학교 통학구역 폴리곤 (EPSG:5186, Korea 2000 중부원점 2010)
zf = zipfile.ZipFile(Z / '초등학교통학구역.zip'); names = zf.namelist()
get = lambda ext: io.BytesIO(zf.read(next(n for n in names if n.endswith(ext))))
rd = shapefile.Reader(shp=get('.shp'), shx=get('.shx'), dbf=get('.dbf'), encoding='cp949')
fields = [f.name for f in rd.fields[1:]]
zones = pd.DataFrame([dict(zip(fields, r.record)) | {'geom': shape(r.shape.__geo_interface__)} for r in rd.iterShapeRecords() if r.record['SD_CD'] == '11'])
print('서울 통학구역', len(zones), '| 구분', zones['HAKGUDO_GB'].value_counts().to_dict(), '| 경계 수정연도', zones['UPD_DT'].str[:4].value_counts().sort_index().to_dict())

# 2) 학구 → 학교 (연계정보) → 학교 지표(02 EDSS 테이블, 학교명으로 연결)
link = pd.read_csv(Z / '학교학구도연계정보.csv', encoding='cp949', dtype=str)
link = link[link['시도교육청명'].eq('서울특별시교육청') & link['학교급구분'].eq('초등학교')]
env = pd.read_parquet(D / 'school/edss_school_environment_2021_2025.parquet')
env = env[env['학교급구분'].eq('초등학교')].sort_values(['개방ID', '기준연도']).reset_index(drop=True)
prev = env.groupby('개방ID')[['기준연도', '학생수']].shift(1)
env['학생수증감률'] = np.where(prev['기준연도'] == env['기준연도'] - 1, (env['학생수'] / prev['학생수'] - 1) * 100, np.nan)
mv = pd.read_parquet(D / 'edss/student_movement_2021_2025.parquet')
mc = ['조사년도3월학생수', '입학증가수', '도내전입증가수', '도외전입증가수', '도내전출감소수', '도외전출감소수']
mv[mc] = mv[mc].apply(pd.to_numeric, errors='coerce'); mv = mv.groupby(['조사년도', '개방ID'])[mc].sum().reset_index().rename(columns={'조사년도': '기준연도'})
start = mv['조사년도3월학생수'] + mv['입학증가수']
mv['학생순전입률'] = np.where(start > 0, (mv['도내전입증가수'] + mv['도외전입증가수'] - mv['도내전출감소수'] - mv['도외전출감소수']) / start * 100, np.nan)
env = env.merge(mv[['기준연도', '개방ID', '학생순전입률']], on=['기준연도', '개방ID'], how='left', validate='one_to_one')
unmatched = sorted(set(link['학교명']) - set(env['학교명']))
print('연계정보 서울 초등 학교', link['학교명'].nunique(), '| EDSS 학교명과 안 맞는 학교', len(unmatched), unmatched[:10])
zone_school = link.groupby('학구ID')['학교명'].apply(lambda s: sorted(set(s))).to_dict()

# 3) 단지 좌표 → 통학구역 (점이 폴리곤 안에 있는지)
t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '위도', '경도', '학교기준연도', '최근접초_학교명'])
locs = t[['위도', '경도']].drop_duplicates().reset_index(drop=True)
x, yv = Transformer.from_crs(4326, 5186, always_xy=True).transform(locs['경도'].to_numpy(), locs['위도'].to_numpy())
tree = STRtree(zones['geom'].to_numpy())
pi, zi = tree.query(points(x, yv), predicate='within')
hit = pd.DataFrame({'loc': pi, '학구ID': zones['HAKGUDO_ID'].to_numpy()[zi]})
hit['학교명'] = hit['학구ID'].map(zone_school)
hit = hit.explode('학교명').dropna(subset=['학교명'])
n_zone = hit.groupby('loc')['학구ID'].nunique()
print(f'좌표 {len(locs)}개 중 통학구역 안: {hit["loc"].nunique()} | 학구 2개 이상 겹침: {(n_zone > 1).sum()}')

# 검증: 서울 초등학교 위치는 대부분 자기 학교 학구 안에 있어야 좌표계 · 연결이 맞다
sl = env[env['기준연도'].eq(2025)][['학교명', '위도', '경도']].drop_duplicates('학교명')
sx, sy = Transformer.from_crs(4326, 5186, always_xy=True).transform(sl['경도'].to_numpy(), sl['위도'].to_numpy())
spi, szi = tree.query(points(sx, sy), predicate='within')
own = pd.DataFrame({'학교명': sl['학교명'].to_numpy()[spi], '학구학교': [zone_school.get(z, []) for z in zones['HAKGUDO_ID'].to_numpy()[szi]]})
own_rate = own.groupby('학교명').apply(lambda g: any(g.name in s for s in g['학구학교'])).mean()
print(f'학교 위치가 자기 학구 안에 있는 비율: {own_rate:.1%}')
assert own_rate > 0.9

# 4) 좌표 × 학교기준연도: 배정 학교가 여러 곳(공동학구 · 겹침)이면 가장 가까운 배정 학교의 지표를 쓴다
rows = []
for year in sorted(t['학교기준연도'].unique()):
    e = env[env['기준연도'].eq(year)].set_index('학교명')
    h = hit[hit['학교명'].isin(e.index)].copy()
    h = h.join(locs, on='loc').join(e[['위도', '경도'] + VALUE_COLS].add_prefix('s_'), on='학교명')
    la1, lo1, la2, lo2 = map(np.radians, [h['위도'], h['경도'], h['s_위도'], h['s_경도']])
    h['거리'] = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    cnt = h.groupby('loc')['학교명'].nunique()
    best = h.sort_values('거리').drop_duplicates('loc').set_index('loc')
    f = locs.copy(); f['학교기준연도'] = year
    f['배정초_학교명'] = best['학교명']; f['배정초_거리_m'] = best['거리']; f['배정초_후보수'] = cnt
    for c in VALUE_COLS: f[f'배정초_{c}'] = best[f's_{c}']
    rows.append(f)
feat = pd.concat(rows, ignore_index=True)
out = t.merge(feat, on=['위도', '경도', '학교기준연도'], how='left', validate='many_to_one')
out['배정초_최근접일치'] = (out['배정초_학교명'] == out['최근접초_학교명']).astype(float).where(out['배정초_학교명'].notna())
HG = ['배정초_거리_m', '배정초_후보수', '배정초_최근접일치'] + [f'배정초_{c}' for c in VALUE_COLS]
print('거래 결측률', out[HG].isna().mean().round(4).to_dict())
print('배정 학교 = 가장 가까운 학교 비율(거래)', round(out['배정초_최근접일치'].mean(), 3))
print(out[HG].describe().T.round(2).to_string())
out[['transaction_id', '배정초_학교명'] + HG].to_parquet(S + 'hakgu_feat.parquet', index=False)
print('저장 hakgu_feat.parquet', out.shape)
