# 11 보완 후보: K-apt 공동주택 기본 · 상세 정보(05c 수집)에서 지금 11에 없는 단지 속성만 뽑아 실거래 단지와 연결 — 실험용 거래 단위 파일 생성
# - 연결: 실거래 (법정동코드 10자리, 본번-부번) = K-apt (bjdCode, kaptAddr 지번) → 안 맞으면 (시군구, 도로명, 건물 본번-부번) = K-apt 도로명주소
# - K-apt는 현재 값 하나뿐이라 바뀌지 않는 속성만 쓴다: 복도유형, 난방방식, 단지분류(주상복합), 분양형태, 시공사(10대 건설사), 지하주차 비중
#   (관리방식 · CCTV · 충전기처럼 시점마다 바뀌는 항목은 화면 표시용으로만 둔다)
import os, sys, re, json
import numpy as np, pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
ROOT = Path(S).parents[4]; D = ROOT / 'data/v2_5year_seoul/processed'; K = ROOT / 'data/external/kapt'

raw = [json.loads(p.read_text(encoding='utf-8')) for p in (K / 'raw').glob('*.json')]
b = pd.DataFrame([r['basic'] for r in raw if r['basic']]); d = pd.DataFrame([r['detail'] for r in raw if r['detail']])
k = b.merge(d[['kaptCode', 'kaptdPcnt', 'kaptdPcntu']], on='kaptCode', how='left')
print('K-apt 단지', len(k))

# 지번 · 도로명 키
jb = k['kaptAddr'].str.extract(r'\s(산?)(\d+)-?(\d*)\s')
k['지번키'] = np.where(jb[1].notna(), k['bjdCode'] + '|' + jb[0].fillna('') + jb[1].fillna('') + '-' + jb[2].replace('', '0').fillna('0'), None)
rd = k['doroJuso'].str.extract(r'서울특별시\s+(\S+구)\s+(.+?)\s+(\d+)(?:-(\d+))?\s*$')
k['도로키'] = np.where(rd[1].notna(), rd[0] + '|' + rd[1].str.replace(' ', '') + '|' + rd[2].fillna('') + '-' + rd[3].fillna('0'), None)

# 단지 속성
TOP10 = {'삼성물산': '삼성물산', '현대건설': '현대건설', '대우건설': '대우건설', '현대엔지니어링': '현대엔지니어링', 'DL이앤씨': 'DL', '디엘이앤씨': 'DL', '대림산업': 'DL',
         'GS건설': 'GS', '지에스건설': 'GS', 'LG건설': 'GS', '엘지건설': 'GS', '포스코': '포스코', '롯데건설': '롯데', 'SK': 'SK', '에스케이': 'SK',
         '현대산업개발': 'HDC', 'HDC': 'HDC', '삼성중공업': '삼성물산', '삼성종합건설': '삼성물산'}
def builder(s):
    s = re.sub(r'\(주\)|주식회사|㈜|\s', '', str(s)).upper()
    return next((v for key, v in TOP10.items() if key.upper() in s), '기타')
k['시공사그룹'] = k['kaptBcompany'].map(builder)
num = lambda c: pd.to_numeric(k[c], errors='coerce')
park = num('kaptdPcnt').fillna(0) + num('kaptdPcntu').fillna(0)
feat = pd.DataFrame({
    'kaptCode': k['kaptCode'], '지번키': k['지번키'], '도로키': k['도로키'], 'K_단지명': k['kaptName'],
    'K_복도유형': k['codeHallNm'].replace({'': None, '기타': None}),
    'K_난방방식': k['codeHeatNm'].str.extract(r'(개별|중앙|지역)')[0],
    'K_주상복합': k['codeAptNm'].str.contains('주상복합').astype(float).where(k['codeAptNm'].ne('')),
    'K_분양형태': k['codeSaleNm'].replace({'': None}),
    'K_시공사10대': k['시공사그룹'].ne('기타').astype(float),
    'K_시공사그룹': k['시공사그룹'],
    'K_지하주차비중': np.where(park > 0, num('kaptdPcntu').fillna(0) / park, np.nan),
})
print('값 분포:', {c: feat[c].value_counts(dropna=False).head(8).to_dict() for c in ['K_복도유형', 'K_난방방식', 'K_분양형태', 'K_시공사그룹']})

# 실거래 단지 키 (같은 단지일련번호의 지번 · 도로명은 하나로 모은다)
raw_t = pd.concat([pd.read_parquet(D / f, columns=['단지일련번호', '법정동시군구코드', '법정동읍면동코드', '지번', '도로명', '도로명건물본번호코드', '도로명건물부번호코드', '단지명'])
                   for f in ['apt_trade_raw.parquet', 'apt_trade_warmup_raw.parquet']]).drop_duplicates()
jj = raw_t['지번'].astype(str).str.extract(r'^(산?)(\d+)-?(\d*)$')
raw_t['지번키'] = (raw_t['법정동시군구코드'].astype(str).str.zfill(5) + raw_t['법정동읍면동코드'].astype(str).str.zfill(5) + '|'
                + jj[0].fillna('') + jj[1].fillna('') + '-' + jj[2].replace('', '0').fillna('0')).where(jj[1].notna())
cx = raw_t.groupby('단지일련번호').agg(지번키=('지번키', lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
                                    도로명=('도로명', 'first'), 본=('도로명건물본번호코드', 'first'), 부=('도로명건물부번호코드', 'first'), 단지명=('단지명', 'first')).reset_index()
gu = pd.read_parquet(D / 'v2_trade.parquet', columns=['단지일련번호', '시군구명']).drop_duplicates('단지일련번호')
cx['단지일련번호'] = cx['단지일련번호'].astype(str); gu['단지일련번호'] = gu['단지일련번호'].astype(str)
cx = cx.merge(gu, on='단지일련번호', how='inner')
to_int = lambda s: pd.to_numeric(s, errors='coerce').astype('Int64').astype(str)
cx['도로키'] = (cx['시군구명'] + '|' + cx['도로명'].astype(str).str.replace(' ', '') + '|' + to_int(cx['본']) + '-' + to_int(cx['부'])).where(cx['도로명'].notna() & cx['본'].notna())

# 1차 지번, 2차 도로명 (한 키에 K-apt 단지가 여럿이면 연결하지 않는다)
f1 = feat.dropna(subset=['지번키']).drop_duplicates('지번키', keep=False).drop(columns='도로키')
f2 = feat.dropna(subset=['도로키']).drop_duplicates('도로키', keep=False).drop(columns='지번키')
a1 = cx.merge(f1, on='지번키', how='left'); hit1 = a1['kaptCode'].notna().to_numpy()
a2 = cx[~hit1].merge(f2, on='도로키', how='left')
a = pd.concat([a1[hit1].assign(연결방법='지번'), a2.assign(연결방법=np.where(a2['kaptCode'].notna(), '도로명', '없음'))], ignore_index=True)
assert a['단지일련번호'].is_unique
print('실거래 단지', len(cx), '| 연결', a['연결방법'].value_counts().to_dict())
same = a[a['kaptCode'].notna()]
print('연결 예시 (실거래 단지명 vs K-apt 단지명):'); print(same.sample(min(12, len(same)), random_state=0)[['단지일련번호', '단지명', 'K_단지명', '연결방법']].to_string())

KF = ['K_복도유형', 'K_난방방식', 'K_주상복합', 'K_분양형태', 'K_시공사10대', 'K_시공사그룹', 'K_지하주차비중']
t = pd.read_parquet(D / 'v2_trade.parquet', columns=['transaction_id', '단지일련번호']); t['단지일련번호'] = t['단지일련번호'].astype(str)
out = t.merge(a[['단지일련번호', 'kaptCode', '연결방법'] + KF], on='단지일련번호', how='left', validate='many_to_one')
print('거래 기준 연결률', round(out['kaptCode'].notna().mean(), 3), '| Feature 결측률', out[KF].isna().mean().round(3).to_dict())
out.drop(columns='단지일련번호').to_parquet(S + 'kapt_feat.parquet', index=False)
print('저장 kapt_feat.parquet', out.shape)
