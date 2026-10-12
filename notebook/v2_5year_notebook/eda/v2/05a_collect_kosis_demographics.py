"""
05 Demographics 원본 수집 (KOSIS OpenAPI 직접 호출, KSKILL_KOSIS_API_KEY)

| 표 | 내용 | 주기 |
|---|---|---|
| 101/DT_1B040A3 | 행정구역(시군구)별 성별 인구수 (주민등록) | 월 |
| 101/DT_1B040B3 | 행정구역(시군구)별 주민등록세대수 | 월 |
| 101/DT_1B04006 | 행정구역(시군구)별 1세별 주민등록인구 | 월 |
| 101/DT_1B26001_A01 | 시군구별 이동자수 (국내인구이동통계) | 월 |
| 408/DT_408_2006_S0065 | 매입자거주지별 아파트매매거래현황 (한국부동산원) | 월 |
| 408/DT_408_2006_S0077 | 매입자연령대별 아파트매매거래현황 (한국부동산원) | 월 |
| 101/DT_1JU1501 | 주택의 종류별 주택 (주택총조사) | 연 |

- 서울 + 25개 구만 받는다. 구 코드는 표마다 형식이 달라 메타데이터에서 이름으로 고른다.
- 한 번에 4만 셀을 넘지 않도록 기간을 나눠 받는다.
- 저장: data/external/kosis/<표ID>.parquet (KOSIS 원본 컬럼 그대로 + 수집일)

실행: python notebook/v2_5year_notebook/eda/v2/05a_collect_kosis_demographics.py
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
OUT_DIR = PROJECT_DIR / "data" / "external" / "kosis"
API_KEY = dotenv_values(PROJECT_DIR / ".env")["KSKILL_KOSIS_API_KEY"]
BASE = "https://kosis.kr/openapi"
TODAY = date.today().strftime("%Y%m%d")
MAX_CELLS = 40_000

SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구",
]

MONTHLY_START, MONTHLY_END = "201801", "202612"  # 끝은 넉넉히: KOSIS가 있는 달까지만 준다


def get(path, **params):
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE}/{path}", params={"apiKey": API_KEY, "format": "json", "jsonVD": "Y", **params},
                             timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("err"):
                if data["err"] == "30":  # 결과 없음
                    return []
                raise RuntimeError(f"KOSIS 오류 {data.get('err')}: {data.get('errMsg')}")
            return data
        except (requests.RequestException, ValueError) as e:
            time.sleep(2 ** attempt)
            last = str(e).replace(API_KEY, "<KEY>")
    raise RuntimeError(f"요청 실패: {path} {params.get('tblId')} — {last}")


def meta(org, tbl):
    return pd.DataFrame(get("statisticsData.do", method="getMeta", type="ITM", orgId=org, tblId=tbl))


def seoul_codes(m):
    # 지역 분류에서 '서울특별시'와 그 아래 25개 구 코드를 고른다 (표마다 코드 형식이 다르다)
    region = m[m["OBJ_ID"].ne("ITEM") & m["OBJ_NM"].str.contains("행정구역")]
    obj_id = region["OBJ_ID"].iloc[0]
    seoul = region[region["ITM_NM"].eq("서울특별시")]["ITM_ID"].iloc[0]
    under = region[region["ITM_ID"].str.startswith(seoul) & region["ITM_NM"].isin(SEOUL_GU)]
    if len(under) != 25:  # 주민등록 표처럼 구 코드가 시도 코드로 시작하지 않는 경우
        under = region[region["ITM_ID"].str.match(r"^11\d{3}$") & region["ITM_NM"].isin(SEOUL_GU)]
    assert len(under) == 25, (obj_id, len(under))
    return obj_id, [seoul] + under["ITM_ID"].tolist()


def months_between(start, end):
    return pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")


def fetch(org, tbl, prd_se, start, end, item_ids=None, extra_objs=None):
    m = meta(org, tbl)
    region_obj, regions = seoul_codes(m)
    items = item_ids or m.loc[m["OBJ_ID"].eq("ITEM"), "ITM_ID"].tolist()

    # 지역 외 분류(연령·매입자거주지 등)는 전체
    others = [o for o in m["OBJ_ID"].unique() if o not in ("ITEM", region_obj)]
    extra_objs = extra_objs or {}
    obj_params, cells_per_period = {}, len(regions) * len(items)
    for i, obj in enumerate([region_obj] + others, 1):
        codes = regions if obj == region_obj else extra_objs.get(obj, m.loc[m["OBJ_ID"].eq(obj), "ITM_ID"].tolist())
        obj_params[f"objL{i}"] = "+".join(codes)
        if obj != region_obj:
            cells_per_period *= len(codes)

    if prd_se == "M":
        periods = [p.strftime("%Y%m") for p in months_between(start, end)]
    else:
        periods = [str(y) for y in range(int(start), int(end) + 1)]
    chunk = max(1, MAX_CELLS // cells_per_period)

    rows = []
    for i in range(0, len(periods), chunk):
        part = periods[i:i + chunk]
        rows += get("Param/statisticsParameterData.do", method="getList", orgId=org, tblId=tbl,
                    itmId="+".join(items), prdSe=prd_se, startPrdDe=part[0], endPrdDe=part[-1], **obj_params)
        time.sleep(0.3)

    data = pd.DataFrame(rows)
    data["collected"] = TODAY
    return data


TABLES = [
    # (org, tbl, 주기, 시작, 끝, 항목, 추가 분류 필터)
    ("101", "DT_1B040A3", "M", MONTHLY_START, MONTHLY_END, ["T20"], None),
    ("101", "DT_1B040B3", "M", MONTHLY_START, MONTHLY_END, ["T1"], None),
    ("101", "DT_1B04006", "M", MONTHLY_START, MONTHLY_END, ["T2"], None),
    ("101", "DT_1B26001_A01", "M", MONTHLY_START, MONTHLY_END, None, None),
    ("408", "DT_408_2006_S0065", "M", MONTHLY_START, MONTHLY_END, None, None),
    ("408", "DT_408_2006_S0077", "M", "201901", MONTHLY_END, None, None),
    ("101", "DT_1JU1501", "Y", "2015", "2025", ["T10", "T30"], None),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for org, tbl, prd, start, end, items, extra in TABLES:
        data = fetch(org, tbl, prd, start, end, items, extra)
        path = OUT_DIR / f"{tbl}.parquet"
        data.to_parquet(path, index=False)
        print(f"{tbl}: {data.shape}, 기간 {data['PRD_DE'].min()} ~ {data['PRD_DE'].max()} → {path.name}")


if __name__ == "__main__":
    main()
