"""
부동산원 주간 아파트 매매 · 전세가격지수 (구별) · 수급동향 (서울 권역별) 수집 — 31 분류 정답 · Feature용
(KOSIS OpenAPI 직접 호출, KSKILL_KOSIS_API_KEY)

| 표 | 내용 | 주기 |
|---|---|---|
| 408/DT_304004_WEEK_002_C | 주간 아파트 매매가격지수 (전국주택가격동향조사, 지정통계) | 주 (월요일 기준, 그 주 목요일 공표) |
| 408/DT_304004_WEEK_004_C | 주간 아파트 전세가격지수 | 〃 |
| 408/DT_304004_WEEK_013_C | 주간 아파트 수급동향 (매매수급 · 전세수급, 100 = 수요 · 공급 균형) — 구별이 없어 서울 5개 권역 | 〃 |

- 가격지수는 서울 25개 구만 받는다. 지수 기준은 최신 리베이스(2026.7.6 = 100)라 수준값이 아니라 변화율로만 쓴다.
- 월간 표(408/DT_30404_A012)는 KOSIS에 2026-01까지만 있어 주간 표를 쓴다 (2026-11 확인).
- 저장 (data/external/rone_weekly/):
  - apt_sale_index_weekly_gu.csv · apt_jeonse_index_weekly_gu.csv (자치구, 주(월요일), 지수, 단위, 수집일)
  - apt_supply_demand_weekly_seoul.csv (권역, 주, 매매수급, 전세수급, 수집일) — 권역: 서울 · 도심권 · 동북권 · 서북권 · 서남권 · 동남권

실행: python notebook/v2_5year_notebook/eda/v2/01b_collect_apt_weekly_index.py
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
OUT_DIR = PROJECT_DIR / "data" / "external" / "rone_weekly"
API_KEY = dotenv_values(PROJECT_DIR / ".env")["KSKILL_KOSIS_API_KEY"]
BASE = "https://kosis.kr/openapi"
ORG = "408"
INDEX_TABLES = {"DT_304004_WEEK_002_C": "apt_sale_index_weekly_gu.csv", "DT_304004_WEEK_004_C": "apt_jeonse_index_weekly_gu.csv"}
SD_TABLE, SD_OUT = "DT_304004_WEEK_013_C", "apt_supply_demand_weekly_seoul.csv"
PERIODS = [("20200101", "20231231"), ("20240101", "20261231")]  # 끝은 넉넉히: KOSIS가 있는 주까지만 준다

# KOSIS 지역 코드 (메타데이터 ITM_ID) → 자치구
GU_CODE = {
    "a7010101": "종로구", "a7010102": "중구", "a7010103": "용산구", "a7010201": "성동구", "a7010202": "광진구",
    "a7010203": "동대문구", "a7010204": "중랑구", "a7010205": "성북구", "a7010206": "강북구", "a7010207": "도봉구",
    "a7010208": "노원구", "a7010301": "은평구", "a7010302": "서대문구", "a7010303": "마포구", "a7020101": "양천구",
    "a7020102": "강서구", "a7020103": "구로구", "a7020104": "금천구", "a7020105": "영등포구", "a7020106": "동작구",
    "a7020107": "관악구", "a7020201": "서초구", "a7020202": "강남구", "a7020203": "송파구", "a7020204": "강동구",
}
# 수급동향 표의 지역 코드 (가격지수 표와 체계가 다름)
SD_REGION = {"a5": "서울", "a502": "도심권", "a503": "동북권", "a504": "서북권", "a506": "서남권", "a507": "동남권"}


def get(path, **params):
    last = ""
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


def region_names(tbl, obj_id):
    meta = pd.DataFrame(get("statisticsData.do", method="getMeta", type="ITM", orgId=ORG, tblId=tbl))
    return meta[meta["OBJ_ID"].eq(obj_id)].set_index("ITM_ID")["ITM_NM"]


def fetch(tbl, code):
    data = []
    for start, end in PERIODS:
        data += get("Param/statisticsParameterData.do", method="getList", orgId=ORG, tblId=tbl, prdSe="IR",
                    startPrdDe=start, endPrdDe=end, itmId="ALL", objL1=code)
        time.sleep(0.2)
    return data


def save(out, name):
    out["수집일"] = date.today().isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    print(f"저장 {(OUT_DIR / name).relative_to(PROJECT_DIR)}: {len(out):,}행, 주 {out['주'].min()} ~ {out['주'].max()}")


def main():
    for tbl, name in INDEX_TABLES.items():
        names = region_names(tbl, "region")
        for code, gu in GU_CODE.items():  # 코드 ↔ 이름 확인 (예: a7020202 = 강남)
            assert gu.startswith(names[code].rstrip("구")) or names[code] == gu, (code, names[code], gu)
        rows = [{"자치구": gu, "주": x["PRD_DE"], "지수": float(x["DT"]), "단위": x.get("UNIT_NM")}
                for code, gu in GU_CODE.items() for x in fetch(tbl, code)]
        out = pd.DataFrame(rows).drop_duplicates(["자치구", "주"]).sort_values(["자치구", "주"])
        assert out.groupby("자치구").size().nunique() == 1, "구마다 주 수가 다름"
        save(out, name)

    names = region_names(SD_TABLE, "region_2")
    for code, region in SD_REGION.items():
        assert names[code] == region, (code, names[code], region)
    rows = [{"권역": region, "주": x["PRD_DE"], "항목": x["ITM_NM"], "값": float(x["DT"])}
            for code, region in SD_REGION.items() for x in fetch(SD_TABLE, code)]
    sd = (pd.DataFrame(rows).drop_duplicates(["권역", "주", "항목"])
          .pivot(index=["권역", "주"], columns="항목", values="값").reset_index()[["권역", "주", "매매수급", "전세수급"]])
    assert sd.groupby("권역").size().nunique() == 1, "권역마다 주 수가 다름"
    save(sd.sort_values(["권역", "주"]), SD_OUT)


if __name__ == "__main__":
    main()
