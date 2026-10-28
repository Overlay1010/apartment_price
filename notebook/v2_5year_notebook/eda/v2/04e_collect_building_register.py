"""
서울 건축물대장 표제부 수집 (국토교통부 건축HUB 건축물대장정보, BldRgstHubService/getBrTitleInfo)

- 용도: 04 구 단위 준공(사용승인) 물량, 05b 단지 속성(층수·주차·승강기 등)
- API는 시군구코드 + 법정동코드로만 조회된다. 서울 법정동코드(10100~19900)를 구마다 1건씩 조회해
  기록이 있는 동을 찾고, 그 동의 전체 페이지를 받는다.
- 동 단위 parquet으로 저장하므로 끊겨도 다시 실행하면 이어서 받는다 (일일 호출 한도에 걸리면 다음 날 재실행).
- 원장과 마찬가지로 현재 snapshot이다. 파일명에 수집일을 남긴다.

- 총괄표제부(getBrRecapTitleInfo, 단지 전체 주차대수·동수 등)도 받을 수 있다: 인자 recap.
  법정동 목록은 표제부 탐색 결과(_dong_index_*.csv)를 재사용한다.

실행: python notebook/v2_5year_notebook/eda/v2/04e_collect_building_register.py [title|recap]
"""

import sys

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
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
KIND = sys.argv[1] if len(sys.argv) > 1 else "title"
OPERATIONS = {"title": "getBrTitleInfo", "recap": "getBrRecapTitleInfo"}
INDEX_DIR = EXTERNAL_DIR / "building_register_raw"  # 법정동 목록은 표제부 탐색 결과를 공용으로 쓴다
RAW_DIR = INDEX_DIR if KIND == "title" else EXTERNAL_DIR / f"building_register_{KIND}_raw"
SNAPSHOT = date.today().strftime("%Y%m%d")
OUTPUT_PATH = EXTERNAL_DIR / f"building_register_{KIND}_seoul_{SNAPSHOT}.parquet"

URL = f"https://apis.data.go.kr/1613000/BldRgstHubService/{OPERATIONS[KIND]}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAGE_SIZE = 100
MIN_INTERVAL = 0.4

SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구",
    "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구",
    "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구",
    "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구",
}
DONG_CODES = [str(c) for c in range(10100, 20000, 100)]

SERVICE_KEY = dotenv_values(PROJECT_DIR / ".env")["DATA_GO_KR_API_KEY"]
session = requests.Session()
_last_call = [0.0]


class DailyLimitError(RuntimeError):
    pass


def call(sigungu, bjdong, page=1, rows=PAGE_SIZE, retries=8):
    params = {
        "serviceKey": SERVICE_KEY, "sigunguCd": sigungu, "bjdongCd": bjdong,
        "numOfRows": rows, "pageNo": page, "_type": "json",
    }
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = session.get(URL, params=params, headers=HEADERS, timeout=60)
            if "LIMITED_NUMBER_OF_SERVICE_REQUESTS_EXCEEDS" in r.text:
                raise DailyLimitError("일일 호출 한도 초과 — 내일 다시 실행하면 이어서 받습니다.")
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            r.raise_for_status()
            body = r.json()["response"]["body"]
            items = body.get("items") or {}
            items = items.get("item", []) if isinstance(items, dict) else []
            items = items if isinstance(items, list) else [items]
            return int(body.get("totalCount") or 0), items
        except DailyLimitError:
            raise
        except Exception as e:
            msg = str(e).replace(SERVICE_KEY, "<KEY>")
            wait = 30 if "429" in msg else 2 ** attempt
            print(f"  재시도 {attempt + 1}/{retries} ({sigungu}-{bjdong} p{page}): {msg[:120]} → {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"요청 실패: {sigungu}-{bjdong} page {page}")


def discover_dongs():
    if KIND != "title":
        # 표제부 탐색에서 찾은 법정동만 다시 조회해 이 기능의 건수를 얻는다
        base = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(INDEX_DIR.glob("_dong_index_*.csv"))])
        path = RAW_DIR / "_dong_index.csv"
        if not path.exists():
            rows = []
            for r in base.itertuples():
                total, _ = call(r.sigunguCd, r.bjdongCd, rows=1)
                if total:
                    rows.append({"sigunguCd": r.sigunguCd, "자치구": r.자치구, "bjdongCd": r.bjdongCd, "totalCount": total})
            pd.DataFrame(rows, columns=["sigunguCd", "자치구", "bjdongCd", "totalCount"]).to_csv(
                path, index=False, encoding="utf-8-sig")
        return pd.read_csv(path, dtype=str)

    parts = []
    for sigungu, gu in SEOUL_GU.items():
        path = RAW_DIR / f"_dong_index_{sigungu}.csv"
        if not path.exists():
            rows = []
            for bjdong in DONG_CODES:
                total, _ = call(sigungu, bjdong, rows=1)
                if total:
                    rows.append({"sigunguCd": sigungu, "자치구": gu, "bjdongCd": bjdong, "totalCount": total})
            pd.DataFrame(rows, columns=["sigunguCd", "자치구", "bjdongCd", "totalCount"]).to_csv(
                path, index=False, encoding="utf-8-sig")
            print(f"{gu}: {len(rows)}개 동")
        parts.append(pd.read_csv(path, dtype=str))
    return pd.concat(parts, ignore_index=True)


def collect_dong(sigungu, bjdong, total):
    path = RAW_DIR / f"{sigungu}_{bjdong}.parquet"
    if path.exists():
        return len(pd.read_parquet(path, columns=["mgmBldrgstPk"]))

    items = []
    for page in range(1, (total + PAGE_SIZE - 1) // PAGE_SIZE + 1):
        try:
            _, page_items = call(sigungu, bjdong, page=page, retries=4)
        except DailyLimitError:
            raise
        except RuntimeError:
            # 서버 시간 초과(SERVICETIMEOUT)가 반복되는 페이지는 20건씩 나눠 받는다
            small, page_items = 20, []
            for sub in range((page - 1) * PAGE_SIZE // small + 1, page * PAGE_SIZE // small + 1):
                _, sub_items = call(sigungu, bjdong, page=sub, rows=small)
                page_items.extend(sub_items)
        items.extend(page_items)

    pd.DataFrame(items).astype(str).to_parquet(path, index=False)
    return len(items)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index = discover_dongs()
    print(f"기록이 있는 법정동: {len(index)}개, 예상 기록 수 {index['totalCount'].astype(int).sum():,}")

    mismatched, failed = [], []
    for i, r in enumerate(index.itertuples(), 1):
        try:
            got = collect_dong(r.sigunguCd, r.bjdongCd, int(r.totalCount))
        except DailyLimitError:
            raise
        except RuntimeError as e:
            failed.append((r.sigunguCd, r.bjdongCd))  # 다시 실행하면 이 동부터 이어서 받는다
            print(f"  건너뜀: {e}")
            continue
        if got != int(r.totalCount):
            mismatched.append((r.sigunguCd, r.bjdongCd, r.totalCount, got))
        if i % 25 == 0:
            print(f"  {i}/{len(index)} 동 완료")

    if failed:
        print(f"실패한 동 {len(failed)}개 — 다시 실행하세요: {failed}")
        return

    data = pd.concat([pd.read_parquet(p) for p in sorted(RAW_DIR.glob("11*.parquet"))], ignore_index=True)
    data["자치구"] = data["sigunguCd"].map(SEOUL_GU)
    data["snapshot_date"] = SNAPSHOT
    data.to_parquet(OUTPUT_PATH, index=False)

    print(f"저장: {OUTPUT_PATH} {data.shape}")
    print("건수 불일치 동:", mismatched or "없음")


if __name__ == "__main__":
    main()
