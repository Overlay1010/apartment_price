"""
서울 주택인허가 기본개요 수집 (국토교통부 건축HUB 주택인허가정보, data.go.kr 15136560)

- API는 시군구코드 + 법정동코드로만 조회된다. 서울 법정동코드(10100~19900, 100 단위)를
  구마다 1건씩 조회해 기록이 있는 동을 찾은 뒤, 그 동의 전체 페이지를 받는다.
- 동 단위로 JSONL을 저장하므로 중간에 끊겨도 다시 실행하면 이어서 받는다.
- 원장은 현재 시점 snapshot이다. 파일명에 수집일을 남겨 이후 vintage를 쌓는다.

실행: python notebook/v2_5year_notebook/eda/v2/04a_collect_housing_permit.py
"""

import json
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
RAW_DIR = EXTERNAL_DIR / "housing_permit_raw"
SNAPSHOT = date.today().strftime("%Y%m%d")
OUTPUT_PATH = EXTERNAL_DIR / f"housing_permit_basis_seoul_{SNAPSHOT}.parquet"

URL = "https://apis.data.go.kr/1613000/HsPmsHubService/getHpBasisOulnInfo"
HEADERS = {"User-Agent": "Mozilla/5.0"}
PAGE_SIZE = 100  # API 최대
MIN_INTERVAL = 0.4  # 초당 호출 제한(429) 회피용 최소 간격(초)

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
            r = session.get(URL, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            r.raise_for_status()
            body = r.json()["response"]["body"]
            items = body.get("items") or {}
            items = items.get("item", []) if isinstance(items, dict) else []
            items = items if isinstance(items, list) else [items]
            return int(body.get("totalCount") or 0), items
        except Exception as e:  # 일시 오류는 재시도, 키는 메시지에 남기지 않는다
            msg = str(e).replace(SERVICE_KEY, "<KEY>")
            wait = 30 if "429" in msg else 2 ** attempt
            print(f"  재시도 {attempt + 1}/{retries} ({sigungu}-{bjdong} p{page}): {msg[:120]} → {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"요청 실패: {sigungu}-{bjdong} page {page}")


def discover_dongs():
    # 구 단위로 탐색 결과를 저장해 두고, 다시 실행하면 끝난 구는 건너뛴다.
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
    path = RAW_DIR / f"{sigungu}_{bjdong}.jsonl"
    if path.exists() and sum(1 for _ in path.open(encoding="utf-8")) == total:
        return total

    items = []
    for page in range(1, (total + PAGE_SIZE - 1) // PAGE_SIZE + 1):
        _, page_items = call(sigungu, bjdong, page=page)
        items.extend(page_items)

    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def read_raw():
    # 관리번호(mgmHsrgstPk)가 커서 pd.read_json이 실패하므로 json으로 읽고 모든 값을 문자열로 둔다.
    records = []
    for path in sorted(RAW_DIR.glob("11*.jsonl")):
        with path.open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return pd.DataFrame(records).astype(str)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index = discover_dongs()
    print(f"기록이 있는 법정동: {len(index)}개, 예상 기록 수 {index['totalCount'].astype(int).sum():,}")

    mismatched = []
    for i, r in enumerate(index.itertuples(), 1):
        got = collect_dong(r.sigunguCd, r.bjdongCd, int(r.totalCount))
        if got != int(r.totalCount):
            mismatched.append((r.sigunguCd, r.bjdongCd, r.totalCount, got))
        if i % 25 == 0:
            print(f"  {i}/{len(index)} 동 완료")

    data = read_raw()
    data["자치구"] = data["sigunguCd"].astype(str).map(SEOUL_GU)
    data["snapshot_date"] = SNAPSHOT
    data.to_parquet(OUTPUT_PATH, index=False)

    print(f"저장: {OUTPUT_PATH} {data.shape}")
    print("건수 불일치 동:", mismatched or "없음")


if __name__ == "__main__":
    main()
