"""
청약홈 APT 분양정보 수집 (한국부동산원_청약홈 분양정보 조회 서비스, data.go.kr 15098547, DATA_GO_KR_API_KEY)

- 서울(공급지역명 = 서울) APT 분양공고 전체: 민영·국민(LH·SH 공공분양 포함)·민간사전청약·신혼희망타운
- 공고일, 공급규모(세대수), 공급위치, 입주예정월, 정비사업·공공주택지구 여부 등을 저장한다.
- 청약홈은 2020-02 개시라 그 이후 공고만 있다.

실행: python notebook/v2_5year_notebook/eda/v2/04f_collect_applyhome.py
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
OUT_DIR = PROJECT_DIR / "data" / "external" / "applyhome"
SNAPSHOT = date.today().strftime("%Y%m%d")
URL = "https://api.odcloud.kr/api/ApplyhomeInfoDetailSvc/v1/getAPTLttotPblancDetail"
KEY = dotenv_values(PROJECT_DIR / ".env")["DATA_GO_KR_API_KEY"]
PER_PAGE = 500


def fetch_page(page):
    params = {"serviceKey": KEY, "page": page, "perPage": PER_PAGE, "returnType": "JSON",
              "cond[SUBSCRPT_AREA_CODE_NM::EQ]": "서울"}
    for attempt in range(5):
        r = requests.get(URL, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        if r.status_code == 401:
            raise SystemExit("인증 실패: data.go.kr에서 15098547(청약홈 분양정보 조회 서비스) 활용신청 승인이 필요합니다.")
        if r.ok:
            return r.json()
        time.sleep(2 ** attempt)
    r.raise_for_status()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = fetch_page(1)
    total = int(first["matchCount"])
    rows = list(first["data"])
    for page in range(2, (total + PER_PAGE - 1) // PER_PAGE + 1):
        rows += fetch_page(page)["data"]
        time.sleep(0.3)

    data = pd.DataFrame(rows).astype(str)
    data["snapshot_date"] = SNAPSHOT
    assert len(data) == total, (len(data), total)
    path = OUT_DIR / f"apt_announcements_seoul_{SNAPSHOT}.parquet"
    data.to_parquet(path, index=False)
    print(f"저장: {path} {data.shape}, 공고일 {data['RCRIT_PBLANC_DE'].min()} ~ {data['RCRIT_PBLANC_DE'].max()}")


if __name__ == "__main__":
    main()
