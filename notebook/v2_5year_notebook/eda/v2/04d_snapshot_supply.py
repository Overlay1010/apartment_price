"""
공급 데이터 snapshot 쌓기 (모델에서 시점 복원이 안 되는 자료를 앞으로 회차별로 보관)

1. 주택인허가 원장: 04a_collect_housing_permit.py 를 다시 돌리면
   data/external/housing_permit_basis_seoul_YYYYMMDD.parquet 로 새 snapshot이 생긴다.
   (원본 JSONL 폴더는 매번 새로 받아야 하므로 먼저 비운다: --fresh)
2. 입주예정물량 (한국부동산원·R114, data.go.kr 15111714): 공개 회차마다 받아
   data/external/ipju/ipju_api_YYYYMMDD.parquet 로 저장한다. 직전 snapshot과 같으면 저장하지 않는다.
   * data.go.kr에서 15111714 활용신청이 필요하다 (DATA_GO_KR_API_KEY 사용).

권장 주기: 원장은 매월 1회, 입주예정물량은 공개 직후(2월 말·8월 말 전후).

실행:
    python notebook/v2_5year_notebook/eda/v2/04d_snapshot_supply.py ipju
    python notebook/v2_5year_notebook/eda/v2/04d_snapshot_supply.py permit --fresh
"""

import subprocess
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
IPJU_DIR = EXTERNAL_DIR / "ipju"
TODAY = date.today().strftime("%Y%m%d")

IPJU_URL = "https://api.odcloud.kr/api/15111714/v1/uddi:0b257760-ac19-4841-adb4-b38b4d153397"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def snapshot_ipju():
    key = dotenv_values(PROJECT_DIR / ".env")["DATA_GO_KR_API_KEY"]
    records, page = [], 1
    while True:
        r = requests.get(
            IPJU_URL,
            params={"serviceKey": key, "page": page, "perPage": 1000, "returnType": "JSON"},
            headers=HEADERS, timeout=30,
        )
        if r.status_code == 401:
            sys.exit("입주예정물량 API 인증 실패: data.go.kr에서 15111714 활용신청이 필요합니다.")
        r.raise_for_status()
        body = r.json()
        records.extend(body.get("data", []))
        if page * 1000 >= int(body.get("totalCount", 0)):
            break
        page += 1
        time.sleep(0.5)

    data = pd.DataFrame(records).astype(str)
    previous = sorted(IPJU_DIR.glob("ipju_api_*.parquet"))
    if previous and pd.read_parquet(previous[-1]).drop(columns="snapshot_date").equals(data):
        print(f"직전 snapshot({previous[-1].name})과 같아 저장하지 않습니다.")
        return

    data["snapshot_date"] = TODAY
    IPJU_DIR.mkdir(parents=True, exist_ok=True)
    path = IPJU_DIR / f"ipju_api_{TODAY}.parquet"
    data.to_parquet(path, index=False)
    print(f"저장: {path} {data.shape}")


def snapshot_permit(fresh):
    raw_dir = EXTERNAL_DIR / "housing_permit_raw"
    if fresh and raw_dir.exists():
        # 동별 JSONL은 재사용하면 이전 snapshot이 섞이므로 지운다. 법정동 목록(_dong_index_*)은 남겨 탐색을 건너뛴다.
        for path in raw_dir.glob("11*.jsonl"):
            path.unlink()
    collector = Path(__file__).with_name("04a_collect_housing_permit.py")
    subprocess.run([sys.executable, str(collector)], check=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ipju"
    if target == "ipju":
        snapshot_ipju()
    elif target == "permit":
        snapshot_permit(fresh="--fresh" in sys.argv)
    else:
        sys.exit("사용법: 04d_snapshot_supply.py [ipju | permit --fresh]")
