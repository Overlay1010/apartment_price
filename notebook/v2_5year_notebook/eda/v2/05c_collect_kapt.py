"""
K-apt 공동주택 기본 · 상세 정보 수집 (국토교통부, data.go.kr 15057332 단지 목록 · 15058453 기본 정보, DATA_GO_KR_API_KEY)

- 서울 25개 구의 의무관리 공동주택 단지 목록(단지코드) → 단지별 기본 정보 · 상세 정보
- 기본: 분양형태, 난방방식, 복도유형, 시공사 · 시행사, 사용승인일, 최고층, 세대수, 전용면적대별 세대수
- 상세: 주차대수(지상 · 지하), CCTV, 승강기, 경비 · 청소 관리방식, 버스 · 지하철 도보 시간, 전기차 충전기
- K-apt는 현재 값만 준다(이력 없음). 시공사 · 복도유형처럼 바뀌지 않는 속성만 모델 후보로 쓰고, 관리 항목은 화면 표시용으로 둔다.
- 단지별 응답은 raw/ 에 JSON으로 남겨 중단 후 다시 돌려도 받은 단지는 건너뛴다.

실행: python notebook/v2_5year_notebook/eda/v2/05c_collect_kapt.py
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
OUT_DIR = PROJECT_DIR / "data" / "external" / "kapt"
RAW_DIR = OUT_DIR / "raw"
SNAPSHOT = date.today().strftime("%Y%m%d")
KEY = dotenv_values(PROJECT_DIR / ".env")["DATA_GO_KR_API_KEY"]
BASE = "https://apis.data.go.kr/1613000"
LIST_URL = f"{BASE}/AptListService4/getSigunguAptList4"
INFO_URLS = {"basic": f"{BASE}/AptBasisInfoServiceV5/getAphusBassInfoV5",
             "detail": f"{BASE}/AptBasisInfoServiceV5/getAphusDtlInfoV5"}
SEOUL_GU = ["11110", "11140", "11170", "11200", "11215", "11230", "11260", "11290", "11305", "11320", "11350", "11380", "11410",
            "11440", "11470", "11500", "11530", "11545", "11560", "11590", "11620", "11650", "11680", "11710", "11740"]
PER_PAGE = 1000


def get(url, **params):
    # 게이트웨이 일시 오류(OpenAPI_ServiceResponse 04 HTTP_ERROR 등)는 기다렸다 다시 시도한다
    for attempt in range(8):
        try:
            r = requests.get(url, params={"serviceKey": KEY, "_type": "json", **params}, timeout=60)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 120))
            continue
        if r.status_code == 403 and "NOT_REGISTERED" in r.text:
            raise SystemExit(f"인증 실패: {url.split('/')[-2]} 활용신청 승인(또는 반영 대기)이 필요합니다.")
        if r.ok and "response" in r.text[:50]:
            body = r.json()["response"]
            assert body["header"]["resultCode"] == "00", body["header"]
            return body["body"]
        time.sleep(min(2 ** attempt, 120))
    raise RuntimeError(f"{url.split('/')[-1]} {params} 응답 실패: {r.text[:200]}")


def as_list(items):
    if not items:
        return []
    items = items.get("item", items) if isinstance(items, dict) else items
    return items if isinstance(items, list) else [items]


def collect_list():
    rows = []
    for gu in SEOUL_GU:
        page = 1
        while True:
            body = get(LIST_URL, sigunguCode=gu, pageNo=page, numOfRows=PER_PAGE)
            items = as_list(body.get("items"))
            rows += [{**it, "sigunguCode": gu} for it in items]
            if page * PER_PAGE >= int(body["totalCount"]):
                break
            page += 1
        time.sleep(0.2)
    data = pd.DataFrame(rows).drop_duplicates("kaptCode")
    print("서울 단지 목록:", len(data), "| 구별", data.groupby("sigunguCode").size().to_dict())
    return data


def collect_info(codes):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(codes):
        path = RAW_DIR / f"{code}.json"
        if path.exists():
            continue
        out = {kind: (get(url, kaptCode=code).get("item") or {}) for kind, url in INFO_URLS.items()}
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        if i % 200 == 0:
            print(f"  {i}/{len(codes)}", flush=True)
        time.sleep(0.5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apt_list = collect_list()
    collect_info(apt_list["kaptCode"].tolist())

    raw = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in RAW_DIR.glob("*.json")}
    for kind in INFO_URLS:
        data = pd.DataFrame([{**v[kind], "kaptCode": k} for k, v in raw.items() if k in set(apt_list["kaptCode"])])
        data["snapshot_date"] = SNAPSHOT
        path = OUT_DIR / f"kapt_{kind}_seoul_{SNAPSHOT}.parquet"
        data.astype(str).to_parquet(path, index=False)
        print(f"저장 {path.name}: {data.shape}")
    apt_list.astype(str).assign(snapshot_date=SNAPSHOT).to_parquet(OUT_DIR / f"kapt_list_seoul_{SNAPSHOT}.parquet", index=False)


if __name__ == "__main__":
    main()
