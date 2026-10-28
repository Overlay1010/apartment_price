"""
서울 대규모점포 좌표 보완 (지방행정 인허가 대규모점포 + 카카오 로컬 API, KAKAO_REST_API_KEY)

- 06 Feature Merge는 거래일에 영업 중이던 점포만으로 최근접 거리를 다시 계산한다.
  V1은 현재 영업 중인 점포만 좌표를 보완했으므로, 폐업·휴업 점포까지 포함해 좌표를 만든다.
- 원본 좌표(EPSG:5174)가 없거나 서울 범위를 벗어난 점포만 V1과 같은 순서로 지오코딩한다:
  도로명주소 → 지번주소 → '서울 {사업장명}' 키워드 검색 → (재개발로 번지가 사라진 경우) 같은 도로의 가까운 건물번호.
- 결과: data/external/large_store_coords.csv (관리번호별 위도·경도·좌표출처). 다시 실행하면 캐시를 덮어쓴다.

실행: python notebook/v2_5year_notebook/eda/v2/06b_geocode_large_stores.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values
from pyproj import Transformer


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
OUTPUT_PATH = EXTERNAL_DIR / "large_store_coords.csv"
KAKAO_KEY = dotenv_values(PROJECT_DIR / ".env")["KAKAO_REST_API_KEY"]
HEADERS = {"Authorization": f"KakaoAK {KAKAO_KEY}"}


def valid_seoul(lat, lon):
    return lat.between(37.4, 37.8) & lon.between(126.7, 127.3)


def clean_address(address):
    if pd.isna(address):
        return None
    address = re.sub(r"\([^)]*\)", "", str(address).strip())
    address = re.sub(r"\s+", " ", address).split(",")[0].strip()
    return address or None


def kakao(endpoint, query):
    time.sleep(0.2)
    r = requests.get(f"https://dapi.kakao.com/v2/local/search/{endpoint}.json",
                     headers=HEADERS, params={"query": query}, timeout=10)
    r.raise_for_status()
    docs = r.json().get("documents", [])
    if endpoint == "keyword":
        seoul = [d for d in docs if str(d.get("road_address_name", "")).startswith("서울")
                 or str(d.get("address_name", "")).startswith("서울")]
        docs = seoul or docs
    return (float(docs[0]["y"]), float(docs[0]["x"])) if docs else (None, None)


def main():
    raw = pd.read_csv(EXTERNAL_DIR / "large_scale_store.csv", encoding="cp949", low_memory=False)
    seoul = raw["도로명주소"].str.contains("서울특별시", na=False) | raw["지번주소"].str.contains("서울특별시", na=False)
    stores = raw[seoul & raw["점포구분명"].eq("대규모점포")].copy()

    x = pd.to_numeric(stores["좌표정보(X)"], errors="coerce")
    y = pd.to_numeric(stores["좌표정보(Y)"], errors="coerce")
    lon, lat = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True).transform(x.to_numpy(), y.to_numpy())
    stores["위도"], stores["경도"] = lat, lon
    stores["좌표출처"] = "원본"

    todo = stores.index[~valid_seoul(stores["위도"], stores["경도"])]
    print(f"서울 대규모점포 {len(stores)}곳 중 지오코딩 대상 {len(todo)}곳")
    for idx in todo:
        row = stores.loc[idx]
        lat, lon, source = None, None, None
        for endpoint, query, label in [("address", clean_address(row["도로명주소"]), "도로명"),
                                       ("address", clean_address(row["지번주소"]), "지번"),
                                       ("keyword", f"서울 {row['사업장명']}", "키워드")]:
            if query:
                lat, lon = kakao(endpoint, query)
                if lat is not None:
                    source = label
                    break
        if lat is None:
            # 재개발 등으로 건물번호가 사라진 경우: 같은 도로의 가까운 건물번호로 근사 (±20번 안)
            m = re.match(r"^(.+[로길])\s*(\d+)", clean_address(row["도로명주소"]) or "")
            if m:
                road, num = m.group(1), int(m.group(2))
                for delta in [d for k in range(1, 21) for d in (-k, k)]:
                    if num + delta > 0:
                        lat, lon = kakao("address", f"{road} {num + delta}")
                        if lat is not None:
                            source = f"인근번지({num + delta})"
                            break
        stores.loc[idx, ["위도", "경도", "좌표출처"]] = [lat, lon, source or "실패"]
        print(f"  {row['사업장명']} ({row['영업상태명']}): {source or '실패'}")

    out = stores[["관리번호", "사업장명", "영업상태명", "위도", "경도", "좌표출처"]]
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT_PATH} {out.shape}")
    print(out["좌표출처"].value_counts().to_string())


if __name__ == "__main__":
    main()
