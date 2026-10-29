"""
지하철역 출입구 좌표 (서울교통빅데이터플랫폼 T Data '지하철 출입구 정보 및 주변정류장 정보', data_id=10045, tnSubwayEntrc.csv)

- 21 군집의 역세권 태그와 06의 `역출입구거리_m`에 쓴다. V1 역세권거리는 역 대표 좌표까지 거리라,
  서울시 역세권 기준(승강장 경계)과 차이가 크다. 출입구 좌표로 실제 걸어서 닿는 지점에 가깝게 잰다.
- 원본 열 이름은 '지하철역X좌표/Y좌표'지만 값은 출입구마다 다르다 (출입구 좌표).
  좌표계는 명시가 없어, 역 좌표 파일(WGS84)과 맞춰 EPSG:5181로 확인했다 (출입구 ↔ 최근접 역 중앙값 약 107m,
  EPSG:5174/2097은 300m 이상 어긋남). 06의 직장인구 행정동 중심좌표와 같은 좌표계다.
- 출입구 1개가 주변 정류장마다 여러 행이라 (역ID, 출입구번호)로 한 행만 남긴다.
- 결과: data/external/subway_entrances.csv (지하철역ID, 출입구번호, X_5181, Y_5181, 위도, 경도, 수집일)
  원본 CSV는 시스템 임시폴더에 받고 지운다. 파일은 매일 갱신되는 현재 목록이다.

실행: python notebook/v2_5year_notebook/eda/v2/06c_download_subway_entrances.py
"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from pyproj import Transformer


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
OUTPUT_PATH = PROJECT_DIR / "data" / "external" / "subway_entrances.csv"

PAGE_URL = "https://t-data.seoul.go.kr/dataprovide/trafficdataviewfile.do?data_id=10045"
DOWNLOAD_URL = "https://t-data.seoul.go.kr/dataprovide/download.do?id=11056"  # 페이지 파일 목록의 tnSubwayEntrc.csv


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tnSubwayEntrc.csv"
        r = requests.get(DOWNLOAD_URL, headers={"Referer": PAGE_URL, "User-Agent": "Mozilla/5.0"}, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    raw = raw.apply(lambda s: s.str.replace("﻿", "", regex=False).str.strip())
    expected = {"지하철역ID", "출입구번호", "지하철역X좌표", "지하철역Y좌표"}
    assert expected <= set(raw.columns), raw.columns.tolist()

    out = raw.drop_duplicates(["지하철역ID", "출입구번호"])[["지하철역ID", "출입구번호", "지하철역X좌표", "지하철역Y좌표"]].copy()
    out["X_5181"] = pd.to_numeric(out.pop("지하철역X좌표"), errors="coerce")
    out["Y_5181"] = pd.to_numeric(out.pop("지하철역Y좌표"), errors="coerce")
    missing = out[["X_5181", "Y_5181"]].isna().any(axis=1)
    out = out[~missing].copy()
    out["경도"], out["위도"] = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True).transform(
        out["X_5181"].to_numpy(), out["Y_5181"].to_numpy())
    out["수집일"] = date.today().isoformat()
    OUTPUT_PATH.write_text(out.to_csv(index=False), encoding="utf-8-sig")
    print(f"원본 {len(raw):,}행 → 출입구 {len(out):,}개 (역 {out['지하철역ID'].nunique()}개, 좌표 결측 제외 {int(missing.sum())}개)")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
