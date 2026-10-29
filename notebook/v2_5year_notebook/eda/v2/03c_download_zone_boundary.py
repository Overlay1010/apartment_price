"""
성수 전략정비구역 경계 (서울 열린데이터 OA-20957 '서울시 의제처리구역 위치정보', UPIS UQ181 SHP)

- 03 Policy의 토지거래허가구역 '성수 전략정비구역 1~4지구'(2021-04-27~)는 구역 안의 모든 건물이 허가 대상이다.
  동 단위 구간만으로는 성수동1가·2가 단지 중 어느 단지가 구역 안인지 알 수 없어, 구역 경계 × 단지 좌표로 판정한다.
- 지정 직후 버전(2021-05-27 등록, 파일 seq=1)과 최신 버전(seq=10)을 받는다.
  2021년 버전은 구역 전체가 다각형 1개, 2025년 이후 버전은 1~4지구로 나뉘어 있다. 경계는 2011년 결정고시 이후 같다.
- SHP 문자 인코딩은 cp949, 좌표계는 .prj 그대로 읽어 WGS84로 바꾼다.
- 결과: data/external/jeongbi/seongsu_strategic_zone.geojson (버전별 다각형, 속성: 버전·구역명·결정고시관리코드)
  원본 ZIP은 시스템 임시폴더에 받고 지운다.

실행: python notebook/v2_5year_notebook/eda/v2/03c_download_zone_boundary.py
"""

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests
import shapefile
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
OUTPUT_PATH = PROJECT_DIR / "data" / "external" / "jeongbi" / "seongsu_strategic_zone.geojson"

PAGE_URL = "https://data.seoul.go.kr/dataList/OA-20957/F/1/datasetView.do"
DOWNLOAD_URL = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false"
VERSIONS = {1: "2021-05-27", 10: "2026-09-11"}  # 파일 seq: 등록일 (페이지 파일 목록 기준)
ZONE_KEYWORD = "성수전략정비구역"


def download(seq, folder):
    r = requests.post(DOWNLOAD_URL, data={"infId": "OA-20957", "seqNo": "", "seq": str(seq), "infSeq": "1"},
                      headers={"Referer": PAGE_URL, "User-Agent": "Mozilla/5.0"}, timeout=300)
    r.raise_for_status()
    path = folder / f"uq181_{seq}.zip"
    path.write_bytes(r.content)
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"ZIP이 아님: seq={seq} ({len(r.content)} bytes)")
    out = folder / f"v{seq}"
    zipfile.ZipFile(path).extractall(out)
    return next(out.rglob("*.shp"))


def zone_polygons(shp_path):
    crs = CRS.from_user_input(shp_path.with_suffix(".prj").read_text())
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    zones = {}
    with shapefile.Reader(str(shp_path.with_suffix("")), encoding="cp949") as reader:
        for sr in reader.iterShapeRecords():
            name = sr.record["DGM_NM"].strip()
            if ZONE_KEYWORD in name:
                geom = shapely.transform(shape(sr.shape.__geo_interface__).buffer(0),
                                         lambda xy: np.column_stack(to_wgs84.transform(xy[:, 0], xy[:, 1])))
                zones.setdefault((name, sr.record["NTFC_SN"]), []).append(geom)
    return {key: unary_union(geoms) for key, geoms in zones.items()}  # 같은 구역이 여러 행이면 합친다


def main():
    features = []
    with tempfile.TemporaryDirectory() as tmp:
        for seq, registered in VERSIONS.items():
            zones = zone_polygons(download(seq, Path(tmp)))
            print(f"버전 {registered}: {len(zones)}개 구역")
            for (name, notice), geom in sorted(zones.items()):
                print(f"  {name} ({notice}) 면적(도 단위) {geom.area:.6f}")
                features.append({"type": "Feature", "geometry": mapping(geom),
                                 "properties": {"버전": registered, "구역명": name, "결정고시관리코드": notice}})
    OUTPUT_PATH.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
                           encoding="utf-8")
    print(f"저장: {OUTPUT_PATH} ({len(features)}개 다각형)")


if __name__ == "__main__":
    main()
