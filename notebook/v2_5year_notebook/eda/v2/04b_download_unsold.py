"""
국토교통부 미분양주택현황보고 월별 원본 파일 수집 (통계누리 hRsId=32, 시·군·구별 미분양현황)

- 통계누리 메타 페이지(statMetaView.do?hRsId=32&hFormId=2082)의 첨부 목록을 읽어
  2019-01 ~ 2026-08 월별 파일을 받는다.
  * 2019~2021: 연도별 zip("YYYY년 미분양주택현황(월별).zip") -> 월별 xlsx로 풀어서 저장
  * 2022-01 이후: 월별 xlsx("미분양주택현황(YYYY년M월말)_통계누리.xlsx")
- zip 내부 파일명은 cp437로 깨져 있으므로 cp949로 다시 디코딩한다.
- 저장: data/external/unsold_raw/unsold_YYYYMM.xlsx, 원본 zip은 unsold_zip_YYYY.zip
- 이미 있는 파일은 다시 받지 않는다. 요청 사이에 지연을 둔다.
- _manifest.csv: ref_month, file_name, source_url, posted_date, posted_date_source
  * 통계누리 첨부 목록에는 게시일이 표시되지 않는다. posted_date는 파일 내부 날짜다
    (xlsx는 docProps/core.xml의 modified, zip 내부 파일은 zip 항목 수정시각).
    실제 대외 공표는 '○월 주택통계' 보도자료(기준월 익월 말)다.

실행: python notebook/v2_5year_notebook/eda/v2/04b_download_unsold.py
"""

import csv
import io
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, urlencode

import requests


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
RAW_DIR = PROJECT_DIR / "data" / "external" / "unsold_raw"
MANIFEST_PATH = RAW_DIR / "_manifest.csv"

BASE = "https://stat.molit.go.kr"
META_URL = f"{BASE}/portal/cate/statMetaView.do?hRsId=32&hFormId=2082"
DOWNLOAD_URL = f"{BASE}/portal/common/downLoadFile.do"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": META_URL,
}
DELAY = 2.0  # 요청 사이 지연(초)

START, END = (2019, 1), (2026, 8)
ZIP_YEARS = [2019, 2020, 2021]

session = requests.Session()
session.headers.update(HEADERS)


def month_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def download_url(file_name):
    query = urlencode(
        {"oFileName": file_name, "rFileName": file_name, "midpath": "/stat_file/"},
        quote_via=quote,
    )
    return f"{DOWNLOAD_URL}?{query}"


def fetch(file_name):
    """첨부 파일을 받는다. 없는 파일이면 None (서버가 '파일이 없습니다' HTML을 돌려준다)."""
    time.sleep(DELAY)
    resp = session.get(download_url(file_name), timeout=300)
    resp.raise_for_status()
    if not resp.content.startswith(b"PK"):  # xlsx/zip 모두 zip 시그니처
        return None
    return resp.content


def list_attachments():
    resp = session.get(META_URL, timeout=60)
    resp.raise_for_status()
    return re.findall(r"downFile\('([^']+)'", resp.text)


def ref_month_from_name(name):
    m = re.search(r"\((\d{4})년\s*(\d{1,2})월말\)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def decode_member_name(info):
    if info.flag_bits & 0x800:  # UTF-8 플래그가 있으면 그대로
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def xlsx_modified_date(content):
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            core = zf.read("docProps/core.xml").decode("utf-8", "ignore")
        m = re.search(r"<dcterms:modified[^>]*>(\d{4}-\d{2}-\d{2})", core)
        return m.group(1) if m else ""
    except (KeyError, zipfile.BadZipFile):
        return ""


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    attachments = list_attachments()
    print(f"첨부 {len(attachments)}건 확인")

    records = {}  # (y, m) -> dict

    # 1) 2019~2021 연도별 zip
    for year in ZIP_YEARS:
        zip_name = next(
            (a for a in attachments if a.startswith(f"{year}") and "미분양주택현황" in a and a.endswith(".zip")),
            None,
        )
        if zip_name is None:
            print(f"[누락] {year}년 zip이 첨부 목록에 없음")
            continue
        zip_path = RAW_DIR / f"unsold_zip_{year}.zip"
        if not zip_path.exists():
            content = fetch(zip_name)
            if content is None:
                print(f"[누락] {zip_name} 다운로드 실패")
                continue
            zip_path.write_bytes(content)
            print(f"받음: {zip_name} -> {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            members = []
            for info in zf.infolist():
                name = decode_member_name(info)
                ym = ref_month_from_name(name)
                if ym and name.lower().endswith(".xlsx"):
                    members.append((ym, "(수정)" in name, name, info))
            # 같은 월이 여러 개면 '(수정)' 파일을 우선한다 (2019~2021 zip에는 월당 1개뿐이었음)
            members.sort(key=lambda x: (x[0], x[1]))
            for ym, _, name, info in members:
                out = RAW_DIR / f"unsold_{ym[0]}{ym[1]:02d}.xlsx"
                if not out.exists() or records.get(ym) is not None:
                    out.write_bytes(zf.read(info))
                d = info.date_time
                records[ym] = {
                    "file_name": out.name,
                    "source_url": f"{download_url(zip_name)}#{name}",
                    "posted_date": f"{d[0]:04d}-{d[1]:02d}-{d[2]:02d}",
                    "posted_date_source": "zip_member_mtime",
                }

    # 2) 2022-01 이후 월별 xlsx
    for y, m in month_range((2022, 1), END):
        name = f"미분양주택현황({y}년{m}월말)_통계누리.xlsx"
        candidates = [a for a in attachments if ref_month_from_name(a) == (y, m)]
        if name not in candidates and candidates:
            name = candidates[0]
        out = RAW_DIR / f"unsold_{y}{m:02d}.xlsx"
        if out.exists():
            content = out.read_bytes()
        elif candidates:
            content = fetch(name)
            if content is not None:
                out.write_bytes(content)
                print(f"받음: {name} -> {out.name}")
        else:
            content = None
        if content is None:
            print(f"[누락] {y}-{m:02d}")
            continue
        records[(y, m)] = {
            "file_name": out.name,
            "source_url": download_url(name),
            "posted_date": xlsx_modified_date(content),
            "posted_date_source": "xlsx_core_modified",
        }

    # 3) manifest
    missing = []
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_month", "file_name", "source_url", "posted_date", "posted_date_source"])
        for y, m in month_range(START, END):
            rec = records.get((y, m))
            if rec is None:
                missing.append(f"{y}-{m:02d}")
                writer.writerow([f"{y}-{m:02d}", "", "", "", ""])
            else:
                writer.writerow([f"{y}-{m:02d}", rec["file_name"], rec["source_url"],
                                 rec["posted_date"], rec["posted_date_source"]])
    print(f"manifest 저장: {MANIFEST_PATH} (누락 {len(missing)}개월: {missing})")


if __name__ == "__main__":
    main()
