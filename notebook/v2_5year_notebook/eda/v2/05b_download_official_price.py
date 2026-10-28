"""
공동주택 공시가격 (국토교통부_주택 공시가격 정보, data.go.kr 파일데이터 3073746) 연도별 원본 수집 → 서울 아파트만 추출

- 출처: https://www.data.go.kr/data/3073746/fileData.do
  현재 파일(최신 공시연도) + "주기성 과거 데이터" 탭(연도별 과거 버전). 로그인 없이 받을 수 있다.
  과거 목록은 페이지가 JS로 부르는 POST /tcs/dss/selectHistAndCsvData.do,
  버전별 첨부파일 ID는 POST /tcs/dss/selectDpkDetailInfo.do (fn_fileDataDown(...) 인자),
  다운로드는 /tcs/dss/selectFileDataDownload.do → /cmm/cmm/check-limit.json → /cmm/cmm/fileDownload.do 순서.
  check-limit이 needCaptcha=true를 돌려주면 멈추고 브라우저에서 직접 받으라고 안내한다 (CAPTCHA는 풀지 않는다).
- 원본: 전국 공동주택(아파트·연립·다세대) 호 단위, ZIP(100~220MB) 안에 CSV(2.4~3.5GB). 2020~2023은 cp949, 2024~는 UTF-8.
  CSV에는 공동주택 유형(아파트/연립/다세대) 열이 없다. → 서울 행만 남긴 뒤 아파트 여부를 따로 판정한다 (아래 '아파트 판정' 참고).
- 공시연도별 parquet: data/external/official_price/apt_official_price_seoul_YYYY.parquet, manifest: _manifest.csv
- 원본 ZIP은 임시 폴더(--raw-dir, 기본: 시스템 임시폴더/gongsi_raw)에 받고, 추출이 끝나면 지운다 (--keep-raw로 유지).
- 이미 저장된 연도는 건너뛴다 (--force로 다시 만든다).
- 호 단위 키는 (단지코드, 동코드, 호코드). 동명이 단지명으로만 적힌 단지가 있어 (단지코드, 동명, 호명)은 중복될 수 있다.

실행: python notebook/v2_5year_notebook/eda/v2/05b_download_official_price.py [--years 2020 2026] [--raw-dir DIR] [--keep-raw] [--force]
"""

import argparse
import io
import json
import re
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
OUT_DIR = EXTERNAL_DIR / "official_price"
MANIFEST_PATH = OUT_DIR / "_manifest.csv"

BASE = "https://www.data.go.kr"
PUBLIC_DATA_PK = "3073746"
PAGE_URL = f"{BASE}/data/{PUBLIC_DATA_PK}/fileData.do"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": PAGE_URL}
SEOUL = "서울특별시"
CHUNK_ROWS = 1_000_000

# 원본 열 이름 (2025 레이아웃 기준). 연도마다 일부 열이 없을 수 있다 → 있는 것만 남긴다.
KEEP_COLS = [
    "기준연도", "기준월", "법정동코드", "도로명주소", "시도", "시군구", "읍면", "동리",
    "특수지코드", "본번", "부번", "특수지명", "단지명", "동명", "호명", "전용면적", "공시가격",
    "단지코드", "동코드", "호코드", "건축물대장PK", "(구)건축물대장PK",
]
# 공동주택 정기공시일 (1월 1일 기준, 결정·공시일; 국토부 보도자료 기준, 원본 파일에는 없음).
# 이의신청 조정분은 6월 말 조정공시에 반영되며, 포털 파일은 그 뒤에 등록된 버전이다.
NOTICE_DATE = {
    2019: "2019-04-30", 2020: "2020-04-29", 2021: "2021-04-29", 2022: "2022-04-29",
    2023: "2023-04-28", 2024: "2024-04-30", 2025: "2025-04-30", 2026: "2026-04-30",
}

session = requests.Session()
session.headers.update(HEADERS)


def post(path, data, retries=5):
    for attempt in range(retries):
        try:
            r = session.post(f"{BASE}{path}", data=data, timeout=60)
            r.raise_for_status()
            return r.text
        except Exception as e:
            wait = 2 ** attempt
            print(f"  재시도 {attempt + 1}/{retries} ({path}): {str(e)[:120]} → {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"요청 실패: {path}")


def year_from_name(name):
    """'..._20250626' / '..._10/29/2021' / '..._20260101' → 연도 (데이터 기준연도와 같은지는 추출 때 확인)."""
    m = re.search(r"_(\d{4})\d{4}$", name) or re.search(r"_\d{1,2}/\d{1,2}/(\d{4})$", name)
    return int(m.group(1)) if m else None


def discover_versions():
    """현재 파일 + 주기성 과거 데이터 목록 → [{year, name, detail_pk, atch_file_id, ...}]"""
    session.get(PAGE_URL, timeout=60)  # 세션 쿠키
    page = session.get(PAGE_URL, timeout=60).text
    current_pk = re.search(r'id="publicDataDetailPk"[^>]*value="([^"]+)"', page).group(1)

    versions = []
    info = json.loads(post("/tcs/dss/selectFileDataDownload.do", {
        "publicDataPk": PUBLIC_DATA_PK, "publicDataDetailPk": current_pk,
        "atchFileId": "", "fileDetailSn": "1", "publicDataTyCode": "PR0051",
    }))
    d = info["fileDataRegistVO"]
    versions.append({
        "name": d["dataNm"], "detail_pk": current_pk, "atch_file_id": info["atchFileId"],
        "file_detail_sn": info["fileDetailSn"], "original_file_name": d["orginlFileNm"],
        "rows_listed": d["atchFileCo"], "registered": d["registDt"][:10],
    })

    hist = post("/tcs/dss/selectHistAndCsvData.do", {"publicDataPk": PUBLIC_DATA_PK, "publicDataDetailPk": current_pk})
    for pk, sn, name in re.findall(
            r'data-public-pk="([^"]+)"\s*data-public-detail-sn="(\d+)">\s*([^<]+?)\s*<', hist):
        if year_from_name(name) is None:  # 2018년 이전 형식(연도 접두) 제외
            continue
        detail = post("/tcs/dss/selectDpkDetailInfo.do", {"publicDataDetailPk": pk, "publicDataHistSn": sn})
        m = re.search(r"fn_fileDataDown\('\d+', '[^']+', '([^']+)', '(\d+)'", detail)
        info = json.loads(post("/tcs/dss/selectFileDataDownload.do", {
            "publicDataPk": PUBLIC_DATA_PK, "publicDataDetailPk": pk,
            "atchFileId": m.group(1), "fileDetailSn": m.group(2), "publicDataTyCode": "PR0051",
        }))
        d = info["fileDataRegistVO"]
        versions.append({
            "name": name, "detail_pk": pk, "atch_file_id": info["atchFileId"],
            "file_detail_sn": info["fileDetailSn"], "original_file_name": d["orginlFileNm"],
            "rows_listed": d["atchFileCo"], "registered": d["registDt"][:10],
        })
    for v in versions:
        v["year"] = year_from_name(v["name"])
        v["source_url"] = (f"{BASE}/cmm/cmm/fileDownload.do?atchFileId={v['atch_file_id']}"
                           f"&fileDetailSn={v['file_detail_sn']}")
    return versions


def download(v, raw_dir):
    path = raw_dir / f"housing_official_price_{v['year']}_{v['atch_file_id']}.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    limit = json.loads(post("/cmm/cmm/check-limit.json",
                            {"atchFileId": v["atch_file_id"], "fileDetailSn": v["file_detail_sn"]}))
    if limit.get("needCaptcha"):
        raise RuntimeError(
            f"data.go.kr 다운로드 한도 CAPTCHA 발생 — 브라우저에서 {PAGE_URL} → '주기성 과거 데이터' → "
            f"'{v['name']}' → 다운로드 로 직접 받아 {path} 로 저장한 뒤 다시 실행하세요.")
    tmp = path.with_suffix(".part")
    with session.get(v["source_url"] + "&dataNm=official_price", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    if not zipfile.is_zipfile(tmp):
        raise RuntimeError(f"ZIP이 아님: {tmp} ({tmp.stat().st_size} bytes)")
    tmp.rename(path)
    return path


def main_csv_member(zf):
    csvs = [i for i in zf.infolist() if i.filename.lower().endswith(".csv") and "샘플" not in i.filename]
    return max(csvs, key=lambda i: i.file_size)


def detect_encoding(zf, member):
    with zf.open(member) as f:
        head = f.read(1 << 16)
    try:
        head[: head.rfind(b"\n")].decode("utf-8")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp949"


def filter_seoul(zip_path):
    """전국 CSV를 청크로 읽어 서울 행만 남긴다 → (서울 DataFrame, 전국 행 수, CSV 이름, 인코딩, 원본 열)"""
    zf = zipfile.ZipFile(zip_path, metadata_encoding="cp949")
    member = main_csv_member(zf)
    enc = detect_encoding(zf, member)
    parts, national = [], 0
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding=enc, errors="replace", newline="")
        reader = pd.read_csv(text, dtype=str, chunksize=CHUNK_ROWS, keep_default_na=False)
        for i, chunk in enumerate(reader):
            chunk.columns = [c.strip().lstrip("﻿") for c in chunk.columns]
            if i == 0:
                columns = list(chunk.columns)
            national += len(chunk)
            # object 문자열은 메모리를 많이 먹으므로(서울 270만 행 ≈ 수 GB) pyarrow 문자열로 바꿔 둔다
            parts.append(chunk[chunk["시도"].str.strip() == SEOUL].astype("string[pyarrow]"))
            print(f"    {national:>12,} 행 읽음, 서울 누적 {sum(len(p) for p in parts):,}", end="\r", flush=True)
    print()
    return pd.concat(parts, ignore_index=True), national, member.filename, enc, columns


def tidy(df):
    out = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    for c in out.columns:
        out[c] = out[c].str.strip()
    out["기준연도"] = out["기준연도"].astype(int)
    if "기준월" in out:
        out["기준월"] = out["기준월"].astype(int)
    out["전용면적"] = pd.to_numeric(out["전용면적"], errors="coerce")
    out["공시가격"] = pd.to_numeric(out["공시가격"], errors="coerce").astype("Int64")
    for c in ["단지코드", "동코드", "호코드"]:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    for c in ["본번", "부번"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    # 법정동코드(10) + 특수지코드 + 본번·부번 → 19자리 PNU (특수지코드 '0' 일반 → 대장구분 1, '1' 산 → 2)
    gb = out["특수지코드"].map({"0": "1", "1": "2"}).fillna("1")
    out["pnu"] = (out["법정동코드"] + gb + out["본번"].astype(str).str.zfill(4)
                  + out["부번"].astype(str).str.zfill(4))
    return out


# ---------------------------------------------------------------------------
# 아파트 판정 — 원본에 공동주택 유형 열이 없어서 단지코드 단위로 근거를 모은다.
#   1) molit    : 같은 필지(법정동코드+본번+부번)에 국토부 아파트 실거래(매매·전월세, 2021~) 기록이 있음
#   2) register : 같은 필지의 건축물대장 표제부(공동주택)의 기타용도에 '아파트'가 있음 (2026-10 snapshot)
#   3) floors   : 호명으로 본 최고층이 10층 이상인 동이 있음 (연립·다세대는 주택 층수 4개 층 이하) — 거래 없는 공공임대 등 보완
#   한 해라도 근거가 있는 단지코드는 모든 연도에서 아파트로 본다 (재건축 이주·철거 단지, 거래 없는 임대단지 보완).
# ---------------------------------------------------------------------------
TRADE_PATH = PROJECT_DIR / "data" / "v2_5year_seoul" / "processed" / "apt_trade_raw.parquet"
RENT_PATH = PROJECT_DIR / "data" / "v2_5year_seoul" / "processed" / "apt_rent_raw.parquet"
FLOOR_MIN_APT = 10


def parcel_key(dong10, bon, bu):
    return (dong10.astype(str) + "-" + pd.to_numeric(bon, errors="coerce").astype("Int64").astype(str)
            + "-" + pd.to_numeric(bu, errors="coerce").fillna(0).astype("Int64").astype(str))


def apartment_parcels():
    tr = pd.read_parquet(TRADE_PATH, columns=["법정동시군구코드", "법정동읍면동코드", "법정동", "법정동본번코드", "법정동부번코드"])
    tr["dong10"] = tr["법정동시군구코드"].astype(str) + tr["법정동읍면동코드"].astype(str).str.zfill(5)
    molit = set(parcel_key(tr["dong10"], tr["법정동본번코드"], tr["법정동부번코드"]))
    # 전월세는 읍면동코드가 없어 매매의 (시군구, 법정동명) → 코드 표로 붙인다
    umd = tr.drop_duplicates(["법정동시군구코드", "법정동"])
    umd = dict(zip(zip(umd["법정동시군구코드"].astype(str), umd["법정동"]), umd["dong10"]))
    re_ = pd.read_parquet(RENT_PATH, columns=["sggCd", "umdNm", "jibun"]).drop_duplicates()
    re_["dong10"] = [umd.get((s, u), "") for s, u in zip(re_["sggCd"].astype(str), re_["umdNm"])]
    j = re_["jibun"].fillna("").str.split("-", expand=True).reindex(columns=[0, 1])
    molit |= set(parcel_key(re_["dong10"], j[0], j[1]))

    br_path = max(EXTERNAL_DIR.glob("building_register_title_seoul_*.parquet"))
    br = pd.read_parquet(br_path, columns=["sigunguCd", "bjdongCd", "bun", "ji", "mainPurpsCd", "etcPurps"])
    br = br[(br["mainPurpsCd"] == "02000") & br["etcPurps"].fillna("").str.contains("아파트")]
    register = set(parcel_key(br["sigunguCd"] + br["bjdongCd"], br["bun"], br["ji"]))
    return molit, register, br_path.name


def complex_evidence(df, molit, register):
    """단지코드별 근거 (molit/register/floors) — 한 연도 서울 공동주택 전체에서"""
    key = parcel_key(df["법정동코드"], df["본번"], df["부번"])
    ev = pd.DataFrame({
        "단지코드": df["단지코드"], "molit": key.isin(molit), "register": key.isin(register),
    }).groupby("단지코드").any()
    # 층수: 호명 앞자리(예: 1203 → 12층). 빌라는 호명에 동 번호를 붙이는 경우가 많아(10동 1011호 = 1층)
    # 동 안에서 층 값이 고르게 퍼져 있을 때만 믿는다: 최고층 ≥ 10 이고 서로 다른 층 수 ≥ 최고층의 절반.
    floor = pd.to_numeric(df["호명"].str.extract(r"^\D*?(\d{3,4})$")[0], errors="coerce") // 100
    fl = pd.DataFrame({"단지코드": df["단지코드"], "동명": df["동명"], "floor": floor.where(floor <= 80)})
    by_dong = fl.dropna().groupby(["단지코드", "동명"])["floor"].agg(["max", "nunique"])
    tall = by_dong[(by_dong["max"] >= FLOOR_MIN_APT) & (by_dong["nunique"] >= by_dong["max"] / 2)]
    ev["floors"] = ev.index.isin(tall.index.get_level_values("단지코드"))
    return ev


def out_path(year):
    return OUT_DIR / f"apt_official_price_seoul_{year}.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2026], metavar=("FROM", "TO"))
    ap.add_argument("--raw-dir", type=Path, default=Path(tempfile.gettempdir()) / "gongsi_raw")
    ap.add_argument("--keep-raw", action="store_true", help="전국 원본 ZIP을 지우지 않는다")
    ap.add_argument("--force", action="store_true", help="저장된 연도도 다시 만든다")
    args = ap.parse_args()
    years = list(range(args.years[0], args.years[1] + 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    todo = [y for y in years if args.force or not out_path(y).exists()]
    if not todo:
        print("모든 연도가 이미 저장되어 있습니다:", years)
        return
    print("수집 대상 연도:", todo)
    versions = {v["year"]: v for v in discover_versions()}
    missing = [y for y in todo if y not in versions]
    if missing:
        print("data.go.kr에 없는 공시연도:", missing)

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str) if MANIFEST_PATH.exists() else pd.DataFrame()

    # 1) 연도별 전국 원본 → 서울 공동주택 전체 (임시 캐시: raw-dir/seoul_all_YYYY.parquet) + 단지별 아파트 근거
    molit, register, br_name = apartment_parcels()
    caches, meta, evidence = {}, {}, []
    for y in [y for y in todo if y in versions]:
        v = versions[y]
        cache = args.raw_dir / f"seoul_all_{y}.parquet"
        meta_path = args.raw_dir / f"seoul_all_{y}.json"
        if cache.exists() and meta_path.exists():
            df, meta[y] = pd.read_parquet(cache), json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            print(f"[{y}] 다운로드: {v['name']} ({v['source_url']})")
            zip_path = download(v, args.raw_dir)
            print(f"[{y}] 서울 추출: {zip_path.name}")
            df, national, member, enc, columns = filter_seoul(zip_path)
            years_in_file = sorted(df["기준연도"].str.strip().unique())
            if years_in_file != [str(y)]:
                raise RuntimeError(f"{v['name']}: 기준연도 {years_in_file} ≠ {y}")
            df.to_parquet(cache, index=False)
            meta[y] = {"national": national, "member": member, "encoding": enc, "columns": columns}
            meta_path.write_text(json.dumps(meta[y], ensure_ascii=False), encoding="utf-8")
            if not args.keep_raw:
                zip_path.unlink()
        caches[y] = cache
        evidence.append(complex_evidence(df, molit, register))
        del df

    # 2) 아파트 판정 — 이번에 처리하는 연도 + 이미 저장된 연도(전부 아파트)의 단지코드를 함께 쓴다
    ev = pd.concat(evidence).groupby(level=0).any()
    ev.index = ev.index.astype(str).str.strip()
    labels = pd.Series(None, index=ev.index, dtype=object)
    for name in ["floors", "register", "molit"]:  # 뒤에 쓴 근거가 우선
        labels[ev[name]] = name
    for y in years:
        if y not in caches and out_path(y).exists():
            saved = pd.read_parquet(out_path(y), columns=["단지코드"])["단지코드"].astype(str)
            labels[labels.index.isin(saved) & labels.isna()] = "saved_year"

    # 3) 저장
    for y, cache in caches.items():
        df = pd.read_parquet(cache)
        code = df["단지코드"].astype(str).str.strip()
        lab = code.map(labels)
        keep = lab.notna().to_numpy()
        apt = tidy(df[keep])
        apt["아파트판정"] = lab[keep].to_numpy()
        apt.insert(2, "공시일", pd.to_datetime(NOTICE_DATE.get(y)))
        apt = apt.drop(columns=[c for c in ["시도", "읍면"] if c in apt])
        apt.to_parquet(out_path(y), index=False)
        v = versions[y]
        row = {
            "year": y, "source_url": v["source_url"], "dataset_page": PAGE_URL,
            "dataset_version_name": v["name"], "registered_on_portal": v["registered"],
            "original_file_name": v["original_file_name"], "csv_member": meta[y]["member"],
            "encoding": meta[y]["encoding"], "rows_national": meta[y]["national"],
            "rows_seoul_all_housing": len(df), "rows_seoul_apt": len(apt),
            "complexes_seoul_apt": apt["단지코드"].nunique(),
            "apt_rule_counts": json.dumps(apt["아파트판정"].value_counts().to_dict(), ensure_ascii=False),
            "notice_date": NOTICE_DATE.get(y, ""), "apt_register_snapshot": br_name,
        }
        print(f"[{y}] 전국 {row['rows_national']:,} → 서울 공동주택 {len(df):,} → 서울 아파트 {len(apt):,} 호 "
              f"({row['complexes_seoul_apt']:,} 단지) {row['apt_rule_counts']}")
        if len(manifest):
            manifest = manifest[manifest["year"].astype(str) != str(y)]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        del df, apt
    manifest["year"] = manifest["year"].astype(int)
    manifest.sort_values("year").to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    print("manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
