"""
서울 아파트 매매 실거래 워밍업 구간 수집 (국토교통부 아파트 매매 실거래가, PublicDataReader, DATA_GO_KR_API_KEY)

- 기간: 2020-01 ~ 2021-06 (학습 행 2021-07 이전)
- 용도: 06 Feature Merge에서 2개월 전 기준 구·서울 시장 피처(최대 12개월 창)를 2021-07 행부터 채우기 위한 과거 구간.
  학습·검증 행으로는 쓰지 않는다.
- 01_data_collection.ipynb와 같은 API·같은 체크포인트 형식(raw/trade/{시군구코드}_{YYYYMM}.csv)을 쓴다.
  끊기면 다시 실행해 이어 받는다.

실행: python notebook/v2_5year_notebook/eda/v2/06a_collect_trade_warmup.py
"""

import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from PublicDataReader import TransactionPrice


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
DATA_DIR = PROJECT_DIR / "data" / "v2_5year_seoul"
RAW_DIR = DATA_DIR / "raw" / "trade"
OUTPUT_PATH = DATA_DIR / "processed" / "apt_trade_warmup_raw.parquet"
MONTHS = [p.strftime("%Y%m") for p in pd.period_range("2020-01", "2021-06", freq="M")]
SLEEP = 0.3

SEOUL_GU = [
    "11110", "11140", "11170", "11200", "11215", "11230", "11260", "11290", "11305", "11320",
    "11350", "11380", "11410", "11440", "11470", "11500", "11530", "11545", "11560", "11590",
    "11620", "11650", "11680", "11710", "11740",
]

api = TransactionPrice(dotenv_values(PROJECT_DIR / ".env")["DATA_GO_KR_API_KEY"])


def fetch(sgg, ym, retries=4):
    for attempt in range(retries):
        try:
            return api.get_data(property_type="아파트", trade_type="매매", sigungu_code=sgg, year_month=ym)
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"  재시도 {attempt + 1}/{retries} ({sgg} {ym}): {str(e)[:120]} → {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"요청 실패: {sgg} {ym}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(s, m) for s in SEOUL_GU for m in MONTHS if not (RAW_DIR / f"{s}_{m}.csv").exists()]
    print(f"대상 {len(SEOUL_GU) * len(MONTHS)}개 중 남은 {len(todo)}개")
    for i, (sgg, ym) in enumerate(todo, 1):
        df = fetch(sgg, ym)
        if df is None or df.empty:
            raise RuntimeError(f"데이터 없음: {sgg} {ym} — 서울 구·월에 거래 0건은 비정상")
        df.to_csv(RAW_DIR / f"{sgg}_{ym}.csv", index=False, encoding="utf-8-sig")
        time.sleep(SLEEP)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}")

    frames = [pd.read_csv(RAW_DIR / f"{s}_{m}.csv", dtype={"법정동시군구코드": str})
              for s in SEOUL_GU for m in MONTHS]
    raw = pd.concat(frames, ignore_index=True)
    raw["거래연월"] = raw["계약년도"].astype(str) + raw["계약월"].astype(int).astype(str).str.zfill(2)
    assert raw["거래연월"].between(MONTHS[0], MONTHS[-1]).all(), "기간 밖 거래가 섞였습니다."
    for col in raw.select_dtypes(include="object").columns:
        raw[col] = raw[col].astype(str)
    raw.to_parquet(OUTPUT_PATH, index=False)
    print(f"저장: {OUTPUT_PATH} {raw.shape}")
    print(raw.groupby("거래연월").size().to_string())


if __name__ == "__main__":
    main()
