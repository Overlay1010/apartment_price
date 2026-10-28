"""
지하철역 → 주요 업무지구 대중교통 소요시간 수집 (ODsay searchPubTransPathT, ODSAY_API_KEY)

- 출발: 서울 거래 단지 2km 이내 지하철역 (역 이름 기준, 환승역은 한 번만)
- 도착: 광화문역(CBD), 강남역(GBD), 여의도역(YBD)
- 경로마다 총 소요시간·도보시간·환승 횟수·이용 노선을 모두 저장한다.
  → 개통 전 기간에는 새 노선(신림선 등)을 쓰지 않는 경로만 골라 쓸 수 있다.
- 무료 한도(일 1,000건)에 걸리면 멈추고, 다시 실행하면 받은 쌍은 건너뛴다.
- ODsay 서버 키는 등록된 IP에서만 동작한다 (ApiKeyAuthFailed → lab.odsay.com에서 서버 IP 갱신).

실행: python notebook/v2_5year_notebook/eda/v2/05b_collect_commute.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import dotenv_values


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
OUT_DIR = PROJECT_DIR / "data" / "external" / "commute"
STATION_PATH = OUT_DIR / "origin_stations.csv"
PATHS_PATH = OUT_DIR / "odsay_station_paths.jsonl"

API_KEY = dotenv_values(PROJECT_DIR / ".env")["ODSAY_API_KEY"]
URL = "https://api.odsay.com/v1/api/searchPubTransPathT"
MIN_INTERVAL = 0.3
RADIUS_M = 2000

DESTINATIONS = {"CBD": ("광화문", "5호선"), "GBD": ("강남", "2호선"), "YBD": ("여의도", "5호선")}


class QuotaError(RuntimeError):
    pass


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def build_stations():
    s = pd.read_excel(PROJECT_DIR / "data" / "external" / "subway_stations.xlsx")
    s["역"] = s["역사명"].str.replace(r"\(.*\)", "", regex=True).str.strip()
    stations = s.groupby("역").agg(
        lat=("역위도", "mean"), lon=("역경도", "mean"),
        노선=("노선명", lambda x: "|".join(sorted(set(x)))),
    ).reset_index()

    complexes = pd.read_parquet(
        PROJECT_DIR / "data" / "v2_5year_seoul" / "processed" / "df_fe_base.parquet",
        columns=["단지일련번호", "위도", "경도"],
    ).drop_duplicates("단지일련번호")
    d = haversine(complexes["위도"].to_numpy()[:, None], complexes["경도"].to_numpy()[:, None],
                  stations["lat"].to_numpy()[None, :], stations["lon"].to_numpy()[None, :])
    stations = stations[(d <= RADIUS_M).any(axis=0)].reset_index(drop=True)

    dest = {}
    for code, (name, line) in DESTINATIONS.items():
        row = s[s["역"].eq(name) & s["노선명"].eq(line)].iloc[0]
        dest[code] = (row["역위도"], row["역경도"])
    return stations, dest


_last = [0.0]


def route(sx, sy, ex, ey):
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    r = requests.get(URL, params={"apiKey": API_KEY, "SX": sx, "SY": sy, "EX": ex, "EY": ey}, timeout=30)
    data = r.json()
    if "error" in data:
        err = data["error"]
        msg = json.dumps(err, ensure_ascii=False)
        if any(k in msg for k in ("ApiKeyAuthFailed", "LIMIT", "limit", "초과", "Quota", "quota")):
            raise QuotaError(msg)
        return {"error": err}
    paths = []
    for p in data["result"]["path"]:
        lanes = []
        for sp in p.get("subPath", []):
            for lane in sp.get("lane", []) or []:
                lanes.append(lane.get("name") or lane.get("busNo") or "")
        info = p["info"]
        paths.append({
            "pathType": p.get("pathType"),
            "totalTime": info.get("totalTime"),
            "totalWalk": info.get("totalWalk"),
            "busTransitCount": info.get("busTransitCount"),
            "subwayTransitCount": info.get("subwayTransitCount"),
            "payment": info.get("payment"),
            "lanes": lanes,
        })
    return {"paths": paths}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations, dest = build_stations()
    stations.to_csv(STATION_PATH, index=False, encoding="utf-8-sig")
    print(f"출발역 {len(stations)}곳 × 목적지 {len(dest)}곳 = {len(stations) * len(dest)}회")

    done = set()
    if PATHS_PATH.exists():
        for line in PATHS_PATH.open(encoding="utf-8"):
            rec = json.loads(line)
            done.add((rec["역"], rec["dest"]))

    n_new = 0
    with PATHS_PATH.open("a", encoding="utf-8") as f:
        for st in stations.itertuples():
            for code, (lat, lon) in dest.items():
                if (st.역, code) in done:
                    continue
                if haversine(st.lat, st.lon, lat, lon) < 300:  # 목적지 역 자체
                    result = {"paths": [{"totalTime": 0, "totalWalk": 0, "lanes": []}]}
                else:
                    try:
                        result = route(st.lon, st.lat, lon, lat)
                    except QuotaError as e:
                        print(f"중단: {e} — 받은 {n_new}건 저장됨. 내일 다시 실행하면 이어서 받습니다.")
                        return
                f.write(json.dumps({"역": st.역, "dest": code, "collected": time.strftime("%Y-%m-%d"),
                                    **result}, ensure_ascii=False) + "\n")
                f.flush()
                n_new += 1
    print(f"완료: 이번 실행 {n_new}건, 전체 {len(done) + n_new}건")


if __name__ == "__main__":
    main()
