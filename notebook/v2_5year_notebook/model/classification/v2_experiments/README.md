# 31 분류 V2 실험 기록

`31_classification_v2.ipynb`가 읽는 실험 결과와, 그 결과를 만든 스크립트다.
조합 수십 개 × 매달 재학습(walk-forward)이라 노트북은 여기 저장된 결과를 불러와 해석하고, 표시용 산출물(주간 흐름 순위 · 기준선 적중률 · 기회 신호 근거)만 직접 계산한다.

| 순서 | 스크립트 | 결과 | 내용 |
|---|---|---|---|
| 0 | `r31_base.py` | — | 공통: 구 × 예측시점 데이터, Feature 묶음, 회귀 OOF Gap(거래월 + 2부터), V1 Feature(`v1_cls_df.parquet`), 정답 공개 시점, walk-forward, 지표 |
| 1 | `r31_target_diag.py` | `r31_target_diag_output.txt`, `r31_target_diag.parquet`, `rs_index_gu_month.parquet` | V1 정답(구 중위 평당가) vs 같은 단지 · 면적 반복거래 지수: 상관 0.14 → 정답이 잡음 |
| 2 | `r31_official_diag.py` | `r31_official_diag_output.txt`, `r31_official_diag.parquet` | 부동산원 공식 주간 지수 정답: 반복거래와 상관 0.57, 자기상관 0.96 |
| 3 | `r31_forward_diag.py` | `r31_forward_diag_output.txt`, `r31_forward_diag.parquet` | 정답을 순수 미래 3개월 변화로 (06 반영 후에는 두 정의가 같음, 상관 1.0) |
| 4 | `r31_stage1.py` (`R31_TARGET=상승여부`) | `r31_stage1.csv`, `r31_stage1.log` | V1 재현(AUC 0.613) · V1 정답에서 Feature 묶음 × 모델 × 학습 창 |
| 5 | `r31_stage2.py` (`R31_TARGET=상승세` / `강한상승`) | `r31_stage2_{정답}.csv`, `.log` | 공식 정답 +1% · +3%: Feature 사다리 × 모델 × 창 (DEV) |
| 6 | `r31_stage3.py` (같은 환경 변수) | `r31_stage3_{정답}.csv`, `.log` | 최근 흐름 + 작은 Feature 조합 (DEV · Test) |
| 7 | `r31_stage4.py` | `r31_stage4_상대상위.csv`, `r31_stage4_output.txt` | 상대 정답: 같은 달 상위 8개 구 — Test 기준선 4주 흐름 0.749 > 모델 |
| 8 | `r31_cheap_rising.py` | `r31_cheap_rising.csv`, `r31_cheap_rising_output.txt` | 31 × 11: 적정가보다 싼 거래의 다음 거래 초과상승 (기회 신호 근거) |
| 9 | `r31_gu_effect.py` | `r31_gu_effect.csv` | 구 이름 원-핫: DEV 0.93 → Test 0.30 (2026 국면 전환) |
| 10 | `r31_weekly_extra.py` | `r31_weekly_extra.csv` | 주간 전세 지수 · 권역 수급 추가: 기준선 미달 |
| 11 | `r31_long_quarterly.py` | `r31_long_quarterly.csv`, `r31_long_quarterly_output.txt`, `kab_gu_quarterly.parquet` | 20년치 구별 분기 실거래가격지수(KOSIS 408/DT_KAB_11672_S5): 모델 < 최근 흐름, 갭 메우기 2025~26 AUC 0.39 |

- 실행 위치는 이 폴더다. 데이터는 `data/v2_5year_seoul/processed/v2_region_month.parquet`(06)과 `data/external/rone_weekly/`를 읽는다.
- `r31_base.py` · `r31_cheap_rising.py`는 `models/regression/v2/`의 OOF를 읽는다. 11 보완 2(2026-11-04)로 OOF가 바뀐 뒤 1 · 4 · 5 · 7 · 8단계를 다시 돌렸다
  (OOF Gap이 들어간 행만 AUC가 최대 0.03 변했고 결론은 같다).
- `r31_long_quarterly.py`는 캐시(`kab_gu_quarterly.parquet`)가 없을 때만 KOSIS를 호출한다 (`.env`의 `KSKILL_KOSIS_API_KEY`).
- DEV 2023-11~2025-12로 고르고 Test 2026-01~06은 확인만 한다. 지표: ROC-AUC, 월내 AUC(같은 달 안에서 구 순위), 상위 8 적중률(무작위 33%), Brier.
