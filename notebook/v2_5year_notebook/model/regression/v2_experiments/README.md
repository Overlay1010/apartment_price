# 11 회귀 V2 실험 기록

`11_regression_v2.ipynb`가 읽는 실험 결과와, 그 결과를 만든 스크립트다.
탐색 · 재학습이 수 시간 걸려 노트북은 여기 저장된 결과를 불러와 해석하고, 오차 범위 · 등급 · OOF Gap · 최종 모델만 직접 계산한다.

| 순서 | 스크립트 | 결과 | 내용 |
|---|---|---|---|
| 1 | `r11_stage1.py` | `r11_stage1.csv`, `r11_naive.json` | 단순 규칙 기준선, Feature 묶음 사다리 (XGB lr 0.1) |
| 2 | `r11_check.py` | `r11_check_output.txt`, `r11_perm_single.csv` | Valid 신규 단지 분석, permutation 중요도 |
| 3 | `r11_ablation.py` | `r11_ablation.csv` | Ablation A0~A11 |
| 4 | `r11_abl2.py` | `r11_abl2.csv`, `r11_features.json` | 시드 흔들림, 예상가격 대체값 처리 A12 · A13 (채택) |
| 5 | `r11_tune.py` | `r11_optuna.db` | Optuna, 시간 fold 2개 (XGB 30 · LGB 30 · Cat 20회) |
| 6 | `r11_refit.py` | `r11_refit_log.jsonl`, `r11_best_params.json` | 최적 설정을 최종 학습률로 재학습 (fold · 홀드아웃 예측) |
| 7 | `r11_blend.py`, `r11_route.py` | `r11_blend.csv` | 앙상블 비교, 신규 단지 분기 |
| 8 | `r11_drift.py`, `r11_v1cmp.py` | `r11_drift.csv`, `r11_v1cmp.csv` | Test 과소추정 진단, 같은 조건 V1 비교 |
| 9 | `r11_rolling.py` | `r11_rolling.csv` (+ `r11_roll/*.npy`, 커밋 안 함) | 매달 재학습 평가 2022-01~2026-06, 최근 확정월 보정 |
| 10 | `r11_interval2.py`, `r11_interval3.py` | `r11_interval_grid.csv` | 오차 범위 분위수 · 창 길이 선택 (Valid), 신규 단지 별도 구간 |
| 11 | `lt_feature.py`, `lt_test.py` | `lt_test_output.txt` | **보완:** 같은 단지 · 면적 직전 거래 Feature 프로토타입, Train → Valid 비교 (6.16 → 5.24%) → 06에 반영 |
| 12 | `r11_tune2.py` | `r11_optuna.db` (`r11_lgb_lasttrade`) | 보완 후 LightGBM 재탐색 30회 |
| 13 | `r11_rolling2.py` | `r11_best_params_lasttrade.json` (+ `r11_roll2/*.npy`, 커밋 안 함) | 보완 후 매달 재학습 평가 2022-01~2026-06 (반복 4,290회) |
| 14 | `r11_interval4.py`, `r11_compare_lasttrade.py` | `r11_interval_grid2.csv`, `r11_lasttrade_compare.csv` | 보완 후 오차 범위, 직전 거래가 기준선 비교 |
| 15 | `r11_sg_build.py`, `r11_sg_test.py`, `r11_sg_test2.py` | `sg_feat.parquet`, `r11_sg_test*.csv` | **보완 2 후보:** 단지 주변 상권 규모 7개 (소상공인 상가정보, 02c와 같은 스냅샷 시점 규칙) |
| 16 | `r11_hakgu_build.py`, `r11_hakgu_test.py` | `hakgu_feat.parquet`, `r11_hakgu_test.csv` | 후보: 학구도 배정 초등학교 8개 (`data/external/schoolzone`, 거래의 23%는 배정 학교 ≠ 최근접 학교) |
| 17 | `r11_kapt_build.py`, `r11_kapt_test.py` | `kapt_feat.parquet`, `r11_kapt_test.csv` | 후보: K-apt 단지 속성 7개 (`05c_collect_kapt.py`, 지번 → 도로명 순으로 연결, 거래 85%) |
| 18 | `r11_noise_check.py` | `r11_noise_check.csv` | 15~17 점검이 학습률 0.06 + 평가 구간 조기 종료(트리 360~400개)라 시드만 바꿔도 f1 0.4%p · 홀드아웃 0.9%p가 흔들림, 난수 열 7개도 같은 "개선" → 15~17 판정 무효 |
| 19 | `r11_idx_build.py`, `r11_eval_fixed.py` | `idx_feat.parquet`, `r11_eval_fixed.csv` | 본 모델 설정(학습률 0.03, 트리 3,734개 고정, 조기 종료 없음) × 시드 2개로 재비교. 상권 · 배정초 · K-apt 불채택, **직전 거래가 시세 보정 2개 채택** (f1 5.60 → 5.31, f2 4.99 → 4.84, 홀드아웃 5.13 → 4.96%) → 06에 반영 |
| 20 | `r11_rolling3.py`, `r11_compare_idx.py` | `r11_best_params_idx.json`, `r11_idx_compare.csv`, `r11_idx_by_half.csv` (+ `r11_roll3/*.npy`, 커밋 안 함) | 보완 2 매달 재학습 평가 2022-01~2026-06 (반복 2,080회), v2.2 · 단순 기준선 비교 → 중개 Valid 4.95 → 4.86, Test 5.53 → 5.35% |
| 21 | `r11_noise_floor.py` | `r11_noise_floor.csv`, `r11_noise_floor_adj.csv`, `r11_noise_floor_output.txt` | 비교 기준: 같은 단지 · 면적 · 같은 달 다른 거래의 실제 가격 평균으로 맞히면 Test 중개 4.47%, 모델 4.86% (같은 달 비교 거래 있는 63%). 비교 거래 없는 기존 단지 6.03% · 신규 단지 13.61% |
| 22 | `r11_nb_build.py`, `r11_nb_test.py` | `nb_feat.parquet`, `nb_neighbors.parquet` (커밋 안 함), `r11_nb_test.csv`, `r11_nb_test_output.txt` | 보완 3 후보: 단지 모든 면적 시세 보정가 3개 · 같은 구 유사 단지(좌표 · 준공 · 세대수) 시세 보정가 2개, 본 모델 설정 × 시드 2개 → f1 악화 또는 0.1%p 미달로 **불채택** |
| 23 | `r11_floor_test.py` | `r11_floor_test.csv`, `r11_floor_test_output.txt` | 층 관계 Feature(층 차이 · 상대 층 · 2층 이하), 기준 결과는 22단계 재사용 → f1 악화로 **불채택** |
| 24 | `r11_unit_build.py`, `r11_unit_test.py` | `unit_feat.parquet` (커밋 안 함), `r11_unit_test.csv`, `r11_unit_test_output.txt` | 보완 4 후보: 호별 공시가격 층별 값 · 층 프리미엄 · 직전 거래 층 대비 비율 · 층 보정 직전 거래가, 기준 결과는 22단계 재사용 → 4개 구성 개선 0.03~0.06%p로 기준 미달, **불채택** |
| 24-1 | `r11_error_share.py` | `r11_error_share.csv` | 오차 구간별 비중 (매달 재학습 OOF, 중개, 2023~25): 같은 면적 직전 거래 6개월 내 거래가 81%, 오차의 72%. 처음 보는 단지는 오차의 3.7% |
| 25 | `r11_method_test.py` | `r11_method_test.csv`, `r11_method_test_output.txt` (첫 실행 `_run1`: 기준 11회 뒤 요약 출력 오류로 중단, 결과는 CSV에 이어 씀) | 보완 5: 학습 방법 비교(잔차 학습 · L1 손실 · 최근 거래 가중), 판정 보강(시간 fold 4개 × 시드 3개, 결과 전에 고정) → 모두 **불채택**. 잔차 학습은 평균 +0.117%p지만 f0a가 시드 3개 모두 악화. 기준 모델 f1 시드 간 차이가 0.27%p로 드러남 |

보완 후 `r11_interval4.py`가 만든 `r11_oof2.parquet`(매달 재학습 OOF 예측)를 `models/regression/v2/oof_predictions.parquet`로 복사해 커밋했고 (보완 전 OOF는 커밋하지 않음), 노트북은 그 파일을 읽는다.
보완 2부터는 `r11_rolling3.py`가 만든 `r11_oof3.parquet`를 같은 위치로 복사했다.
비교용 v2.2 OOF(`r11_oof2.parquet`, 커밋 안 함)는 `git show v2.2-regression:models/regression/v2/oof_predictions.parquet > r11_oof2.parquet`로 복원한다.
15~19 · 22 · 24단계 후보 Feature parquet(`sg_feat` · `hakgu_feat` · `kapt_feat` · `idx_feat` · `nb_feat` · `unit_feat`)도 커밋하지 않고 `r11_*_build.py`로 다시 만든다.

- **Feature 추가 실험은 본 모델 설정 그대로 돌린다** (`r11_eval_fixed.py`, 결과 CSV에 이어 쓰기). 평가 구간으로 조기 종료하지 않고, 시드 간 차이가 채택 기준(0.1%p)보다 충분히 작은지 먼저 본다.

- 실행 위치는 이 폴더다 (`r11_route.py`, `r11_v1cmp.py`는 상대 경로로 다른 스크립트를 읽는다).
- 1~5단계는 06 패치 전 레지스트리(기본 Feature 196개)에서 돌았다. `r11_stage1.py`의 `assert`는 그 시점 레지스트리를 확인한다.
  6~10단계는 06에 반영된 레지스트리(194개, `예상가격_단지만` 포함), 11~19단계는 직전 거래 Feature 5개를 더한 199개, 20단계부터는 시세 보정 2개를 더한 201개를 쓴다.
- `r11_v1cmp.csv`는 `r11_v1cmp.py` 실행 출력을 옮겨 적은 값이다.
