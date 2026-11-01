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

보완 후 `r11_interval4.py`가 만든 `r11_oof2.parquet`(매달 재학습 OOF 예측)를 `models/regression/v2/oof_predictions.parquet`로 복사해 커밋했고 (보완 전 OOF는 커밋하지 않음), 노트북은 그 파일을 읽는다.

- 실행 위치는 이 폴더다 (`r11_route.py`, `r11_v1cmp.py`는 상대 경로로 다른 스크립트를 읽는다).
- 1~5단계는 06 패치 전 레지스트리(기본 Feature 196개)에서 돌았다. `r11_stage1.py`의 `assert`는 그 시점 레지스트리를 확인한다.
  6~10단계는 06에 반영된 레지스트리(194개, `예상가격_단지만` 포함), 11단계 보완부터는 직전 거래 Feature 5개를 더한 199개를 쓴다.
- `r11_v1cmp.csv`는 `r11_v1cmp.py` 실행 출력을 옮겨 적은 값이다.
