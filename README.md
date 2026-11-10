# 서울 아파트 적정가 분석

서울 아파트 실거래 5년치(2021-07~2026-06, 약 23만 건)로 **이 집의 적정가는 얼마이고, 지금 거래가가 싼지 비싼지**를 보여주는 B2C 웹 서비스를 목표로 하는 엔드투엔드 ML 프로젝트입니다.

현재는 데이터 수집 · Feature · 모델링(V2)까지 완료했습니다. 다음 단계(V3)는 집 단위 EDA와 모델 재설계이고, 웹 서비스는 V4입니다.
설계 · 결정 근거 · 로드맵 전체는 [portfolio-architecture.md](portfolio-architecture.md)에 있습니다.

## V2 결과 요약

공통 Feature 레이어 위에 단위가 다른 세 모델을 두고, 검증된 연결만 씁니다.

| 모델 | 단위 | 결과 | 서비스에서 보여줄 값 |
|---|---|---|---|
| **11 회귀** | 거래 | LightGBM, Feature 201개, 매달 재학습. **Test(2026 상반기) 중개거래 MAPE 5.35%, 오차 중앙값 3.95%, 10% 이내 87.4%** | 적정가, 80% 범위, 적정가 대비 등급 |
| **21 군집** | 단지 | 주택 특성은 군집이 뚜렷하지 않아(Silhouette 0.09) 웹 태그는 법령 · 조례 · 관행 기준 **규칙 태그 13종**으로 대체. 입지 · 교육 군집은 분석용으로 유지 | 단지 태그, 유사 단지 |
| **31 분류** | 구 × 월 | 3개월 뒤 상승 구 예측은 어떤 모델도 **최근 4주 흐름 기준선을 넘지 못함** (Test 상위 8개 구 적중 52%, 무작위 33%) → 확률은 보여주지 않음 | 4주 흐름 순위, 기회 신호, 정책 타임라인, 권역 수급 |

**11 회귀 세부 (Test 2026-01~06, 중개거래 37,923건)**

| 구분 | MAPE | 비고 |
|---|---:|---|
| 모델 전체 | 5.35% | 오차 20% 초과 거래 2.0% |
| 기존 단지 | 5.27% | 80% 범위 실제 포함률 81.8% |
| 신규 단지 (학습에 없던 단지) | 13.61% | 332건, 등급 표시 안 함 |
| 비교: 같은 단지 · 면적 직전 실거래가 그대로 | 6.67% | 사용자가 이미 보는 값 |

- 모든 Feature는 **그 거래 시점에 알 수 있던 값만** 씁니다 (공시일 · 신고 시차 · 주간 지수 공표일 기준).
- 후보 Feature와 학습 방법은 **결과를 보기 전에 정한 채택 규칙**으로 판정합니다. 보완 3~5에서 6개 후보를 시험했고 모두 불채택이라 모델을 바꾸지 않았습니다 ([실험 기록](notebook/v2_5year_notebook/model/regression/v2_experiments/README.md)).
- 31에서 확인한 **기회 신호**: 적정가보다 5% 넘게 싸게 거래된 집은 다음 거래에서 시장보다 평균 5.8~6.0%p 더 올랐습니다.

## 폴더 구조

```text
notebook/v2_5year_notebook/
├─ 01_data_collection.ipynb       실거래 수집
├─ eda/v2/                        V2 Feature (01 시장 → 02 학교·학원 → 03 정책 → 04 공급 → 05 인구·단지 속성 → 06 병합)
│   └─ 0Nx_*.py                   외부 데이터 수집 스크립트
├─ model/clustering/21_*.ipynb    군집 · 규칙 태그 · 유사 단지
├─ model/regression/11_*.ipynb    적정가 회귀 (+ v2_experiments/ 실험 스크립트 · 결과)
├─ model/classification/31_*.ipynb 지역 흐름 (+ v2_experiments/)
└─ eda/v1/ · model/*/[123]0_*     V1 기록
data/
├─ v2_5year_seoul/raw · processed  실거래 원본, Feature 테이블 (v2_trade · v2_complex · v2_region_month)
└─ external/                       외부 데이터 (대용량 원본은 .gitignore, 추출 결과만 커밋)
models/{clustering,regression,classification}/v1 · v2   모델 산출물
```

## 실행 방법

```bash
python -m venv .venv && .venv\Scripts\activate    # Python 3.13
pip install -r requirements.txt
```

프로젝트 루트에 `.env`를 만들고 아래 키를 넣습니다 (수집 단계에서만 필요하고, 커밋된 `processed` 데이터로 모델 노트북은 바로 실행됩니다).

| 키 | 발급처 | 쓰임 |
|---|---|---|
| `DATA_GO_KR_API_KEY` | 공공데이터포털 | 국토부 실거래가 · 건축물대장 등 |
| `MOLIT_RENT_API_KEY` | 공공데이터포털 | 전월세 실거래 |
| `KAKAO_REST_API_KEY` | Kakao Developers | 주소 → 좌표 |
| `KSKILL_KOSIS_API_KEY` | KOSIS | 부동산원 주간 지수 · 인구 통계 |
| `KSKILL_BOK_ECOS_API_KEY` | 한국은행 ECOS | 기준금리 등 |
| `ODSAY_API_KEY` | ODsay LAB | 대중교통 통근 시간 |

실행 순서는 `01_data_collection` → `eda/v2` 01~06 → `21` → `11` → `31`입니다.
11 · 31의 탐색 · 매달 재학습은 수 시간이 걸려 `v2_experiments/` 스크립트로 돌렸고, 노트북은 저장된 결과를 불러와 해석합니다.

## 데이터 원칙

- 공식 공개 데이터만 씁니다: 국토부 실거래가 · 건축물대장 · 공동주택 공시가격, 한국부동산원 주간 지수(KOSIS), 한국은행 ECOS, 학교 · 학원 · 상가 공공데이터, 청약홈 등.
- 민간 매물 사이트(호가)는 크롤링하지 않습니다. 호가는 서비스에서 사용자 입력으로 받습니다.

## 버전

| 태그 | 내용 |
|---|---|
| `v1.0` | V1: 5년치 데이터 군집 · 회귀 · 분류 |
| `v2.0-features` · `v2.1-clustering` · `v2.2-regression` | V2 Feature, 군집, 회귀 |
| `v2.3-classification` | **V2 Release**: 회귀 보완 2 + 31 분류 ([Release](https://github.com/Overlay1010/apartment_price/releases/tag/v2.3-classification)) |
| `v2.4-regression-review` | V2 마무리: 회귀 보완 3~5 점검 (모델 변경 없음) |

다음 계획:

| 버전 | 내용 |
|---|---|
| V3 | 집 단위 EDA(동 · 호별 공시가격, 큰 오차 거래, 등기 여부) · 신규 단지(분양권 · 분양가) · 단지 단위 가격 변화 → 회귀 · 분류 재설계 |
| V4 | 서비스: `apt-core` 모듈화 · 데이터 갱신 파이프라인 · 웹 MVP · AI 레이어 |
| V5 | 오피스텔 · 빌라 등 주택 유형 확장 |

## 라이선스

[GPL-3.0](LICENSE)
