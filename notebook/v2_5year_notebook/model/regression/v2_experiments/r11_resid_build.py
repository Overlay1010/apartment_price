# V3 추가 시험 후보: 단지 최근 모델 오차(OOF 잔차) Feature — 실험용 거래 단위 파일 생성
# - 근거: V3 EDA 3(eda/v3/03_complex_change.ipynb)에서 적정가보다 비싸게 거래되던 단지(OOF 잔차 +)가 다음 해에도 구 평균보다 더 오름 (2023→24 · 2024→25 일관)
#   → 11 적정가가 오르는 단지를 늦게 따라간다면, 그 단지의 최근 잔차가 다음 거래 예측을 바로잡을 수 있다
# - 잔차: 매달 재학습 OOF(models/regression/v2/oof_predictions.parquet, 2022-01~) r = log(실거래 / 예측). 거래 k의 잔차는 그 거래가 신고된 뒤(계약일 + 30일) 알 수 있다
# - 후보(결과를 보기 전에 고정, 2026-11-12):
#   단지_최근잔차      = 같은 단지 · 직거래 아닌 거래 중 계약일이 [d − 395일, d − 30일] 안인 거래의 OOF 잔차 평균 (직전 거래 Feature와 같은 사용가능일 규칙)
#   단지_최근잔차_건수 = 위 평균에 쓴 거래 수
#   2022-02 이전 거래 · 기간 안에 단지 거래가 없는 거래는 결측 (LightGBM이 그대로 처리)
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), ''); ROOT = os.path.join(S, *['..'] * 5, '')
LAG, WIN = 30, 395
t = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '단지일련번호', '계약일자', '거래유형'])
o = pd.read_parquet(ROOT + 'models/regression/v2/oof_predictions.parquet', columns=['transaction_id', 'r', '거래유형'])
t['단지일련번호'] = t['단지일련번호'].astype(str); t['계약일자'] = pd.to_datetime(t['계약일자'])
ref = o[o['거래유형'].ne('직거래')].merge(t[['transaction_id', '단지일련번호', '계약일자']], on='transaction_id', validate='one_to_one')
print(f"잔차 참조 거래 {len(ref):,}건, 계약일 {ref['계약일자'].min().date()} ~ {ref['계약일자'].max().date()}")

day = lambda s: (s - pd.Timestamp('2000-01-01')).dt.days.to_numpy()
ref = ref.sort_values(['단지일련번호', '계약일자'])
mean_out = np.full(len(t), np.nan); cnt_out = np.zeros(len(t))
q = t.assign(i=np.arange(len(t)), d=day(t['계약일자']))
groups = {k: (day(g['계약일자']), g['r'].to_numpy()) for k, g in ref.groupby('단지일련번호', sort=False)}
for cx, g in q.groupby('단지일련번호', sort=False):
    if cx not in groups: continue
    rd, rr = groups[cx]; cs = np.concatenate([[0.0], np.cumsum(rr)])
    hi = np.searchsorted(rd, g['d'].to_numpy() - LAG, side='right')   # 계약일 ≤ d − 30
    lo = np.searchsorted(rd, g['d'].to_numpy() - WIN, side='left')    # 계약일 ≥ d − 395
    n = hi - lo; idx = g['i'].to_numpy()
    cnt_out[idx] = n
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_out[idx] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
out = pd.DataFrame({'transaction_id': t['transaction_id'], '단지_최근잔차': mean_out, '단지_최근잔차_건수': cnt_out})
out.loc[out['단지_최근잔차'].isna(), '단지_최근잔차_건수'] = np.nan

# ---------- 재계산 검증: 표본 300건을 원본에서 직접 다시 계산, 쓰는 거래는 모두 계약일 30일 전까지 ----------
rng = np.random.default_rng(0); refi = ref.set_index('단지일련번호')
for i in rng.choice(np.flatnonzero(~np.isnan(mean_out)), 300, replace=False):
    r_ = t.iloc[i]; g = refi.loc[[r_['단지일련번호']]]; d0 = pd.Timestamp(r_['계약일자'])
    use = g[(g['계약일자'] <= d0 - pd.Timedelta(days=LAG)) & (g['계약일자'] >= d0 - pd.Timedelta(days=WIN))]
    assert len(use) == cnt_out[i] and np.isclose(use['r'].mean(), mean_out[i])
    assert r_['transaction_id'] not in set(use['transaction_id'])
print('✅ 표본 300건 재계산 일치 · 자기 거래 미포함 · 계약일 30일 전까지만 사용')
yr = t['계약일자'].dt.year
print('연도별 값 있음:', pd.Series(~np.isnan(mean_out)).groupby(yr.to_numpy()).mean().round(3).to_dict())
print('분위 1/10/50/90/99%:', np.round(np.nanquantile(mean_out, [.01, .1, .5, .9, .99]), 4), '| 건수 중앙', np.nanmedian(out['단지_최근잔차_건수']))
out.to_parquet(S + 'resid_feat.parquet', index=False); print('저장 resid_feat.parquet', out.shape)
