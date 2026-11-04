# 보완 2 전 · 후 매달 재학습 OOF와 단순 기준선 비교 (중개거래, 같은 정보 시점: 계약일 30일 전까지 거래 · 계약일 전 공표 지수)
# - 보완 2 후: r11_oof3.parquet (r11_rolling3), 보완 2 전(v2.2): r11_oof2.parquet
#   (models/regression/v2는 보완 2 OOF로 바뀌었으므로 git show v2.2-regression:models/regression/v2/oof_predictions.parquet 로 복원)
# - 단순 기준선에 '시세 보정 직전 거래가'를 더한다: 모델 없이 이 값만 써도 되는지 확인
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
new = pd.read_parquet(S + 'r11_oof3.parquet')
old = pd.read_parquet(S + 'r11_oof2.parquet')[['transaction_id', 'p']].rename(columns={'p': 'p_old'})
c = new.merge(old, on='transaction_id', validate='one_to_one').merge(
    t[['transaction_id', '직전거래가_동일면적', '직전3건중위_동일면적', '직전거래가_시세보정']], on='transaction_id')
c['구간'] = np.select([c['월'].astype(str).between('2025-07', '2025-12'), c['월'].astype(str) >= '2026-01'], ['Valid', 'Test'], '')
c['연'] = c['월'].astype(str).str[:4]
mk_all = c['거래유형'].eq('중개거래')
a = np.exp(c['y']); has = c['직전거래가_동일면적'].notna(); rows = []
METHODS = [('보완 2 모델 (시세 보정)', np.exp(c['p'])), ('v2.2 모델', np.exp(c['p_old'])),
           ('직전 거래가 × 시세 보정', c['직전거래가_시세보정']), ('같은 단지·면적 직전 거래가', c['직전거래가_동일면적']),
           ('같은 단지·면적 직전 3건 중위', c['직전3건중위_동일면적'])]
for nm, pr in METHODS:
    e = (pr - a).abs() / a * 100
    for k in ['Valid', 'Test']:
        mk = c['구간'].eq(k) & mk_all
        rows.append({'방법': nm, '구간': k, '직전 거래 있는 거래 MAPE': e[mk & has].mean(), '직전 거래 있는 거래 중앙 APE': e[mk & has].median(),
                     '직전 거래 있는 거래 10%이내': (e[mk & has] <= 10).mean() * 100,
                     '전체 MAPE': e[mk].mean() if nm.endswith(('모델', '(시세 보정)')) else np.nan,
                     '직전 거래 없는 거래 MAPE': e[mk & ~has].mean() if nm.endswith(('모델', '(시세 보정)')) else np.nan,
                     '직전 거래 있음 비율': has[mk].mean() * 100})
r = pd.DataFrame(rows).round(3); r.to_csv(S + 'r11_idx_compare.csv', index=False, encoding='utf-8-sig'); print(r.to_string(index=False))
# 연도별 · 편향 (중개거래): 급락 · 급등 구간에서 얼마나 나아졌나
c['ape_new'] = np.abs(1 - np.exp(c['p'] - c['y'])) * 100; c['ape_old'] = np.abs(1 - np.exp(c['p_old'] - c['y'])) * 100
c['bias_new'] = (np.exp(c['p'] - c['y']) - 1) * 100; c['bias_old'] = (np.exp(c['p_old'] - c['y']) - 1) * 100
by = c[mk_all].groupby('연')[['ape_old', 'ape_new', 'bias_old', 'bias_new']].mean().round(2)
by.columns = ['v2.2 MAPE', '보완 2 MAPE', 'v2.2 편향', '보완 2 편향']; print(by.to_string())
half = c[mk_all].assign(반기=c['월'].astype(str).str[:4] + np.where(c['월'].astype(str).str[5:7].astype(int) <= 6, 'H1', 'H2'))
hb = half.groupby('반기')[['ape_old', 'ape_new']].mean().round(2); hb.columns = ['v2.2 MAPE', '보완 2 MAPE']
hb.to_csv(S + 'r11_idx_by_half.csv', encoding='utf-8-sig'); print(hb.to_string())
