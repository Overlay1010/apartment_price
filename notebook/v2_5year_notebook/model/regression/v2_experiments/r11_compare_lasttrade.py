# 보완 전 · 후 매달 재학습 OOF와 같은 단지 · 면적 직전 거래가 비교 (중개거래, 같은 정보 시점: 계약일 30일 전까지)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
new = pd.read_parquet(S + 'r11_oof2.parquet'); old = pd.read_parquet(S + 'r11_oof.parquet')[['transaction_id', 'p']].rename(columns={'p': 'p_old'})
c = new.merge(old, on='transaction_id').merge(t[['transaction_id', '직전거래가_동일면적', '직전3건중위_동일면적']], on='transaction_id')
c['구간'] = np.select([c['월'].astype(str).between('2025-07', '2025-12'), c['월'].astype(str) >= '2026-01'], ['Valid', 'Test'], '')
c = c[c['구간'].ne('') & c['거래유형'].eq('중개거래')]
a = np.exp(c['y']); has = c['직전거래가_동일면적'].notna(); rows = []
for nm, pr in [('보완 후 모델', np.exp(c['p'])), ('보완 전 모델', np.exp(c['p_old'])),
               ('같은 단지·면적 직전 거래가', c['직전거래가_동일면적']), ('같은 단지·면적 직전 3건 중위', c['직전3건중위_동일면적'])]:
    e = (pr - a).abs() / a * 100
    for k in ['Valid', 'Test']:
        mk = c['구간'].eq(k)
        rows.append({'방법': nm, '구간': k, '직전 거래 있는 거래 MAPE': e[mk & has].mean(), '직전 거래 있는 거래 중앙 APE': e[mk & has].median(),
                     '직전 거래 있는 거래 10%이내': (e[mk & has] <= 10).mean() * 100,
                     '전체 MAPE': e[mk].mean() if nm.endswith('모델') else np.nan, '직전 거래 없는 거래 MAPE': e[mk & ~has].mean() if nm.endswith('모델') else np.nan,
                     '직전 거래 있음 비율': has[mk].mean() * 100})
r = pd.DataFrame(rows).round(3); r.to_csv(S + 'r11_lasttrade_compare.csv', index=False, encoding='utf-8-sig'); print(r.to_string(index=False))
