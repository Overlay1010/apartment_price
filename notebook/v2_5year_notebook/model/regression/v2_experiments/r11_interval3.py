# 신규 단지 구간: 가격대를 합쳐 표본을 늘리고 분위수 수준을 Valid에서 따로 고름 (기존 단지는 창 3개월 · 0.07 유지)
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read().split('t = pd.read_parquet')[0])
import numpy as np, pandas as pd
d = pd.read_parquet(S + 'r11_oof.parquet'); MKT = d['거래유형'].eq('중개거래')
VALM = pd.period_range('2025-07', '2025-12', freq='M'); TESTM = pd.period_range('2026-01', '2026-06', freq='M')
def new_cov(ms, alpha, win):
    out = []
    for m in ms:
        w = d[MKT & d['단지'].eq('신규') & (d['월'] >= m - (win + 1)) & (d['월'] <= m - 2)]['r'].quantile([alpha, 1 - alpha])
        s = d[(d['월'] == m) & d['단지'].eq('신규') & MKT]
        out.append(pd.DataFrame({'포함': (s['r'] >= w[alpha]) & (s['r'] <= w[1 - alpha]), '폭': (np.exp(w[1 - alpha]) - np.exp(w[alpha])) * 100}))
    o = pd.concat(out); return o['포함'].mean() * 100, o['폭'].mean(), len(o)
rows = [dict(창=win, 분위수=a, **dict(zip(['Valid 포함률', 'Valid 폭%', 'n'], new_cov(VALM, a, win)))) for win in (6, 12) for a in (0.07, 0.05, 0.04, 0.03)]
g = pd.DataFrame(rows).round(1); print(g.to_string(index=False))
b = g[g['Valid 포함률'] >= 80].sort_values('Valid 폭%').iloc[0]; print('선택', b.to_dict())
print('Test 신규', dict(zip(['포함률', '폭%', 'n'], np.round(new_cov(TESTM, b['분위수'], int(b['창'])), 1))))
