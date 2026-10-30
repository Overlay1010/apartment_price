# 4단계 (2): 이동 창 잔차 분위수 구간 — 분위수 수준·창 길이는 Valid 기간에서 정하고 Test에서 확인
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read())
per = t['거래월일자'].dt.to_period('M'); months = pd.period_range('2022-01', '2026-06', freq='M')
parts = []
for m in months:
    evm = (per == m).to_numpy(); p = np.load(S + f'r11_roll/{m}.npy')
    seen = t.loc[evm, '단지일련번호'].isin(set(t.loc[(per <= m - 2).to_numpy(), '단지일련번호'])).to_numpy()
    parts.append(pd.DataFrame({'transaction_id': t.loc[evm, 'transaction_id'].to_numpy(), '월': m, 'p': p, 'y': y[evm],
                               '단지': np.where(seen, '기존', '신규'), '거래유형': t.loc[evm, '거래유형'].to_numpy()}))
d = pd.concat(parts, ignore_index=True); d['r'] = d['y'] - d['p']
d['가격대'] = pd.cut(np.exp(d['p']) / 1e4, [0, 6, 9, 15, np.inf], labels=['6억 미만', '6~9억', '9~15억', '15억 이상'], right=False).astype(str)
d['ape'] = np.abs(1 - np.exp(d['p'] - d['y'])) * 100
d.to_parquet(S + 'r11_oof.parquet')
MKT = d['거래유형'].eq('중개거래'); SEG = ['단지', '가격대']
VALM = pd.period_range('2025-07', '2025-12', freq='M'); TESTM = pd.period_range('2026-01', '2026-06', freq='M')
print('월별 MAPE(중개)', d[MKT].groupby(d['월'].dt.year)['ape'].mean().round(2).to_dict())

def intervals(m, alpha, win):
    w = d[MKT & (d['월'] >= m - (win + 1)) & (d['월'] <= m - 2)]
    q = w.groupby(SEG)['r'].quantile([alpha, 1 - alpha]).unstack(); qa = w['r'].quantile([alpha, 1 - alpha])
    sub = d[(d['월'] == m)].copy(); key = pd.MultiIndex.from_frame(sub[SEG])
    sub['lo'] = q[alpha].reindex(key).fillna(qa[alpha]).to_numpy(); sub['hi'] = q[1 - alpha].reindex(key).fillna(qa[1 - alpha]).to_numpy()
    return sub

def evaluate(ms, alpha, win):
    r = pd.concat([intervals(m, alpha, win) for m in ms], ignore_index=True)
    r['포함'] = (r['r'] >= r['lo']) & (r['r'] <= r['hi']); r['폭%'] = (np.exp(r['hi']) - np.exp(r['lo'])) * 100
    return r

grid = []
for win in (3, 6, 12):
    for alpha in (0.10, 0.08, 0.07, 0.06, 0.05):
        r = evaluate(VALM, alpha, win); mk = r['거래유형'].eq('중개거래')
        grid.append({'창(개월)': win, '분위수': alpha, 'Valid 포함률': r.loc[mk, '포함'].mean() * 100, 'Valid 폭%': r.loc[mk, '폭%'].mean(),
                     'Valid 최저 세그먼트 포함률': r[mk].groupby(SEG)['포함'].mean().min() * 100})
g = pd.DataFrame(grid).round(2); print(g.to_string(index=False))
ok = g[g['Valid 포함률'] >= 80]; best = ok.sort_values('Valid 폭%').iloc[0]
print('선택', best.to_dict())
for nm, (win, alpha) in {'기본(0.10, 6개월)': (6, 0.10), '선택': (int(best['창(개월)']), float(best['분위수']))}.items():
    r = evaluate(TESTM, alpha, win); mk = r['거래유형'].eq('중개거래')
    print('== Test', nm, '포함률', round(r.loc[mk, '포함'].mean() * 100, 1), '폭%', round(r.loc[mk, '폭%'].mean(), 1), '직거래 포함률', round(r.loc[~mk, '포함'].mean() * 100, 1))
    print(r[mk].groupby(SEG)[['포함', '폭%']].mean().mul([100, 1]).round(1).unstack(0).to_string())
    print(r[mk].groupby(r['월'].astype(str))['포함'].mean().mul(100).round(1).to_dict())
    if nm == '선택':
        def grade(v, lo, hi):
            if lo <= v <= hi: return '적정'
            k = (v - hi) / hi if v > hi else (lo - v) / -lo   # 구간 반폭 대비 초과 정도
            s = '고평가' if v > hi else '저평가'                 # 실제 거래가가 적정가보다 높으면 고평가
            return f'다소 {s}' if k <= 1 else s
        r['등급'] = [grade(a, b, c) for a, b, c in zip(r['r'], r['lo'], r['hi'])]
        print(r[mk]['등급'].value_counts(normalize=True).mul(100).round(1).to_dict())
        print('직거래', r[~mk]['등급'].value_counts(normalize=True).mul(100).round(1).to_dict())
        r.to_parquet(S + 'r11_interval_test.parquet')
g.to_csv(S + 'r11_interval_grid.csv', index=False, encoding='utf-8-sig')
