# 11 상권 점검 2: 두 검증 구간(f1: 2024-07~12, f2: 2025-01~06) × 시드 2개 — 처음 보는 단지 표본이 작아(328건) 구간 · 시드를 늘려 확인
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_sg_test.py'), encoding='utf-8').read().split('rows = []')[0])
rows = []
for fold, (trm, evm) in FOLDS.items():
    seen_cx = set(t.loc[trm, '단지일련번호']); old = t['단지일련번호'].isin(seen_cx).to_numpy()
    for seed in (42, 43):
        for name, feats in [('현재 11 (199)', FEATS), ('+ 상권 7개', FEATS + SG)]:
            m = fit(feats, trm, evm, seed=seed); pv = pd.Series(m.predict(X.loc[evm, feats]), index=np.where(evm)[0])
            r = {'구간': fold, '시드': seed, '구성': name}
            for lab, mk in [('전체', evm & broker), ('기존 단지', evm & broker & old), ('처음 보는 단지', evm & broker & ~old)]:
                s = score(mk, pv.loc[np.where(mk)[0]].to_numpy()); r[f'{lab} MAPE'] = round(s['MAPE'], 3); r[f'{lab} n'] = s['n']
            rows.append(r); print(r, flush=True)
d = pd.DataFrame(rows); d.to_csv(S + 'r11_sg_test2.csv', index=False, encoding='utf-8-sig')
print(d.groupby(['구간', '구성'])[['전체 MAPE', '기존 단지 MAPE', '처음 보는 단지 MAPE']].mean().round(3).to_string())
