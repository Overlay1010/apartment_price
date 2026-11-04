# 신규 단지(학습에 없는 단지)만 균등 블렌드로 보내면 나아지는지 — fold1 · fold2 모두에서 확인
exec(open('r11_blend.py', encoding='utf-8').read().split('rows = [row')[0])
f1_cx = set(t.loc[f1_tr, '단지일련번호'])
NEWM = {'f1': ~t['단지일련번호'].isin(f1_cx).to_numpy(), 'f2': ~in_train}
mem = ['lgb_bag3', 'xgb_s42', 'cat_s42', 'lgb_py_s42']
for s in ['f1', 'f2']:
    ev = EV[s]; nm = NEWM[s][ev]
    base = pred['lgb_bag3'][s]; eq = np.mean([pred[m][s] for m in mem], axis=0)
    routed = np.where(nm, eq, base)
    a0, a1 = ape(ev, base), ape(ev, routed)
    print(s, '신규 건수', nm.sum(), f'전체 {a0.mean():.3f} → {a1.mean():.3f}', f'신규 {a0[nm].mean():.2f} → {a1[nm].mean():.2f}',
          f'기존 {a0[~nm].mean():.3f}')
