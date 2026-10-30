# 3단계: 앙상블 비교 — 단일 · 시드 배깅 · 정답 다양화 · 가중 블렌딩 (가중치는 다른 fold에서 학습)
import os, itertools
from scipy.optimize import minimize
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_base.py', encoding='utf-8').read())
P = S + 'r11_preds/'
SPL = ['f1', 'f2', 'h1', 'h2', 'h3']
EV = {'f1': FOLDS['f1'][1], 'f2': FOLDS['f2'][1], **{h: HOLD[h][1] for h in HOLD}}
jobs = sorted({f.rsplit('_', 1)[0] for f in os.listdir(P) if f.endswith('.npy')})
pred = {j: {s: np.load(P + f'{j}_{s}.npy') for s in SPL if os.path.exists(P + f'{j}_{s}.npy')} for j in jobs}
pred = {j: v for j, v in pred.items() if len(v) == len(SPL)}
if all(f'lgb_s{s}' in pred for s in (42, 43, 44)):
    pred['lgb_bag3'] = {s: np.mean([pred[f'lgb_s{k}'][s] for k in (42, 43, 44)], axis=0) for s in SPL}
new_va = va & ~in_train

def row(name, pr, note=''):
    r = {'구성': name, 'fold1': mape(EV['f1'], pr['f1']), 'fold2(Valid)': mape(EV['f2'], pr['f2'])}
    r['fold 평균'] = (r['fold1'] + r['fold2(Valid)']) / 2
    a = pd.Series(ape(va, pr['f2'])); nv = new_va[va]
    r['Valid 기존 단지'] = a[~nv].mean(); r['Valid 신규 단지'] = a[nv].mean()
    r['Valid 10% 이내'] = (a <= 10).mean() * 100
    r['홀드아웃'] = np.mean([mape(EV[h], pr[h]) for h in HOLD]); r['비고'] = note
    return r

def blend_w(members, fit_split):
    P_ = np.vstack([pred[m][fit_split] for m in members]); ev = EV[fit_split]
    obj = lambda w: mape(ev, w @ P_)
    k = len(members); res = minimize(obj, np.ones(k) / k, method='SLSQP', bounds=[(0, 1)] * k,
                                     constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}])
    return res.x

rows = [row(j, pred[j]) for j in pred]
base_members = [m for m in ['lgb_bag3', 'xgb_s42', 'cat_s42', 'lgb_py_s42'] if m in pred]
for k in range(2, len(base_members) + 1):
    for mem in itertools.combinations(base_members, k):
        eq = {s: np.mean([pred[m][s] for m in mem], axis=0) for s in SPL}
        rows.append(row('균등 ' + '+'.join(mem), eq))
        w1, w2 = blend_w(mem, 'f1'), blend_w(mem, 'f2')
        # 교차 적용: fold1 가중치 → fold2 평가, fold2 가중치 → fold1 평가, 홀드아웃은 fold1 가중치 (Valid를 보지 않은 가중치)
        cw = {'f1': w2 @ np.vstack([pred[m]['f1'] for m in mem]), 'f2': w1 @ np.vstack([pred[m]['f2'] for m in mem]),
              **{h: w1 @ np.vstack([pred[m][h] for m in mem]) for h in HOLD}}
        rows.append(row('가중 ' + '+'.join(mem), cw, f'w(f1)={np.round(w1, 2).tolist()} w(f2)={np.round(w2, 2).tolist()}'))
out = pd.DataFrame(rows).round(3).sort_values('fold 평균')
out.to_csv(S + 'r11_blend.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 80)
print(out.to_string(index=False))
