# 11 보완 후보 점검: 학구도 배정 초등학교 Feature 8개를 더하면 나아지는가 (채택 규칙: 두 구간 모두 0.1%p 이상 개선 + 홀드아웃 단지 악화 없음)
# - 기준선(현재 11)은 같은 코드 · 같은 시드로 돌린 r11_sg_test2.csv(구간 × 시드), r11_sg_test.csv(홀드아웃) 값을 그대로 쓴다
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_sg_test.py'), encoding='utf-8').read().split('sg = pd.read_parquet')[0])
P = json.load(open(S + 'r11_best_params_lasttrade.json', encoding='utf-8'))
hg = pd.read_parquet(S + 'hakgu_feat.parquet'); HG = [c for c in hg.columns if c.startswith('배정초_') and c != '배정초_학교명']
X = X.join(t[['transaction_id']].merge(hg, on='transaction_id', how='left')[HG].set_axis(X.index))
def fit(feats, trm, esm, seed=42):
    m = LGBMRegressor(**P['lgb'], n_estimators=6000, learning_rate=0.06, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    m.fit(X.loc[trm, feats], y[trm], eval_set=[(X.loc[esm, feats], y[esm])], callbacks=[early_stopping(100, verbose=False)])
    return m
def score(mask, p):
    a = ape(mask, p); return {'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(mask.sum())}
broker = (t['거래유형'].eq('중개거래') if '거래유형' in t else pd.Series(True, index=t.index)).to_numpy()
NAME = f'+ 배정초 {len(HG)}개'
rows = []
for fold, (trm, evm) in FOLDS.items():
    old = t['단지일련번호'].isin(set(t.loc[trm, '단지일련번호'])).to_numpy()
    for seed in (42, 43):
        m = fit(FEATS + HG, trm, evm, seed=seed); pv = pd.Series(m.predict(X.loc[evm, FEATS + HG]), index=np.where(evm)[0])
        r = {'구간': fold, '시드': seed, '구성': NAME}
        for lab, mk in [('전체', evm & broker), ('기존 단지', evm & broker & old), ('처음 보는 단지', evm & broker & ~old)]:
            s = score(mk, pv.loc[np.where(mk)[0]].to_numpy()); r[f'{lab} MAPE'] = round(s['MAPE'], 3); r[f'{lab} n'] = s['n']
        if fold == 'f2' and seed == 42:
            imp = pd.Series(m.booster_.feature_importance('gain'), index=FEATS + HG); r['gain 순위'] = imp.rank(ascending=False)[HG].astype(int).to_dict()
        rows.append(r); print(r, flush=True)
hs = []
for k, (trh, evh, esh) in HOLD.items():
    mh = fit(FEATS + HG, trh, esh, seed=1); evb = evh & broker; hs.append(score(evb, mh.predict(X.loc[evb, FEATS + HG]))); print(k, hs[-1], flush=True)
base = pd.read_csv(S + 'r11_sg_test2.csv'); base = base[base['구성'].str.startswith('현재')]
d = pd.concat([base, pd.DataFrame(rows)], ignore_index=True); d.to_csv(S + 'r11_hakgu_test.csv', index=False, encoding='utf-8-sig')
print(d.groupby(['구간', '구성'])[['전체 MAPE', '기존 단지 MAPE', '처음 보는 단지 MAPE']].mean().round(3).to_string())
hb = pd.read_csv(S + 'r11_sg_test.csv').iloc[0]
print(f"홀드아웃 단지 MAPE: 현재 11 {hb['홀드아웃 단지 MAPE']:.3f} → {NAME} {np.mean([h['MAPE'] for h in hs]):.3f} (회차별 {[round(h['MAPE'], 3) for h in hs]})")
