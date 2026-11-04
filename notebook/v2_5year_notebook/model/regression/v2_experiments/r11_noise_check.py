# 11 평가 잡음 점검: 상권 · 배정초 · K-apt처럼 서로 무관한 Feature 묶음이 모두 홀드아웃 MAPE를 0.6%p 안팎 낮췄다 — Feature 효과인가, 평가 잡음인가
# - 기준선을 시드만 바꿔 다시 학습 (홀드아웃 시드 1~3, f1 시드 44~46)
# - 의미 없는 난수 열 7개를 더한 모델: 이것도 좋아지면 "열이 늘어난 효과"(열 샘플링 · 조기 종료가 달라짐)이지 정보 효과가 아니다
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_hakgu_test.py'), encoding='utf-8').read().split("hg = pd.read_parquet")[0])
def fit(feats, trm, esm, seed=42):
    m = LGBMRegressor(**P['lgb'], n_estimators=6000, learning_rate=0.06, subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    m.fit(X.loc[trm, feats], y[trm], eval_set=[(X.loc[esm, feats], y[esm])], callbacks=[early_stopping(100, verbose=False)])
    return m
def score(mask, p):
    a = ape(mask, p); return {'MAPE': a.mean(), 'n': int(mask.sum())}
broker = (t['거래유형'].eq('중개거래') if '거래유형' in t else pd.Series(True, index=t.index)).to_numpy()
rng = np.random.default_rng(7); NOISE = [f'난수_{i}' for i in range(7)]
cx_noise = {c: dict(zip(sorted(set(t['단지일련번호'])), rng.normal(size=t['단지일련번호'].nunique()))) for c in NOISE}
for c in NOISE: X[c] = t['단지일련번호'].map(cx_noise[c]).to_numpy()  # 단지마다 고정된 난수 (단지 속성처럼 생긴 무의미한 값)
rows = []
for k, (trh, evh, esh) in HOLD.items():
    evb = evh & broker
    for seed in (1, 2, 3):
        rows.append({'평가': f'홀드아웃 {k}', '구성': '현재 11', '시드': seed, 'MAPE': score(evb, fit(FEATS, trh, esh, seed=seed).predict(X.loc[evb, FEATS]))['MAPE']}); print(rows[-1], flush=True)
    rows.append({'평가': f'홀드아웃 {k}', '구성': '+ 난수 7개', '시드': 1, 'MAPE': score(evb, fit(FEATS + NOISE, trh, esh, seed=1).predict(X.loc[evb, FEATS + NOISE]))['MAPE']}); print(rows[-1], flush=True)
trm, evm = FOLDS['f1']; evb = evm & broker
for seed in (44, 45, 46):
    rows.append({'평가': 'f1', '구성': '현재 11', '시드': seed, 'MAPE': score(evb, fit(FEATS, trm, evm, seed=seed).predict(X.loc[evb, FEATS]))['MAPE']}); print(rows[-1], flush=True)
for seed in (42, 43):
    m = fit(FEATS + NOISE, trm, evm, seed=seed)
    rows.append({'평가': 'f1', '구성': '+ 난수 7개', '시드': seed, 'MAPE': score(evb, m.predict(X.loc[evb, FEATS + NOISE]))['MAPE'], '트리수': m.best_iteration_}); print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv(S + 'r11_noise_check.csv', index=False, encoding='utf-8-sig')
print(d.groupby(['평가', '구성'])['MAPE'].agg(['mean', 'min', 'max', 'count']).round(3).to_string())
