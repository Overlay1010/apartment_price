# 같은 고정 분할 · 같은 빠른 설정(XGB lr 0.1)으로 V1 시점 Feature vs V2 기본 Feature의 Test 비교
exec(open('r11_stage1.py', encoding='utf-8').read().split('assert set(')[0])
src = open('r11_stage1.py', encoding='utf-8').read()
exec('def frame(' + src.split('def frame(')[1].split('train_complexes')[0])
reg2 = pd.read_csv(D + 'v2_feature_registry.csv'); reg2 = reg2[reg2.dataset.eq('v2_trade')]
A13 = reg2[reg2.role.eq('model')].column.tolist()
for name, feats in [('V1 시점 그대로', V1RAW), ('V1 정의 + V2 시점', CONFIGS['F0 V1정의+V2시점 (출처·응급·유치원·학교거리 포함)']), ('V2 기본(A13)', A13)]:
    Xf = frame(feats); m = fit(Xf, tr, va); p = m.predict(Xf[te]); pv = m.predict(Xf[va])
    print(name, len(feats), 'Valid', round(metrics(va, pv)['MAPE'], 2), 'Test', round(metrics(te, p)['MAPE'], 2),
          '편향', round(float((np.exp(p - y[te]) - 1).mean() * 100), 2), flush=True)
