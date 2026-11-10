# 11 보완 5: Feature 대신 학습 방법 비교 — 잔차 학습 · L1 손실 · 최근 거래 가중치 (지금까지 시험한 적 없음)
# - 근거: 오차의 72%가 같은 면적 직전 거래 6개월 내 거래에서 나온다(r11_error_share). Feature 추가 3회(보완 3 · 4)는 모두 0.1%p 미만
# - 설정은 본 모델 그대로(r11_best_params_idx, 학습률 0.03, 트리 1,805개 고정, 평가 구간 조기 종료 없음), Feature도 기본 201개 그대로
#
# [판정 방식 보강 — 결과를 보기 전에 고정, 2026-11-10]
#   f1 하나는 Feature 하나만 더해도 0.2~0.4%p 흔들려(보완 3 · 4) 시간 fold와 시드를 늘린다
#   - 시간 fold 4개: f0a 2024 상반기 · f0b 2024 하반기 · f1 2025 상반기 · f2 2025 하반기(Valid). 각 fold는 평가 시작 전 거래로만 학습
#   - 단지 홀드아웃 3개(h1~h3, 기존과 같음), 시드 3개(42 · 43 · 44), 평가는 중개거래, Test는 보지 않는다
#   - 채택 규칙: (1) 시간 fold 4개 × 시드 3개 평균 개선 ≥ 0.1%p
#                (2) 시간 fold 4개 각각 시드 평균 개선 > 0 (어느 국면에서도 나빠지지 않음)
#                (3) 시드 3개 각각 시간 fold 평균 개선 > 0
#                (4) 홀드아웃 3개 × 시드 3개 평균 개선 ≥ 0
#   - 이 규칙은 보완 5 후보에만 쓴다. 보완 3 · 4에서 불채택된 후보를 이 규칙으로 다시 판정하지 않는다
#
# [후보 — 결과를 보기 전에 고정]
#   잔차 학습: 정답은 그대로 log(거래금액), LightGBM 시작값(init_score)을 기준가의 log로 준다 → 나무는 기준가 대비 차이를 배운다
#              기준가: 직전거래가_시세보정 → 없으면 예상가격_단지만 × k1 → 없으면 공시기반_추정가 × k2 → 없으면 학습 평균
#              k1 · k2는 그 fold 학습 거래의 (거래금액 / 기준가) 중위수
#   L1 손실:   objective='l1' (지금은 log 가격 제곱오차, 평가는 MAPE)
#   최근 가중: 학습 마지막 달에서 멀수록 가중치 0.5^(개월 / 12) (반감기 12개월)
# 실행: python r11_method_test.py           (결과 CSV에 이어 쓰기, 후보 하나가 끝날 때마다 중간 요약)
#       python r11_method_test.py --smoke   (트리 50개 · f0a · 시드 42로 동작만 확인, 저장 안 함)
import os, sys, json, time
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r11_base.py'), encoding='utf-8').read())
from lightgbm import LGBMRegressor
P = json.load(open(S + 'r11_best_params_idx.json', encoding='utf-8'))
SMOKE = '--smoke' in sys.argv
SEEDS = [42] if SMOKE else [42, 43, 44]
N_TREES = 50 if SMOKE else P['best_iteration']
broker = t['거래유형'].eq('중개거래').to_numpy()


def time_fold(start, end):
    return (tr & (mon < start)).to_numpy(), (tr & (mon >= start) & (mon < end)).to_numpy()


TIME = {'f0a': time_fold('2024-01-01', '2024-07-01'), 'f0b': time_fold('2024-07-01', '2025-01-01'), 'f1': FOLDS['f1'], 'f2': FOLDS['f2']}
assert (TIME['f1'][0] == FOLDS['f1'][0]).all() and (TIME['f1'][1] == time_fold('2025-01-01', '2025-07-01')[1]).all()
EVALS = {**TIME, **{k: (trh, evh) for k, (trh, evh, _) in HOLD.items()}}
if SMOKE: EVALS = {'f0a': EVALS['f0a']}

price = t['거래금액'].to_numpy(float)
B0, B1, B2 = (t[c].to_numpy(float) for c in ['직전거래가_시세보정', '예상가격_단지만', '공시기반_추정가'])


def baseline(trm):
    k1 = np.nanmedian((price / B1)[trm]); k2 = np.nanmedian((price / B2)[trm])
    b = np.where(np.isnan(B0), B1 * k1, B0); b = np.where(np.isnan(b), B2 * k2, b)
    return np.where(np.isnan(b), y[trm].mean(), np.log(np.where(np.isnan(b), 1, b))), (k1, k2)  # log 단위


months = (mon.dt.year * 12 + mon.dt.month).to_numpy()


def fit_predict(cfg, trm, evm, seed):
    kw = dict(**P['lgb'], n_estimators=N_TREES, learning_rate=P['learning_rate'], subsample_freq=1, random_state=seed, n_jobs=8, verbose=-1)
    fit_kw, add = {}, 0
    if cfg == 'L1 손실': kw['objective'] = 'l1'
    if cfg == '최근 가중':
        age = months[trm].max() - months[trm]; fit_kw['sample_weight'] = 0.5 ** (age / 12)
    if cfg == '잔차 학습':
        init, _ = baseline(trm); fit_kw['init_score'] = init[trm]; add = init[evm]
    m = LGBMRegressor(**kw).fit(X.loc[trm, FEATS], y[trm], **fit_kw)
    return m.predict(X.loc[evm, FEATS]) + add  # sklearn predict는 init_score를 더하지 않는다


CONFIGS = ['현재 11', '잔차 학습', 'L1 손실', '최근 가중']
if SMOKE:
    trm, evm = EVALS['f0a']; init, k = baseline(trm)
    src = np.select([~np.isnan(B0), ~np.isnan(B1), ~np.isnan(B2)], ['직전 시세보정', '단지 예상가', '공시 추정'], '학습 평균')
    print('기준가 출처', pd.Series(src[evm & broker]).value_counts(normalize=True).round(3).to_dict(), 'k1 k2', np.round(k, 3))
    print(f'기준가 단독 MAPE (f0a 중개) {ape(evm & broker, init[evm & broker]).mean():.3f}%')
    for cfg in CONFIGS:
        st = time.time(); a = ape(evm & broker, fit_predict(cfg, trm, evm & broker, 42))
        print(f'{cfg}: 트리 {N_TREES}개 MAPE {a.mean():.3f}% ({time.time() - st:.0f}초)')
    sys.exit()

out_path = S + 'r11_method_test.csv'
done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=['평가', '구성', '시드'])
rows = done.to_dict('records')
# '현재 11'의 f1 · f2 · h1~h3 시드 42 · 43은 같은 코드 · 설정으로 돌린 r11_nb_test.csv 결과를 그대로 쓴다
prev = pd.read_csv(S + 'r11_nb_test.csv'); prev = prev[prev['구성'].eq('현재 11')]
have = {(r['평가'], r['구성'], r['시드']) for r in rows}
for r in prev[['평가', '구성', '시드', 'MAPE', '10%이내', 'n']].to_dict('records'):
    if (r['평가'], r['구성'], r['시드']) not in have: rows.append({**r, '출처': 'r11_nb_test'})
have = {(r['평가'], r['구성'], r['시드']) for r in rows}


def summary(d):
    d = d.copy(); base = d[d['구성'].eq('현재 11')].set_index(['평가', '시드'])['MAPE']
    x = d[~d['구성'].eq('현재 11')].copy()
    print('\n[MAPE] 시드 평균'); print(d.pivot_table(index='구성', columns='평가', values='MAPE').round(3).to_string())
    print('\n시드 간 차이(최대-최소):'); print(d.groupby(['평가', '구성'])['MAPE'].agg(lambda s: s.max() - s.min()).unstack().round(3).to_string())
    x = x[[k in base.index for k in zip(x['평가'], x['시드'])]]
    if x.empty: return
    x['개선'] = base.loc[list(zip(x['평가'], x['시드']))].to_numpy() - x['MAPE'].to_numpy()
    print('\n현재 11 대비 개선 (%p, +가 좋아짐):'); print(x.pivot_table(index=['구성', '시드'], columns='평가', values='개선').round(3).to_string())
    for cfg, g in x.groupby('구성'):
        tm = g[g['평가'].isin(TIME)]; ho = g[g['평가'].str.startswith('h')]
        if tm['평가'].nunique() < 4 or tm['시드'].nunique() < 3 or len(ho) < 9:
            print(f'{cfg}: 아직 결과가 다 모이지 않음'); continue
        c1 = tm['개선'].mean(); c2 = tm.groupby('평가')['개선'].mean(); c3 = tm.groupby('시드')['개선'].mean(); c4 = ho['개선'].mean()
        ok = [c1 >= 0.1, (c2 > 0).all(), (c3 > 0).all(), c4 >= 0]
        print(f"{cfg}: (1) 시간 fold 평균 {c1:+.3f} {'O' if ok[0] else 'X'} | (2) fold별 {c2.round(3).to_dict()} {'O' if ok[1] else 'X'} | "
              f"(3) 시드별 {c3.round(3).to_dict()} {'O' if ok[2] else 'X'} | (4) 홀드아웃 {c4:+.3f} {'O' if ok[3] else 'X'} → {'채택' if all(ok) else '불채택'}")


for cfg in CONFIGS:
    for seed in SEEDS:
        for ev, (trm, evm) in EVALS.items():
            if (ev, cfg, seed) in have: continue
            evb = evm & broker
            st = time.time(); a = ape(evb, fit_predict(cfg, trm, evb, seed))
            row = {'평가': ev, '구성': cfg, '시드': seed, 'MAPE': a.mean(), '10%이내': (a <= 10).mean() * 100, 'n': int(evb.sum()), '초': round(time.time() - st)}
            rows.append(row); have.add((ev, cfg, seed)); print(row, flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n===== {cfg} 완료 ====='); summary(pd.DataFrame(rows)); sys.stdout.flush()
