# 11 공통: 데이터 · 기본 Feature(레지스트리 role=model) · 분할 · 지표 · 단지 홀드아웃
import os
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), *['..'] * 5, ''); D = ROOT + 'data/v2_5year_seoul/processed/'
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
reg = pd.read_csv(D + 'v2_feature_registry.csv'); reg = reg[reg.dataset.eq('v2_trade')]
FEATS = reg[reg.role.eq('model')].column.tolist()
t = pd.read_parquet(D + 'v2_trade.parquet'); t['단지일련번호'] = t['단지일련번호'].astype(str)
X = t[FEATS].copy()
CAT = [c for c in FEATS if X[c].dtype == object]
for c in CAT: X[c] = X[c].astype('category')
y = np.log(t['거래금액'].to_numpy(float))
PY = np.log(t['전용면적'].to_numpy(float) * 0.3025)  # log(평) — 평당가 정답용 오프셋
split = t['데이터구분'].to_numpy(); tr, va, te = split == 'train', split == 'valid', split == 'test'
mon = t['거래월일자']
f1_tr = (mon < '2025-01-01').to_numpy(); f1_ev = tr & ~f1_tr
FOLDS = {'f1': (f1_tr, f1_ev), 'f2': (tr, va)}
train_cx = set(t.loc[tr, '단지일련번호']); in_train = t['단지일련번호'].isin(train_cx).to_numpy()
rng = np.random.default_rng(0); uniq = np.array(sorted(train_cx)); HOLD = {}
for s in range(3):
    held = t['단지일련번호'].isin(set(rng.choice(uniq, int(0.2 * len(uniq)), replace=False))).to_numpy()
    HOLD[f'h{s+1}'] = (tr & ~held, va & held, va & ~held)  # 학습, 평가(홀드아웃 단지), 조기 종료
def ape(mask, p): a = np.exp(y[mask]); return np.abs(a - np.exp(p)) / a * 100
def mape(mask, p): return float(ape(mask, p).mean())
