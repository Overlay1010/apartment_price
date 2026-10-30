import os
import sys, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
S=os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
exec(open(S+'r11_ablation.py',encoding='utf-8').read().split('rows=[]')[0])
# 예상가격을 단지 자체 이력일 때만 사용 (동·면적대·구 대체값은 결측 처리)
own = t['예상가격_출처'].eq('단지').to_numpy()
t['예상가격_단지만'] = t['예상가격_만원'].where(own)
src = t['예상가격_출처'].to_numpy()
def run(name, feats, seeds=(42,), holdout=True):
    X=frame(feats); r={'구성':name,'Feature수':len(feats)}
    vs=[]; 
    for sd in seeds:
        p=fit(X,tr,va,seed=sd).predict(X[va]); vs.append(p)
    p=np.mean(vs,axis=0) if len(seeds)==1 else vs
    for i,pp in enumerate(vs):
        ape=np.abs(np.exp(y[va])-np.exp(pp))/np.exp(y[va])*100
        r[f'Valid_MAPE_s{seeds[i]}']=ape.mean()
    pp=vs[0]; ape=pd.Series(np.abs(np.exp(y[va])-np.exp(pp))/np.exp(y[va])*100)
    nv=unseen_va[va]; r['Seen_MAPE']=ape[~nv].mean(); r['Valid신규_MAPE']=ape[nv].mean()
    for s in ['단지','동·면적대','구']:
        r[f'신규_{s}']=ape[nv & (src[va]==s)].mean()
    if holdout:
        r['홀드아웃_MAPE']=float(np.mean([metrics(ev_h,fit(X,tr_h,es_h,seed=1).predict(X[ev_h]))['MAPE'] for tr_h,ev_h,es_h in holdouts]))
    print(r,flush=True); return r
rows=[]
rows.append(run('A0 전체(196) 시드 3개',FULL,seeds=(42,43,44),holdout=False))
rows.append(run('A1 예상가격 계열 제외',[c for c in FULL if c not in EXPECT],holdout=False))
rows.append(run('A12 예상가격 → 단지 이력만',[c for c in FULL if c!='예상가격_만원']+['예상가격_단지만']))
rows.append(run('A13 A12 + 주담대 계열 제외',[c for c in FULL if c not in EXPECT]+['예상가격_단지만']))
pd.DataFrame(rows).to_csv(S+'r11_abl2.csv',index=False,encoding='utf-8-sig')
