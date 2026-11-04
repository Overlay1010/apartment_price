import os
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width',220)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_stage1.py',encoding='utf-8').read().split('rows = []')[0])
nv=t[unseen_va]
print('Valid 신규 단지 거래', unseen_va.sum(), '| 단지', nv['단지일련번호'].nunique(), '| 연식 중앙', nv['연식'].median(), '| 신축(연식<=1) 비율', (nv['연식']<=1).mean().round(3), '| 거래유형', nv['거래유형'].value_counts(normalize=True).round(2).to_dict())
print('공시 결측 비율(신규)', nv['공시_m2당_면적대'].isna().mean().round(3), '| 전체 Valid', t.loc[va,'공시_m2당_면적대'].isna().mean().round(3))
res={}
for name in ['F3 +단지속성+공시','F5 +구×월 (전체 196)']:
    X=frame(CONFIGS[name]); m=fit(X,tr,va); p=m.predict(X[va]); a=np.exp(y[va]); ape=np.abs(a-np.exp(p))/a*100
    s=pd.Series(ape,index=np.where(va)[0]); res[name]=(m,X,s)
    u=s.loc[np.where(unseen_va)[0]]
    print(name,'신규 단지 APE 중앙', round(u.median(),2), '평균', round(u.mean(),2), '| 상위 5% 제외 평균', round(u[u<u.quantile(.95)].mean(),2))
# 신규 단지별 오차 비교
both=pd.DataFrame({'F3':res['F3 +단지속성+공시'][2].loc[np.where(unseen_va)[0]],'F5':res['F5 +구×월 (전체 196)'][2].loc[np.where(unseen_va)[0]]})
both['단지']=t.loc[both.index,'단지명'].to_numpy(); both['구']=t.loc[both.index,'시군구명'].to_numpy(); both['공시없음']=t.loc[both.index,'공시_m2당_면적대'].isna().to_numpy()
print(both.groupby('공시없음')[['F3','F5']].agg(['count','mean']).round(2))
print(both.groupby(['단지','구'])[['F3','F5']].mean().assign(diff=lambda d:d.F5-d.F3).sort_values('diff').tail(8).round(1))
# permutation importance (묶음 단위) on F5
m,X,s=res['F5 +구×월 (전체 196)']; Xv=X[va]; base=np.abs(np.exp(y[va])-np.exp(m.predict(Xv)))/np.exp(y[va])*100; b=base.mean()
groups={'V1 기본':V1,'시장(확정월)':MARKET,'전월세(확정월)':RENT,'금리':RATE,'위치(시점)':LOC,'단지속성':COMPLEX,'공시가격':GONGSI,'학교·학원':SCHOOL,'구×월(01·03·04·05)':REGION}
rng=np.random.default_rng(0); out={}
for g,cs in groups.items():
    Xp=Xv.copy(); idx=rng.permutation(len(Xp))
    for c in cs: Xp[c]=Xv[c].to_numpy()[idx] if not hasattr(Xv[c],'cat') else pd.Categorical(Xv[c].to_numpy()[idx],categories=Xv[c].cat.categories)
    pp=m.predict(Xp); out[g]=(np.abs(np.exp(y[va])-np.exp(pp))/np.exp(y[va])*100).mean()-b
print('묶음 permutation: MAPE 증가(%p)', {k:round(v,2) for k,v in sorted(out.items(),key=lambda x:-x[1])})
single={}
for c in X.columns:
    Xp=Xv.copy(); v=Xv[c].to_numpy()[rng.permutation(len(Xv))]
    Xp[c]=pd.Categorical(v,categories=Xv[c].cat.categories) if str(Xv[c].dtype)=='category' else v
    single[c]=(np.abs(np.exp(y[va])-np.exp(m.predict(Xp)))/np.exp(y[va])*100).mean()-b
ss=pd.Series(single).sort_values(ascending=False); print('단일 Feature 상위 15'); print(ss.head(15).round(3)); print('MAPE 증가 0 이하 Feature 수', (ss<=0).sum(), '/', len(ss))
ss.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '') + 'r11_perm_single.csv',encoding='utf-8-sig')
