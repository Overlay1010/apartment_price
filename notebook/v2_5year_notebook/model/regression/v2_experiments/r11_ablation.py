import os
import sys, time, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
S=os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
exec(open(S+'r11_stage1.py',encoding='utf-8').read().split('rows = []')[0])
FULL=CONFIGS['F5 +구×월 (전체 196)']
EXPECT=['예상가격_만원','주담대_최대한도_만원','주담대_한도비율']
SUPPLY_RAW=reg[reg.role.eq('ablation')&reg.source.eq('04_supply')].column.tolist()
SUPPLY_RATIO=cols(['06_supply_ratio'])
V1MKT=[c for c in reg.column if c.endswith('_v1') and any(k in c for k in ['구_','서울_','전세','월세','금리'])]
PIT_MKT=MARKET+RENT+[c for c in RATE if not c.startswith('기준금리')]
HIGH15=['고가주택_주담대금지_해당','고가주택_주담대금지','고가15억초과_비중']
perm=pd.read_csv(S+'r11_perm_single.csv',index_col=0).iloc[:,0]
NONPOS=perm[perm<=0].index.tolist()
new_va=unseen_va
ABL={
 'A0 전체(196)':FULL,
 'A1 예상가격 계열 제외':[c for c in FULL if c not in EXPECT],
 'A2 + 예상가격_출처':FULL+['예상가격_출처'],
 'A3 + 전세피처출처':FULL+RENT_FLAGS,
 'A4 + 응급·유치원 거리':FULL+['응급의료기관거리_m','유치원거리_m'],
 'A5 공급 재고비율 → 원수치':[c for c in FULL if c not in SUPPLY_RATIO]+SUPPLY_RAW,
 'A6 역 출입구 제외(역 좌표만)':[c for c in FULL if c!='역출입구거리_m'],
 'A7 역 좌표 제외(출입구만)':[c for c in FULL if c!='역세권거리_m'],
 'A8 시장·전월세·금리 V1 시점(_v1)':[c for c in FULL if c not in PIT_MKT]+V1MKT,
 'A9 15억 판정 제외':[c for c in FULL if c not in HIGH15],
 'A10 permutation 0 이하 제외':[c for c in FULL if c not in NONPOS],
 'A11 A1 + A10':[c for c in FULL if c not in NONPOS and c not in EXPECT],
}
rows=[]
for name,feats in ABL.items():
    st=time.time(); X=frame(feats); m=fit(X,tr,va); p=m.predict(X[va]); pv=pd.Series(p,index=np.where(va)[0])
    r={'구성':name,'Feature수':len(feats),'Valid_MAPE':metrics(va,p)['MAPE'],'Valid_중앙APE':metrics(va,p)['중앙APE'],
       'Seen_MAPE':metrics(seen_va,pv.loc[np.where(seen_va)[0]].to_numpy())['MAPE'],
       'Valid신규_MAPE':metrics(new_va,pv.loc[np.where(new_va)[0]].to_numpy())['MAPE']}
    hm=[metrics(ev_h,fit(X,tr_h,es_h,seed=1).predict(X[ev_h]))['MAPE'] for tr_h,ev_h,es_h in holdouts]
    r['홀드아웃_MAPE']=float(np.mean(hm)); r['초']=round(time.time()-st); rows.append(r); print(r,flush=True)
    pd.DataFrame(rows).to_csv(S+'r11_ablation.csv',index=False,encoding='utf-8-sig')
