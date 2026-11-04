# 31 점검: 빨리 변하는 주간 지표(구별 전세 지수 · 권역별 매매/전세 수급)를 더하면 나아지는가 (DEV · Test)
# 시점 규칙은 매매 지수와 같다: 월요일 기준 주 값은 그 주 목요일(+3일) 공표 → 예측시점 p 전에 공표된 주만 쓴다
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r31_base.py'), encoding='utf-8').read())
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
W = ROOT + 'data/external/rone_weekly/'
GU_REGION = {'종로구': '도심권', '중구': '도심권', '용산구': '도심권', '성동구': '동북권', '광진구': '동북권', '동대문구': '동북권', '중랑구': '동북권',
             '성북구': '동북권', '강북구': '동북권', '도봉구': '동북권', '노원구': '동북권', '은평구': '서북권', '서대문구': '서북권', '마포구': '서북권',
             '양천구': '서남권', '강서구': '서남권', '구로구': '서남권', '금천구': '서남권', '영등포구': '서남권', '동작구': '서남권', '관악구': '서남권',
             '서초구': '동남권', '강남구': '동남권', '송파구': '동남권', '강동구': '동남권'}
js = pd.read_csv(W + 'apt_jeonse_index_weekly_gu.csv', dtype={'주': str}); js['주'] = pd.to_datetime(js['주'])
jw = np.log(js.pivot(index='주', columns='자치구', values='지수').sort_index())
sd = pd.read_csv(W + 'apt_supply_demand_weekly_seoul.csv', dtype={'주': str}); sd['주'] = pd.to_datetime(sd['주'])
sw = {c: sd.pivot(index='주', columns='권역', values=c).sort_index() for c in ('매매수급', '전세수급')}
rows = []
for p in r['p'].unique():
    day = p.to_timestamp()
    j = jw[jw.index + pd.Timedelta(days=3) < day]; s = {c: v[v.index + pd.Timedelta(days=3) < day] for c, v in sw.items()}
    if len(j) < 14: continue
    for gu in jw.columns:
        reg_ = GU_REGION[gu]
        rows.append({'p': p, '자치구': gu, '전세_4주변화': j[gu].iloc[-1] - j[gu].iloc[-5], '전세_13주변화': j[gu].iloc[-1] - j[gu].iloc[-14],
                     '매매수급_권역': s['매매수급'][reg_].iloc[-1], '매매수급_권역_4주변화': s['매매수급'][reg_].iloc[-1] - s['매매수급'][reg_].iloc[-5],
                     '전세수급_권역': s['전세수급'][reg_].iloc[-1], '전세수급_권역_4주변화': s['전세수급'][reg_].iloc[-1] - s['전세수급'][reg_].iloc[-5],
                     '매매수급_서울': s['매매수급']['서울'].iloc[-1], '매매수급_서울_4주변화': s['매매수급']['서울'].iloc[-1] - s['매매수급']['서울'].iloc[-5]})
r = r.merge(pd.DataFrame(rows), on=['p', '자치구'], how='left')
NEW = ['전세_4주변화', '전세_13주변화', '매매수급_권역', '매매수급_권역_4주변화', '전세수급_권역', '전세수급_권역_4주변화', '매매수급_서울', '매매수급_서울_4주변화']
chg = r['공식_향후3개월_변화율']
LABELS = {'상승세(+1%)': r['상승여부'].copy(), '상대 상위 1/3': (chg.groupby(r['p']).rank(ascending=False, method='first') <= 8).astype(int)}
def logit(): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
def lgbm(): return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=42)
OFF = G['공식지수']
SETS = {'공식지수': OFF, '공식지수 + 주간 전세': OFF + NEW[:2], '공식지수 + 주간 수급': OFF + NEW[2:], '공식지수 + 주간 전세 · 수급': OFF + NEW,
        '공식지수 + 확정월 + 주간 전세 · 수급': OFF + G['시장(확정월)'] + G['전월세(확정월)'] + NEW}
out = []
for lname, lab in LABELS.items():
    r['상승여부'] = lab
    for period, months in [('DEV', DEV), ('Test', TEST)]:
        t = r[r['p'].isin(months)]
        bl = metrics(t['상승여부'].to_numpy(), t['공식지수_4주변화'].rank(pct=True).to_numpy(), t['p'].astype(str).to_numpy())
        out.append({'정답': lname, '기간': period, '방법': '기준선: 4주 흐름', 'ROC_AUC': bl['ROC_AUC'], '월내_AUC': bl['월내_AUC']})
        for sname, feats in SETS.items():
            for mname, mk in [('로지스틱', logit), ('LightGBM', lgbm)]:
                pr = walk_forward(mk, feats, months)
                m = metrics(pr['y'].to_numpy(), pr['prob'].to_numpy(), pr['p'].astype(str).to_numpy())
                out.append({'정답': lname, '기간': period, '방법': f'{mname} · {sname}', 'ROC_AUC': m['ROC_AUC'], '월내_AUC': m['월내_AUC'], 'Brier': m['Brier']})
        for o in out[-11:]: print(o['정답'], o['기간'], o['방법'], round(o['ROC_AUC'], 3), round(o['월내_AUC'], 3), flush=True)
pd.DataFrame(out).to_csv(HERE + 'r31_weekly_extra.csv', index=False, encoding='utf-8-sig')
