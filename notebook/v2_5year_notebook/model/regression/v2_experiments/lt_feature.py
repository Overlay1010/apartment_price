# 프로토타입 (06 반영 전): 같은 단지 · 같은 면적(전용 1㎡ 반올림) 직전 거래 Feature — 계약일 30일 전까지 계약된 거래만 (신고 기한 30일)
import numpy as np, pandas as pd
import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
D = HERE + '../../../../../data/v2_5year_seoul/processed/'
def last_trade_features(trade, warm):
    ref_main = trade.loc[trade['거래유형'].ne('직거래'), ['단지일련번호', '전용면적', '계약일자', '거래금액', '층']]
    w = warm[warm['해제여부'].ne('O') & warm['거래유형'].ne('직거래')].copy()
    w['계약일자'] = pd.to_datetime(dict(year=w['계약년도'], month=w['계약월'], day=w['계약일']))
    w['거래금액'] = pd.to_numeric(w['거래금액'].astype(str).str.replace(',', ''), errors='coerce')
    ref = pd.concat([w[['단지일련번호', '전용면적', '계약일자', '거래금액', '층']], ref_main], ignore_index=True)
    ref['단지일련번호'] = ref['단지일련번호'].astype(str); ref['면적키'] = ref['전용면적'].astype(float).round(0)
    ref['거래금액'] = ref['거래금액'].astype(float)
    ref = ref.sort_values('계약일자', kind='stable')
    g = ref.groupby(['단지일련번호', '면적키'])
    ref['직전3건_중위'] = g['거래금액'].transform(lambda s: s.rolling(3, min_periods=1).median())
    ref['누적건수'] = g.cumcount() + 1
    ref['사용가능일'] = ref['계약일자'] + pd.Timedelta(days=30)
    # 같은 날 여러 건이면 마지막 행이 3건 중위 · 누적을 모두 반영 → 마지막만 남김
    ref = ref.drop_duplicates(['단지일련번호', '면적키', '사용가능일'], keep='last')
    q = trade[['transaction_id', '단지일련번호', '전용면적', '계약일자']].copy()
    q['단지일련번호'] = q['단지일련번호'].astype(str); q['면적키'] = q['전용면적'].astype(float).round(0)
    q = q.sort_values('계약일자')
    m = pd.merge_asof(q, ref[['단지일련번호', '면적키', '사용가능일', '계약일자', '거래금액', '층', '직전3건_중위', '누적건수']]
                      .rename(columns={'계약일자': '직전계약일'}).sort_values('사용가능일'),
                      left_on='계약일자', right_on='사용가능일', by=['단지일련번호', '면적키'], direction='backward')
    out = pd.DataFrame({'transaction_id': m['transaction_id'],
                        '직전거래가_동일면적': m['거래금액'],
                        '직전거래_경과일': (m['계약일자'] - m['직전계약일']).dt.days,
                        '직전거래_층': m['층'], '직전3건중위_동일면적': m['직전3건_중위'], '동일면적_누적거래수': m['누적건수']})
    return out
if __name__ == '__main__':
    t = pd.read_parquet(D + 'v2_trade.parquet'); w = pd.read_parquet(D + 'apt_trade_warmup_raw.parquet')
    print(t['거래유형'].value_counts(dropna=False).to_dict(), w['거래유형'].value_counts(dropna=False).to_dict(), w['해제여부'].value_counts(dropna=False).to_dict())
    f = last_trade_features(t, w); f.to_parquet(HERE + 'lt_feat.parquet')
    x = t[['transaction_id', '계약일자', '데이터구분', '거래금액']].merge(f, on='transaction_id')
    print(x.groupby('데이터구분')[['직전거래가_동일면적']].apply(lambda s: s.notna().mean()).round(3).T)
    print(x.describe().T[['count', 'mean', '50%', 'min', 'max']].round(1))
    # 누수 점검: 경과일 30일 이상, 같은 거래 자신 제외
    assert (x['직전거래_경과일'].dropna() >= 30).all()
    print('✅ 직전 거래는 모두 계약일 30일 전 이전')
