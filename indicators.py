import numpy as np
import pandas as pd
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
import streamlit as st
import ssl
from urllib import request
#python -m streamlit run app.py

def Hisse_Temel_Veriler():
    url1="https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/Temel-Degerler-Ve-Oranlar.aspx#page-1"
    context = ssl._create_unverified_context()
    response = request.urlopen(url1, context=context)
    url1 = response.read()
    df = pd.read_html(url1,decimal=',', thousands='.')                         #Tüm Hisselerin Tablolarını Aktar
    df1=df[2]                                                                  #Tüm Hisselerin Özet Tablosu                                   
    return df1   

tv = TvDatafeed()


def TillsonT3(Close, high, low, vf, length):
    ema_first_input = (high + low + 2 * Close) / 4
    e1 = ta.ema(ema_first_input, length)
    e2 = ta.ema(e1, length)
    e3 = ta.ema(e2, length)
    e4 = ta.ema(e3, length)
    e5 = ta.ema(e4, length)
    e6 = ta.ema(e5, length)

    c1 = -1 * vf * vf * vf
    c2 = 3 * vf * vf + 3 * vf * vf * vf
    c3 = -6 * vf * vf - 3 * vf - 3 * vf * vf * vf
    c4 = 1 + 3 * vf + vf * vf * vf + 3 * vf * vf
    T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return T3

def OTT(df,prt,prc):
    pds = prt
    percent = prc
    alpha = 2 / (pds + 1)

    df['ud1'] = np.where(df['Close'] > df['Close'].shift(1), (df['Close'] - df['Close'].shift()) , 0)
    df['dd1'] = np.where(df['Close'] < df['Close'].shift(1), (df['Close'].shift() - df['Close']) , 0)
    df['UD'] = df['ud1'].rolling(9).sum()
    df['DD'] = df['dd1'].rolling(9).sum()
    df['CMO'] = ((df['UD'] - df['DD']) / (df['UD'] + df['DD'])).fillna(0).abs()

    df['Var'] = 0.0
    for i in range(pds, len(df)):
        df['Var'].iat[i] = (alpha * df['CMO'].iat[i] * df['Close'].iat[i]) + (1 - alpha * df['CMO'].iat[i]) * df['Var'].iat[i-1]

    df['fark'] = df['Var'] * percent * 0.01
    df['newlongstop'] = df['Var'] - df['fark']
    df['newshortstop'] = df['Var'] + df['fark']
    df['longstop'] = 0.0
    df['shortstop'] = 999999999999999999
    # df['dir'] = 1
    for i in df['UD']:

        def maxlongstop():
            df.loc[(df['newlongstop'] > df['longstop'].shift(1)) , 'longstop'] = df['newlongstop']
            df.loc[(df['longstop'].shift(1) > df['newlongstop']), 'longstop'] = df['longstop'].shift(1) 

            return df['longstop']

        def minshortstop():
            df.loc[(df['newshortstop'] < df['shortstop'].shift(1)), 'shortstop'] = df['newshortstop']
            df.loc[(df['shortstop'].shift(1) < df['newshortstop']), 'shortstop'] = df['shortstop'].shift(1)

            return df['shortstop']

        df['longstop']= np.where (((df['Var'] > df['longstop'].shift(1))),maxlongstop(),df['newlongstop'])
        df['shortstop'] = np.where(((df['Var'] < df['shortstop'].shift(1))), minshortstop(), df['newshortstop'])

    #get xover

    df['xlongstop'] = np.where (((df['Var'].shift(1) > df['longstop'].shift(1)) & (df['Var'] < df['longstop'].shift(1))), 1,0)
    df['xshortstop'] =np.where(((df['Var'].shift(1) < df['shortstop'].shift(1)) & (df['Var'] > df['shortstop'].shift(1))), 1,0)

    df['trend']=0
    df['dir'] = 0

    for i in df['UD']:
            df['trend'] = np.where(((df['xshortstop'] == 1)),1, (np.where((df['xlongstop'] == 1),-1,df['trend'].shift(1))))
            df['dir'] = np.where(((df['xshortstop'] == 1)),1, (np.where((df['xlongstop'] == 1),-1,df['dir'].shift(1).fillna(1))))


    df['MT'] = np.where(df['dir'] == 1, df['longstop'], df['shortstop'])
    df['OTT'] = np.where(df['Var'] > df['MT'], (df['MT'] * (200 + percent) / 200), (df['MT'] * (200 - percent) / 200))
    # round the numeric columns
    df = df.round(2)
    
    #this OTT2 column now is to be shifted by 2 prev values
    df['OTT2'] = df['OTT'].shift(2)
    df['OTT3'] = df['OTT'].shift(3)
    
    return df

def indicator_Signals(Hisse_Adı,Lenght_1,vf,prt,prc):
    data = tv.get_hist(symbol=Hisse_Adı,exchange='BIST',interval=Interval.in_daily,n_bars=500)
    data.rename(columns = {'open':'Open', 'high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace = True)
    #data = yf.download(Hisse_Adı+'.IS',start='2023-01-01',interval='1d',progress=False)
    OTT_Signal=OTT(data.copy(deep=True),prt,prc)
    Tillson=TillsonT3(data['Close'],data['High'],data['Low'],vf,Lenght_1)
    Zscore=ta.zscore(data['Close'],21,1)
    Zsma=ta.sma(Zscore)
    data['OTT']=OTT_Signal['OTT3']
    data['Var']=OTT_Signal['Var']
    data['Tillson']=Tillson
    data['Zscore']=Zscore
    data['ZSMA']=Zsma

    #True Condition
    data['OTT_Signal']=(data['Var'])>OTT_Signal['OTT3']
    data['Zscore_Signal']=data['Zscore']>0.85
    #True Condition
    data['Entry']=(data['OTT_Signal'] & data['Zscore_Signal']) 
    data['Exit']=False
    for i in range(1,len(data['Entry'])-1):
        t3_prev = data['Tillson'][i-2]
        t3_mid = data['Tillson'][i-1]
        t3_fwd = data['Tillson'][i]
        if t3_mid > t3_prev and t3_mid < t3_fwd:
            data['Exit'][i]=True
    return data

st.set_page_config(
    page_title="Hisse Sinyalleri",
    layout="wide",
    initial_sidebar_state="expanded")
with st.sidebar:
    Hisse_Ozet=Hisse_Temel_Veriler()
    st.header('Hisse Arama')
    Hisse_Adı = st.selectbox('Hisse Adı',Hisse_Ozet['Kod'])
    Lenght_1=6
    vf = 0.8
    prt=2
    prc=1.2
    data=indicator_Signals(Hisse_Adı,Lenght_1,vf,prt,prc)

Son_Durum=data.tail(1)
col1, col2, col3, col4, col5 = st.columns(5)
Close=Son_Durum['Close'].iloc[0]
OTT_Signal=Son_Durum['OTT_Signal'].iloc[0]
if OTT_Signal==False:
    OTT_Signal='Bekle'
if OTT_Signal==True:
    OTT_Signal='Alınabilir' 

Zscore_Signal=Son_Durum['Zscore_Signal'].iloc[0]
if Zscore_Signal==False:
    Zscore_Signal='Bekle'
if Zscore_Signal==True:
    Zscore_Signal='Alınabilir' 

Tillson_Sinyal=Son_Durum['Exit'].iloc[0]
if Tillson_Sinyal==True:
    Tillson_Sinyal='Satılabilir'
if Tillson_Sinyal==False:
    Tillson_Sinyal='Bekle'

col2.metric('Kapanış Fiyatı',str(Close))
col3.metric('OTT Sinyal',str(OTT_Signal))
col4.metric('Z Skor Sinyal',str(Zscore_Signal))
col5.metric('Tillson Sinyal',str(Tillson_Sinyal))


st.dataframe(data.iloc[::-1],use_container_width=True)
