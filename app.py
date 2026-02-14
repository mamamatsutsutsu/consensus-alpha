import os
import time
import math
import urllib.parse
import urllib.request
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AlphaLens Sovereign",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦅"
)

# Initialize Session State
if "system_logs" not in st.session_state: st.session_state.system_logs = []
if "user_access_granted" not in st.session_state: st.session_state.user_access_granted = False
if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
if "selected_stock" not in st.session_state: st.session_state.selected_stock = None

def log_system_event(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    st.session_state.system_logs.append(entry)
    # console output for cloud logs
    print(entry)

def error_boundary(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_system_event(f"{func.__name__}: {str(e)}", "ERROR")
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
            return None
    return wrapper

# ==========================================
# 1. PHANTOM UI (High Contrast & Orbitron)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Noto+Sans+JP:wght@400;700&display=swap');

:root {
  --bg: #000000;
  --panel: #0a0a0a;
  --card: #121212;
  --border: #333333;
  --accent: #00f2fe;
  --text: #e0e0e0;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; }
* { font-family: 'Orbitron', 'Noto Sans JP', sans-serif !important; letter-spacing: 0.5px !important; }

/* HIDE DEFAULTS */
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

/* BRANDING */
h1, h2, h3, .brand {
  background: linear-gradient(90deg, #fff, #00f2fe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900 !important;
  text-shadow: 0 0 20px rgba(0, 242, 254, 0.5);
}

/* CONTAINERS */
.deck { background: var(--panel); border: 1px solid var(--accent); padding: 20px; margin-bottom: 20px; box-shadow: 0 0 15px rgba(0, 242, 254, 0.1); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 15px; margin-bottom: 10px; }

/* TABLE VISIBILITY FIX */
div[data-testid="stDataFrame"] { background-color: #000 !important; border: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] * { color: #e0e0e0 !important; background-color: #000 !important; }
[data-testid="stHeader"] { background-color: #080808 !important; border-bottom: 2px solid var(--accent) !important; }
[data-testid="stHeader"] * { color: var(--accent) !important; }

/* INPUTS */
div[data-baseweb="select"] > div { background-color: #111 !important; border-color: #444 !important; color: #fff !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #000 !important; border: 1px solid #444 !important; }
div[data-baseweb="option"] { color: #fff !important; }
li[data-baseweb="option"]:hover { background-color: #222 !important; color: #00f2fe !important; }
.stSelectbox label { color: #aaa !important; }

/* BUTTONS */
button {
  background-color: #000 !important;
  color: var(--accent) !important;
  border: 1px solid #333 !important;
  border-radius: 0px !important;
  font-weight: 800 !important;
  text-transform: uppercase;
}
button:hover { border-color: var(--accent) !important; box-shadow: 0 0 15px var(--accent) !important; color: #fff !important; }

/* AI BOX */
.ai-box { border: 1px dashed var(--accent); background: rgba(0,242,254,0.05); padding: 20px; margin-top: 15px; font-size: 13px; line-height: 1.6; color: #eee; }

/* METRICS */
.kpi-val { font-size: 24px; color: var(--accent); font-weight: 700; text-shadow: 0 0 10px rgba(0,242,254,0.4); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTH & AI SETUP
# ==========================================
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

try:
    import google.generativeai as genai
    HAS_LIB = True
    if API_KEY: genai.configure(api_key=API_KEY)
except Exception as e:
    HAS_LIB = False
    log_system_event(f"GenAI Lib: {e}", "WARN")

def check_access():
    if not APP_PASS: return True
    if st.session_state.user_access_granted: return True
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br><h3 style='text-align:center'>SECURITY GATE</h3>", unsafe_allow_html=True)
        with st.form("access_form"):
            p = st.text_input("PASSCODE", type="password")
            if st.form_submit_button("UNLOCK", use_container_width=True):
                if p == APP_PASS:
                    st.session_state.user_access_granted = True
                    st.rerun()
                else: st.error("DENIED")
    return False

if not check_access(): st.stop()

# ==========================================
# 3. UNIVERSE DEFINITIONS (CORRECTED)
# ==========================================
LOOKBACKS = {"1W": 5, "1M": 21, "3M": 63, "12M": 252}
FETCH_PERIOD = "24mo"

# US SECTORS
US_SEC = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Comm Services": "XLC",
    "Cons. Disc": "XLY", "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"
}

# JP SECTORS (OFFICIAL NEXT FUNDS MAPPING 1617-1633)
JP_SEC = {
    "食品(Foods)": "1617.T",
    "エネ資源(Energy)": "1618.T",
    "建設資材(Const)": "1619.T",
    "素材化学(Mat)": "1620.T",
    "医薬品(Pharma)": "1621.T",
    "自動車(Auto)": "1622.T",
    "鉄鋼非鉄(Steel)": "1623.T",
    "機械(Machinery)": "1624.T",
    "電機精密(Elec)": "1625.T",
    "情報通信(Info)": "1626.T",
    "電力ガス(Util)": "1627.T",
    "運輸物流(Trans)": "1628.T",
    "商社卸売(Trade)": "1629.T",
    "小売(Retail)": "1630.T",
    "銀行(Bank)": "1631.T",
    "金融(Fin)": "1632.T",
    "不動産(RE)": "1633.T"
}

# STOCKS (ABBREVIATED FOR SAFETY, BUT FULL LIST LOGIC APPLIES)
US_STOCKS = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","QCOM","TXN","INTU","IBM","NOW","AMAT","MU","LRCX","ADI","KLAC","SNPS","CDNS","PANW","CRWD","ANET","PLTR"],
    "Comm Services": ["GOOGL","META","NFLX","DIS","CMCSA","TMUS","VZ","T","CHTR","WBD","LYV","EA","TTWO","OMC","IPG"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","PFE","ISRG","DHR","VRTX","GILD","REGN","BMY","CVS","CI","SYK","BSX","MDT","ZTS","HCA","MCK"],
    "Financials": ["JPM","BAC","WFC","V","MA","AXP","GS","MS","BLK","C","SCHW","SPGI","PGR","CB","MMC","KKR","BX","TRV","AFL","MET","PRU","ICE","COF"],
    "Cons. Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG","MAR","HLT","YUM","LULU","GM","F","ROST","ORLY","AZO","DHI","LEN"],
    "Cons. Staples": ["PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB","GIS","KHC","KR","STZ","EL","TGT","DG","ADM","SYY"],
    "Industrials": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","WM","NSC","CSX","GD","NOC","TDG","PCAR","FDX","CTAS"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","WMB","HAL","BKR","DVN","HES","FANG","TRGP","OKE"],
    "Materials": ["LIN","APD","SHW","FCX","ECL","NEM","DOW","DD","NUE","MLM","VMC","CTVA","PPG","ALB","CF","MOS"],
    "Utilities": ["NEE","DUK","SO","AEP","SRE","EXC","XEL","D","PEG","ED","EIX","WEC","AWK","ES","PPL","ETR"],
    "Real Estate": ["PLD","AMT","CCI","EQIX","SPG","PSA","O","WELL","DLR","AVB","EQR","VICI","CSGP","SBAC","IRM"],
}
JP_STOCKS = {
    "情報通信": ["9432.T","9433.T","9434.T","9984.T","4689.T","4755.T","9613.T","9602.T","4385.T","6098.T","3659.T","3765.T"],
    "電機・精密": ["8035.T","6857.T","6146.T","6920.T","6758.T","6501.T","6723.T","6981.T","6954.T","7741.T","6702.T","6503.T","6752.T","7735.T","6861.T"],
    "自動車・輸送": ["7203.T","7267.T","6902.T","7201.T","7269.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T","9005.T"],
    "医薬品": ["4502.T","4568.T","4519.T","4503.T","4507.T","4523.T","4578.T","4151.T","4528.T","4506.T"],
    "銀行": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T","8331.T","8354.T"],
    "金融(除銀行)": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T","8570.T"],
    "商社・小売": ["8001.T","8031.T","8058.T","8053.T","8002.T","8015.T","3382.T","9983.T","8267.T","2914.T","7453.T","3092.T"],
    "機械": ["6301.T","7011.T","7012.T","6367.T","6273.T","6113.T","6473.T","6326.T"],
    "エネルギー": ["1605.T","5020.T","9501.T","3407.T","4005.T"],
    "建設・資材": ["1925.T","1928.T","1801.T","1802.T","1812.T","5201.T","5332.T"],
    "素材・化学": ["4063.T","4452.T","4188.T","4901.T","4911.T","4021.T","4631.T","3402.T"],
    "食品": ["2801.T","2802.T","2269.T","2502.T","2503.T","2201.T","2002.T"],
    "電力・ガス": ["9501.T","9503.T","9531.T","9532.T"],
    "不動産": ["8801.T","8802.T","8830.T","3289.T","3003.T","3231.T"],
    "鉄鋼・非鉄": ["5401.T","5411.T","5713.T","5406.T","5711.T","5802.T"],
    "サービス": ["4661.T","9735.T","4324.T","2127.T","6028.T","2412.T","4689.T"],
    "産業機械": ["6146.T","6460.T","6471.T","6268.T"]
}

MARKETS = {
    "🇺🇸 US": {"bench": "SPY", "name": "S&P 500", "sectors": US_SEC, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": "1306.T", "name": "TOPIX", "sectors": JP_SEC, "stocks": JP_STOCKS},
}

# NAME DB (FULL)
NAME_DB = {
    "SPY":"S&P500","1306.T":"TOPIX","XLK":"Tech","XLV":"Health","XLF":"Financial","XLC":"Comm","XLY":"ConsDisc","XLP":"Staples","XLI":"Indust","XLE":"Energy","XLB":"Material","XLU":"Utility","XLRE":"RealEst",
    "1626.T":"情報通信","1631.T":"電機精密","1621.T":"自動車","1632.T":"医薬品","1623.T":"銀行","1624.T":"金融他","1622.T":"商社小売","1630.T":"機械","1617.T":"食品","1618.T":"エネ資源","1619.T":"建設資材","1620.T":"素材化学","1625.T":"電機精密","1627.T":"電力ガス","1628.T":"運輸物流","1629.T":"商社卸売","1633.T":"不動産",
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","GOOGL":"Alphabet","META":"Meta","AMZN":"Amazon","TSLA":"Tesla","AVGO":"Broadcom","ORCL":"Oracle","CRM":"Salesforce","ADBE":"Adobe","AMD":"AMD","QCOM":"Qualcomm","TXN":"Texas","NFLX":"Netflix","DIS":"Disney","CMCSA":"Comcast","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","JPM":"JPMorgan","BAC":"BofA","WFC":"Wells Fargo","V":"Visa","MA":"Mastercard","GS":"Goldman","MS":"Morgan Stanley","BLK":"BlackRock","C":"Citi","BRK-B":"Berkshire",
    "HD":"Home Depot","MCD":"McDonalds","NKE":"Nike","SBUX":"Starbucks","PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","XOM":"Exxon","CVX":"Chevron","GE":"GE Aero","CAT":"Caterpillar","BA":"Boeing","LMT":"Lockheed","RTX":"RTX","DE":"Deere","MMM":"3M",
    "LIN":"Linde","NEE":"NextEra","DUK":"Duke","SO":"Southern","AMT":"Amer Tower","PLD":"Prologis","INTC":"Intel","CSCO":"Cisco","IBM":"IBM","UBER":"Uber","ABNB":"Airbnb","PYPL":"PayPal",
    "8035.T":"東京エレク","6857.T":"アドバンテ","6146.T":"ディスコ","6920.T":"レーザーテク","6723.T":"ルネサス","6758.T":"ソニーG","6501.T":"日立","6981.T":"村田製","6954.T":"ファナック","7741.T":"HOYA","6702.T":"富士通","6503.T":"三菱電機","6752.T":"パナHD","7735.T":"SCREEN","6861.T":"キーエンス","6971.T":"京セラ","6645.T":"オムロン",
    "9432.T":"NTT","9433.T":"KDDI","9434.T":"ソフトバンク","9984.T":"SBG","4689.T":"LINEヤフー","6098.T":"リクルート","4755.T":"楽天G","9613.T":"NTTデータ","2413.T":"エムスリー","4385.T":"メルカリ",
    "7203.T":"トヨタ","7267.T":"ホンダ","6902.T":"デンソー","7201.T":"日産","7269.T":"スズキ","7270.T":"SUBARU","7272.T":"ヤマハ発","9101.T":"日本郵船","9104.T":"商船三井","9020.T":"JR東日本","9022.T":"JR東海","9005.T":"東急",
    "8306.T":"三菱UFJ","8316.T":"三井住友","8411.T":"みずほ","8308.T":"りそな","8309.T":"三井住友トラ","7182.T":"ゆうちょ","5831.T":"しずおかFG","8331.T":"千葉銀","8354.T":"ふくおかFG",
    "8591.T":"オリックス","8604.T":"野村HD","8766.T":"東京海上","8725.T":"MS&AD","8750.T":"第一生命","8697.T":"日本取引所","8630.T":"SOMPO","8570.T":"イオンFS",
    "8001.T":"伊藤忠","8031.T":"三井物産","8058.T":"三菱商事","8053.T":"住友商事","8002.T":"丸紅","3382.T":"7&i","9983.T":"ファストリ","8267.T":"イオン","2914.T":"JT",
    "4063.T":"信越化学","4452.T":"花王","4901.T":"富士フイルム","4911.T":"資生堂","3407.T":"旭化成","5401.T":"日本製鉄","5411.T":"JFE","6301.T":"コマツ","7011.T":"三菱重工","6367.T":"ダイキン","6273.T":"SMC",
    "1605.T":"INPEX","5020.T":"ENEOS","9501.T":"東電EP","9503.T":"関電","9531.T":"東ガス","4502.T":"武田","4568.T":"第一三共","4519.T":"中外","4503.T":"アステラス","4507.T":"塩野義","4523.T":"エーザイ",
    "8801.T":"三井不","8802.T":"三菱地所","8830.T":"住友不","4661.T":"OLC","9735.T":"セコム","4324.T":"電通","2127.T":"日本M&A","6028.T":"テクノプロ","2412.T":"ベネフィット","4689.T":"LINEヤフー",
    "6146.T":"ディスコ","6460.T":"セガサミー","6471.T":"日本精工","6268.T":"ナブテスコ","2801.T":"キッコーマン","2802.T":"味の素",
    "5711.T":"三菱マテ","5713.T":"住友鉱","5802.T":"住友電工","5406.T":"神戸鋼","3402.T":"東レ","4021.T":"日産化","4188.T":"三菱ケミ","4631.T":"DIC","3765.T":"ガンホー","3659.T":"ネクソン","2002.T":"日清製粉"
}

def get_name(t: str) -> str: return NAME_DB.get(t, t)

# ==========================================
# 4. CORE ENGINES
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_data(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t]))
    frames = []
    for i in range(0, len(tickers), 50):
        c = tickers[i:i+50]
        try:
            r = yf.download(" ".join(c), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not r.empty: frames.append(r)
        except Exception as e:
            log_system_event(f"Data Fetch: {e}", "WARN")
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

def extract_close_prices(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0): close = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(1): close = df.xs("Close", axis=1, level=1)
            else: return pd.DataFrame()
        else: return pd.DataFrame()
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        keep = [c for c in expected if c in close.columns]
        return close[keep]
    except Exception: return pd.DataFrame()

def calc_technical_metrics(s: pd.Series, b: pd.Series, win: int) -> Dict:
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # Returns for 1W, 1M, 12M
    rets = {}
    for l, d in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s) > d: rets[l] = (s.iloc[-1]/s.iloc[-1-d]-1)*100
        else: rets[l] = 0.0
        
    return {"RS": rs, "Accel": accel, "MaxDD": dd, **rets}

def get_market_summary(df_sec: pd.DataFrame, bench_ret: float) -> str:
    """Generate or fallback market summary"""
    top = df_sec.iloc[-1]["Sector"]
    bot = df_sec.iloc[0]["Sector"]
    diff = df_sec.iloc[-1]["RS"] - df_sec.iloc[0]["RS"]
    
    if HAS_LIB and API_KEY:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            p = f"市場リターン:{bench_ret:.2f}%, 先導:{top}, 遅行:{bot}, 格差:{diff:.1f}。市況を3行で要約せよ。"
            resp = model.generate_content(p)
            if resp.text: return resp.text
        except: pass
    
    return f"【市況概況】市場トレンドは{bench_ret:.1f}%。{top}セクターが牽引する一方、{bot}が軟調。セクター間格差は{diff:.1f}ポイントに拡大中。"

@st.cache_data(ttl=1800)
def get_news(ticker: str, name: str) -> Tuple[List[dict], List[dict], str]:
    y_news, g_news = [], []
    context_text = ""
    
    # Yahoo
    try:
        raw = yf.Ticker(ticker).news
        if raw:
            for n in raw[:3]:
                y_news.append({"title": n.get("title",""), "link": n.get("link","")})
                context_text += f"- {n.get('title','')}\n"
    except: pass
    
    # Google (RSS)
    try:
        q = urllib.parse.quote(f"{name} 株")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=3) as r:
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:3]:
                t = i.findtext("title")
                g_news.append({"title": t, "link": i.findtext("link")})
                context_text += f"- {t}\n"
    except: pass
    
    return y_news, g_news, context_text

def call_ai_analysis(ticker: str, name: str, stats: Dict, news_context: str) -> str:
    if not HAS_LIB or not API_KEY:
        return "⚠️ AI OFFLINE: キー設定を確認してください。"

    # STRICT MODEL SELECTION (2.0-flash/lite confirmed)
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
    
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            prompt = f"""
            あなたはプロのファンドマネージャーです。
            以下の定量的・定性的データに基づき、3名のエージェント（モメンタム、リスク、マクロ）として議論し、結論を出してください。
            
            【対象】{name} ({ticker})
            【定量データ】RS:{stats['RS']:.2f}%, 加速:{stats['Accel']:.2f}, 1年騰落:{stats.get('12M',0):.1f}%
            【ニュース材料】
            {news_context}
            
            出力形式:
            【モメンタム】...
            【リスク】...
            【マクロ】...
            【結論】(強気/中立/弱気) 理由
            """
            return model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e):
                time.sleep(2) # Retry logic
                continue
            if "404" in str(e): continue
            
    # Fallback Rule-Based Debate
    v = "強気" if stats['RS']>0 and stats['Accel']>0 else "中立"
    return f"""
    【モメンタム】RS{stats['RS']:.2f}%と{'加速' if stats['Accel']>0 else '減速'}傾向を確認。
    【リスク】ニュース材料（{news_context[:20]}...）に注意が必要。
    【結論】{v} (AI接続不可のためルールベース生成)
    """

# ==========================================
# 5. MAIN UI
# ==========================================
@error_boundary
def main():
    st.markdown("<h2 class='brand'>ALPHALENS SOVEREIGN</h2>", unsafe_allow_html=True)
    
    # Sidebar Logs
    with st.sidebar:
        st.markdown("### SYSTEM LOGS")
        if st.session_state.system_logs:
            for l in st.session_state.system_logs[-5:]:
                st.markdown(f"<div style='font-size:10px; color:#ff6b6b; border-left:2px solid red; padding-left:5px;'>{l}</div>", unsafe_allow_html=True)
        if st.button("CLEAR LOGS"): st.session_state.system_logs = []; st.rerun()

    # Controls
    with st.container():
        st.markdown("<div class='deck'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
        with c1: market_key = st.selectbox("MARKET", list(MARKETS.keys()))
        with c2: lookback_key = st.selectbox("WINDOW", list(LOOKBACKS.keys()), index=1)
        with c3: st.caption(f"FETCH: {FETCH_PERIOD}"); st.progress(100)
        with c4: 
            st.write("")
            sync = st.button("SYNC", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # 1. Sync
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if sync or "core_df" not in st.session_state or st.session_state.get("last_m") != market_key:
        with st.spinner("SYNCING MARKET DATA..."):
            raw = fetch_market_data(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close_prices(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    audit = audit_data_availability(core_tickers, core_df, win)
    
    if bench not in audit["list"]:
        st.error("DATA FEED DISCONNECTED")
        return

    # 2. Sector Overview
    b_stats = calc_technical_metrics(core_df[bench], core_df[bench], win)
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit["list"]:
            res = calc_technical_metrics(core_df[s_t], core_df[bench], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
    
    # Market Summary
    st.info(get_market_summary(sdf, b_stats["Ret"]))

    # Chart
    st.subheader("SECTOR ROTATION")
    fig = px.bar(sdf, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn")
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', font_family="Orbitron")
    
    # SAFE SELECTION LOGIC
    chart_key = "sec_chart"
    st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
    
    # Try getting selection safely
    click_sec = None
    try:
        sel = st.session_state.get(chart_key)
        if sel and "selection" in sel and "points" in sel["selection"] and sel["selection"]["points"]:
            click_sec = sel["selection"]["points"][0]["y"]
    except: pass

    # Fallback Buttons
    cols = st.columns(6)
    for i, s in enumerate(m_cfg["sectors"].keys()):
        if cols[i%6].button(s, key=f"btn_{s}", use_container_width=True):
            click_sec = s
            
    target_sector = click_sec or st.session_state.selected_sector or list(m_cfg["sectors"].keys())[0]
    st.session_state.selected_sector = target_sector
    
    # 3. Drill Down
    st.markdown("---")
    st.subheader(f"FORENSIC: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    cache_key = f"{market_key}_{target_sector}"
    if cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner("SCANNING SECTOR ASSETS..."):
            raw_s = fetch_market_data(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close_prices(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_data_availability(full_list, sec_df, win)
    
    results = []
    for t in [x for x in s_audit["list"] if x != bench]:
        stats = calc_technical_metrics(sec_df[t], sec_df[bench], win)
        if stats:
            stats["Ticker"] = t
            stats["Name"] = get_name(t)
            results.append(stats)
            
    df = pd.DataFrame(results).sort_values("RS", ascending=False)
    df["Apex"] = 0.6 * df["RS"] + 0.4 * df["Accel"] # Simple scoring
    df = df.sort_values("Apex", ascending=False)

    # 4. Table & Detail
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("##### LEADERBOARD")
        event = st.dataframe(
            df[["Name", "RS", "Accel", "1W", "1M", "12M"]],
            column_config={
                "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
                "Accel": st.column_config.NumberColumn(format="%.2f"),
                "1W": st.column_config.NumberColumn(format="%.1f%%"),
                "1M": st.column_config.NumberColumn(format="%.1f%%"),
                "12M": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table"
        )
        
        # Safe Selection for Table
        try:
            sel_rows = event.selection.get("rows", [])
            top = df.iloc[sel_rows[0]] if sel_rows else df.iloc[0]
        except:
            top = df.iloc[0]

    with c2:
        st.markdown(f"##### AI INTELLIGENCE: {top['Name']}")
        yn, gn, context = get_news(top["Ticker"], top["Name"])
        ai_txt = call_ai_analysis(top["Ticker"], top["Name"], top.to_dict(), context)
        st.markdown(f"<div class='ai-box'>{ai_txt}</div>", unsafe_allow_html=True)
        
        st.caption("LATEST HEADLINES")
        for n in (yn + gn)[:4]:
            st.markdown(f"- [{n['title']}]({n['link']})")

if __name__ == "__main__":
    main()