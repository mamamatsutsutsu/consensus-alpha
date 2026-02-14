import os
import time
import re
import math
import urllib.parse
import urllib.request
import traceback
import xml.etree.ElementTree as ET
import email.utils
from datetime import datetime, timedelta
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
    page_title="AlphaLens",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦅"
)

# Initialize Session State
if "system_logs" not in st.session_state: st.session_state.system_logs = []
if "user_access_granted" not in st.session_state: st.session_state.user_access_granted = False
if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
if "last_market_key" not in st.session_state: st.session_state.last_market_key = None

def log_system_event(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {msg}"
    st.session_state.system_logs.append(entry)
    print(entry)

def error_boundary(func):
    """Decorator to catch errors without crashing the app"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_system_event(f"{func.__name__}: {str(e)}", "ERROR")
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
            return None
    return wrapper

# ==========================================
# 1. PHANTOM UI (Professional, Cyberpunk)
# ==========================================
st.markdown("""
<style>
/* FONTS */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Noto+Sans+JP:wght@400;700&display=swap');

:root {
  --bg: #000000;
  --panel: #0a0a0a;
  --card: #121212;
  --border: #333333;
  --accent: #00f2fe;     /* Cyan */
  --accent-2: #ff0055;   /* Pink/Red */
  --accent-3: #00ff88;   /* Green */
  --accent-4: #ffcc00;   /* Yellow */
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
  margin-bottom: 0px !important;
  padding-bottom: 10px;
}

/* CONTAINERS */
.deck { background: var(--panel); border-bottom: 1px solid var(--accent); padding: 15px; margin-bottom: 20px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 20px; margin-bottom: 15px; }

/* TABLE VISIBILITY FIX */
div[data-testid="stDataFrame"] {
    background-color: #151515 !important;
    border: 1px solid var(--border) !important;
}
div[data-testid="stDataFrame"] * {
    color: #ffffff !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stHeader"] {
    background-color: #222 !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* INPUTS */
div[data-baseweb="select"] > div { background-color: #111 !important; border-color: #444 !important; color: #fff !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #111 !important; border: 1px solid #444 !important; }
div[data-baseweb="option"] { color: #fff !important; }
li[data-baseweb="option"]:hover { background-color: #222 !important; color: #00f2fe !important; }
.stSelectbox label { color: #aaa !important; }

/* BUTTONS */
button {
  background-color: #111 !important;
  color: var(--accent) !important;
  border: 1px solid #333 !important;
  border-radius: 4px !important;
  font-weight: 700 !important;
  text-transform: uppercase;
}
button:hover { border-color: var(--accent) !important; box-shadow: 0 0 15px var(--accent) !important; color: #fff !important; }

/* AI AGENT BOXES */
.ai-box { border: 1px dashed var(--accent); background: rgba(0,242,254,0.05); padding: 20px; margin: 15px 0; border-radius: 8px; color: #eee; }

.agent-box {
    padding: 15px; margin-bottom: 10px; border-radius: 6px; font-size: 13px; line-height: 1.6; border-left-width: 4px; border-left-style: solid;
}
.agent-fundamental { border-left-color: #00f2fe; background: rgba(0, 242, 254, 0.05); }
.agent-sentiment { border-left-color: #ff0055; background: rgba(255, 0, 85, 0.05); }
.agent-valuation { border-left-color: #00ff88; background: rgba(0, 255, 136, 0.05); }
.agent-skeptic { border-left-color: #ffcc00; background: rgba(255, 204, 0, 0.05); }
.agent-risk { border-left-color: #888888; background: rgba(136, 136, 136, 0.1); }
.agent-verdict { border: 1px solid #ffffff; background: #1a1a1a; font-weight: bold; margin-top: 15px; padding: 20px; }

/* LOGS */
.log-box { font-family: monospace; font-size: 10px; color: #ff6b6b; background: #1a0505; padding: 5px; border-left: 3px solid #ff6b6b; margin-bottom: 2px; }

/* MARKET SUMMARY */
.market-box {
    border-left: 5px solid var(--accent);
    background: #0f0f0f;
    padding: 20px;
    margin-bottom: 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #ddd;
}

/* REPORT BOX */
.report-box {
    background: #111;
    border-top: 3px solid var(--accent);
    padding: 20px;
    margin-top: 10px;
    line-height: 1.8;
    color: #eee;
}

/* METRICS */
.kpi-val { font-size: 20px; color: var(--accent); font-weight: 700; }
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
# 3. UNIVERSE DEFINITIONS
# ==========================================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

US_SEC = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Comm Services": "XLC",
    "Cons. Disc": "XLY", "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"
}

JP_SEC = {
    "食品(Foods)": "1617.T", "エネルギー(Energy)": "1618.T", "建設・資材(Const)": "1619.T", 
    "素材・化学(Mat)": "1620.T", "医薬品(Pharma)": "1621.T", "自動車・輸送(Auto)": "1622.T", 
    "鉄鋼・非鉄(Steel)": "1623.T", "機械(Machinery)": "1624.T", "電機・精密(Elec)": "1625.T", 
    "情報通信(Info)": "1626.T", "電力・ガス(Util)": "1627.T", "運輸・物流(Trans)": "1628.T", 
    "商社・卸売(Trade)": "1629.T", "小売(Retail)": "1630.T", "銀行(Bank)": "1631.T", 
    "金融(Fin)": "1632.T", "不動産(RE)": "1633.T"
}

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
    "情報通信(Info)": ["9432.T","9433.T","9434.T","9984.T","4689.T","4755.T","9613.T","9602.T","4385.T","6098.T","3659.T","3765.T"],
    "電機・精密(Elec)": ["8035.T","6857.T","6146.T","6920.T","6758.T","6501.T","6723.T","6981.T","6954.T","7741.T","6702.T","6503.T","6752.T","7735.T","6861.T"],
    "自動車・輸送(Auto)": ["7203.T","7267.T","6902.T","7201.T","7269.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T","9005.T"],
    "医薬品(Pharma)": ["4502.T","4568.T","4519.T","4503.T","4507.T","4523.T","4578.T","4151.T","4528.T","4506.T"],
    "銀行(Bank)": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T","8331.T","8354.T"],
    "金融(Fin)": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T","8570.T"],
    "商社・卸売(Trade)": ["8001.T","8031.T","8058.T","8053.T","8002.T","8015.T","3382.T","9983.T","8267.T","2914.T","7453.T","3092.T"], 
    "機械(Machinery)": ["6301.T","7011.T","7012.T","6367.T","6273.T","6113.T","6473.T","6326.T"],
    "エネルギー(Energy)": ["1605.T","5020.T","9501.T","3407.T","4005.T"],
    "建設・資材(Const)": ["1925.T","1928.T","1801.T","1802.T","1812.T","5201.T","5332.T"],
    "素材・化学(Mat)": ["4063.T","4452.T","4188.T","4901.T","4911.T","4021.T","4631.T","3402.T"],
    "食品(Foods)": ["2801.T","2802.T","2269.T","2502.T","2503.T","2201.T","2002.T"],
    "電力・ガス(Util)": ["9501.T","9503.T","9531.T","9532.T"],
    "不動産(RE)": ["8801.T","8802.T","8830.T","3289.T","3003.T","3231.T"],
    "鉄鋼・非鉄(Steel)": ["5401.T","5411.T","5713.T","5406.T","5711.T","5802.T"],
    "小売(Retail)": ["3382.T", "8267.T", "9983.T", "3092.T", "7453.T"], 
    "運輸・物流(Trans)": ["9101.T", "9104.T", "9020.T", "9021.T", "9022.T"] 
}

MARKETS = {
    "🇺🇸 US": {"bench": "SPY", "name": "S&P 500", "sectors": US_SEC, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": "1306.T", "name": "TOPIX", "sectors": JP_SEC, "stocks": JP_STOCKS},
}

# FULL NAME DB
NAME_DB = {
    "SPY":"S&P500","1306.T":"TOPIX","XLK":"Tech","XLV":"Health","XLF":"Fin","XLC":"Comm","XLY":"ConsDisc","XLP":"Staples","XLI":"Indust","XLE":"Energy","XLB":"Material","XLU":"Utility","XLRE":"RealEst",
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
    chunk = 40 
    for i in range(0, len(tickers), chunk):
        c = tickers[i:i+chunk]
        try:
            r = yf.download(" ".join(c), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not r.empty: frames.append(r)
        except Exception as e:
            log_system_event(f"Fetch Chunk Error: {e}", "WARN")
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
        cols = [c for c in expected if c in close.columns]
        return close[cols]
    except Exception as e:
        log_system_event(f"Extract Close Error: {e}", "ERROR")
        return pd.DataFrame()

def calc_technical_metrics(s: pd.Series, b: pd.Series, win: int) -> Dict:
    # Strict Length Check (DROPNA first)
    s_clean = s.dropna()
    b_clean = b.dropna()
    
    if len(s_clean) < win + 1 or len(b_clean) < win + 1: return None
    
    # 2. Fill NaN with ffill ONLY (No Bfill!)
    s_win = s.ffill().tail(win+1)
    b_win = b.ffill().tail(win+1)
    
    # 3. Final check
    if s_win.isna().iloc[0] or b_win.isna().iloc[0]: return None

    # Use Filled data for calculation to avoid NaN blowout
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # 52W High Ratio
    if len(s_clean) >= 252:
        year_high = s_clean.tail(252).max()
    else:
        year_high = s_clean.max()
        
    curr = s_win.iloc[-1]
    high_dist = (curr / year_high - 1) * 100 if year_high > 0 else 0
    
    # Multi-Horizon with NaN
    rets = {}
    for l, d in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s_win) > d:
            try:
                rets[l] = (s_win.iloc[-1] / s_win.iloc[-1-d] - 1) * 100
            except: rets[l] = np.nan
        else:
            rets[l] = np.nan
    
    return {"RS": rs, "Accel": accel, "MaxDD": dd, "Ret": p_ret, "HighDist": high_dist, **rets}

def calculate_regime(bench_series: pd.Series) -> Tuple[str, float]:
    if len(bench_series) < 200: return "Unknown", 0.5
    
    curr = bench_series.iloc[-1]
    ma200 = bench_series.rolling(200).mean().iloc[-1]
    vol20 = bench_series.pct_change().tail(20).std() * np.sqrt(252)
    
    trend = "Bull" if curr > ma200 else "Bear"
    vol_state = "High" if vol20 > 0.15 else "Low" # 15% threshold
    regime = f"{trend} / {vol_state} Vol"
    weight_momentum = 0.6 if trend == "Bull" else 0.3
    return regime, weight_momentum

def audit_data_availability(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "list": []}
    
    last_valid = df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_valid.mode().iloc[0] if not last_valid.empty else None
    
    computable = []
    for t in present:
        if last_valid[t] == mode_date and len(df[t].dropna()) >= win + 1:
            computable.append(t)
            
    return {"ok": True, "list": computable, "mode": mode_date, "count": len(computable), "total": len(expected)}

def calculate_zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# --- AI & NEWS (ROBUST & SORTED) ---

@st.cache_data(ttl=1800)
def get_news_consolidated(ticker: str, name: str, limit_each: int = 10) -> Tuple[List[dict], str]:
    news_items = []
    context_lines = []

    # Yahoo
    try:
        raw = yf.Ticker(ticker).news or []
        for n in raw[:limit_each]:
            title = n.get("title","")
            link = n.get("link","")
            pub = n.get("providerPublishTime")
            pub = int(pub) if isinstance(pub, (int, float)) else 0
            news_items.append({"title": title, "link": link, "pub": pub, "src": "Yahoo"})
            if title:
                context_lines.append(f"- {title}")
    except:
        pass

    # Google RSS
    try:
        q = urllib.parse.quote(f"{name} 株")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=3) as r:
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:limit_each]:
                title = i.findtext("title") or ""
                link = i.findtext("link") or ""
                d = i.findtext("pubDate") or ""
                pub = 0
                try:
                    dt = email.utils.parsedate_to_datetime(d)
                    pub = int(dt.timestamp())
                except:
                    pub = 0
                news_items.append({"title": title, "link": link, "pub": pub, "src": "Google"})
                if title:
                    context_lines.append(f"- {title}")
    except:
        pass

    # Sort descending by publish timestamp
    news_items.sort(key=lambda x: x["pub"], reverse=True)
    context = "\n".join(context_lines[:15]) 
    return news_items, context

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker: str) -> Dict[str, Any]:
    """Fetch fundamental data safely"""
    try:
        info = yf.Ticker(ticker).info
        return {
            "forwardPE": info.get("forwardPE", "N/A"),
            "trailingPE": info.get("trailingPE", "N/A"),
            "priceToBook": info.get("priceToBook", "N/A"),
            "pegRatio": info.get("pegRatio", "N/A"),
            "targetMeanPrice": info.get("targetMeanPrice", "N/A"),
            "recommendationKey": info.get("recommendationKey", "N/A")
        }
    except:
        return {}

@st.cache_data(ttl=3600)
def generate_ai_content(prompt_key: str, context: Dict) -> str:
    if not HAS_LIB or not API_KEY: return "⚠️ AI OFFLINE"
    
    if prompt_key == "market":
        p = f"""
        期間:{context['s_date']}〜{context['e_date']}
        市場:{context['market_name']}
        ベンチリターン:{context['ret']:.2f}%
        セクター最強:{context['top']} セクター最弱:{context['bot']}
        素材(見出し):{context['headlines']}

        タスク:この期間の市場概況を、読み手が「これだけで動きが納得できる」ように完結させて書け。
        要件:
        - 450〜650字。段落の間に空行を入れない。
        - 数値(ベンチリターン/最強最弱)と理由(材料)を必ず結びつける。
        - 理由は「何が起きた→どの資産/セクターに資金が動いた→指数に+/-寄与」の順で因果を書く。
        - 断定しすぎず、材料が弱い場合は「可能性が高い/示唆」と表現してよいが、理由そのものは必ず書く。
        - 「ニュースでは、」「〜がありました」等の無駄な前置きは禁止。いきなり本題から。
        最後に【主な変動要因】として3〜6個の箇条書き。各行は必ず「(+)/(−)」で符号を付け、何がどう効いたかを具体的に(例:金利上昇→グロース逆風→テック軟化(−))。
        """
    elif prompt_key == "sector_debate":
        p = f"""
        あなたは5名の専門エージェント(Fundamental, Sentiment, Valuation, Skeptic, Risk)。
        対象セクター:{context['sec']}
        対象期間(見通し):今後3ヶ月(短期)
        構成銘柄数:{context['count']}
        追加情報(上位銘柄/統計):
        - 上位5銘柄:{context.get('top5','')}
        - セクター平均RS:{context.get('avg_rs','')}

        タスク:このセクター内で「短期(3ヶ月)の推奨」を作る。個別企業の深掘りではなく、セクター内の相対推奨(どのタイプ/条件の銘柄が有利か、上位候補は誰か)を結論づける。
        出力形式(必須):
        [FUNDAMENTAL] 3ヶ月で効きやすい業績/需給/マクロの論点→プラス/マイナス
        [SENTIMENT] センチメント/ポジショニング/ニュースフローの方向→プラス/マイナス
        [VALUATION] バリュエーション観点(割安/割高ではなく「短期の再評価余地」)→プラス/マイナス
        [SKEPTIC] 反論:なぜその見方が外れ得るか、逆回転条件
        [RISK] 3ヶ月で起きやすいリスク(イベント、金利、為替、規制、決算集中など)と回避策
        [JUDGE] セクター推奨度(強気/中立/弱気)、短期で優位な“条件”、注目銘柄があるなら最大3つ(理由を一行ずつ)
        """
    elif prompt_key == "stock_report":
        # Inject Fundamental Data
        fund = context.get("fundamentals", {})
        fund_str = f"PER(Fwd):{fund.get('forwardPE')}, PBR:{fund.get('priceToBook')}, PEG:{fund.get('pegRatio')}, Target:{fund.get('targetMeanPrice')}"
        
        p = f"""
        銘柄: {context['name']} ({context['ticker']})
        基礎データ: {fund_str}
        
        上記データと一般的な知識に基づき、企業概要、直近決算のポイント、市場コンセンサスを、
        プロのアナリストレポートとしてまとめよ。挨拶不要。
        数値データがある場合は必ず言及し、割安/割高の判断材料とせよ。
        """
    else: return "Error"

    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            text = model.generate_content(p).text
            # Clean up empty lines
            return re.sub(r"\n{2,}", "\n", text).strip()
        except Exception as e:
            if "429" in str(e): time.sleep(1); continue
            
    return "AI Unavailable"

def parse_agent_debate(text: str) -> str:
    html = ""
    mapping = {
        "[FUNDAMENTAL]": "agent-fundamental",
        "[SENTIMENT]": "agent-sentiment",
        "[VALUATION]": "agent-valuation",
        "[SKEPTIC]": "agent-skeptic",
        "[RISK]": "agent-risk",
        "[JUDGE]": "agent-verdict"
    }
    
    # Remove markdown code blocks
    clean_text = text.replace("```html", "").replace("```", "")
    lines = clean_text.split("\n")
    buffer = []
    current_cls = "agent-box"
    
    for line in lines:
        tag_found = False
        for tag, cls in mapping.items():
            if tag in line:
                if buffer:
                    html += f"<div class='agent-box {current_cls}'>{'<br>'.join(buffer)}</div>"
                    buffer = []
                current_cls = cls
                line = line.replace(tag, f"<b>{tag.replace('[','').replace(']','')}</b>")
                tag_found = True
                break
        buffer.append(line)
    
    if buffer:
        html += f"<div class='agent-box {current_cls}'>{'<br>'.join(buffer)}</div>"
    
    if not html: # Fallback if no tags
        html = f"<div class='agent-box'>{clean_text}</div>"
        
    return html

# ==========================================
# 5. MAIN UI LOGIC
# ==========================================
@error_boundary
def main():
    st.markdown("<h1 class='brand'>ALPHALENS</h1>", unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### SYSTEM LOGS")
        if st.session_state.system_logs:
            for l in st.session_state.system_logs[-5:]:
                st.markdown(f"<div class='log-box'>{l}</div>", unsafe_allow_html=True)
        if st.button("CLEAR LOGS"): st.session_state.system_logs = []; st.rerun()

    # --- Header & Controls ---
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
    with c1: market_key = st.selectbox("MARKET", list(MARKETS.keys()))
    with c2: lookback_key = st.selectbox("WINDOW", list(LOOKBACKS.keys()), index=1)
    with c3: st.caption(f"FETCH: {FETCH_PERIOD}"); st.progress(100)
    with c4: 
        st.write("")
        sync = st.button("SYNC", type="primary", use_container_width=True)

    if st.session_state.last_market_key != market_key:
        st.session_state.selected_sector = None
        st.session_state.last_market_key = market_key

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # --- Data Fetching ---
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if sync or "core_df" not in st.session_state:
        with st.spinner("SYNCING MARKET DATA..."):
            raw = fetch_market_data(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close_prices(raw, core_tickers)
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    if core_df.empty or len(core_df) < win + 1:
        st.warning("WAITING FOR DATA SYNC...")
        return

    audit = audit_data_availability(core_tickers, core_df, win)
    if bench not in audit["list"]:
        st.error("BENCHMARK MISSING")
        return

    # --- 1. Market Overview ---
    b_stats = calc_technical_metrics(core_df[bench], core_df[bench], win)
    if not b_stats:
        st.error("BENCH METRICS FAILED. PLEASE SYNC AGAIN.")
        return

    regime, weight_mom = calculate_regime(core_df[bench].dropna())
    
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit["list"]:
            res = calc_technical_metrics(core_df[s_t], core_df[bench], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    if not sec_rows:
        st.warning("SECTOR ETF DATA INSUFFICIENT. Showing market only.")
        top_sec, bot_sec = "N/A", "N/A"
        sdf = pd.DataFrame([{"Sector":"N/A","RS":0.0}])
    else:
        sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
        top_sec, bot_sec = sdf.iloc[-1]["Sector"], sdf.iloc[0]["Sector"]
    
    # Market AI Summary
    s_date = core_df.index[-win-1].strftime('%Y/%m/%d')
    e_date = core_df.index[-1].strftime('%Y/%m/%d')
    _, market_context = get_news_consolidated(bench, m_cfg["name"])
    
    market_text = generate_ai_content("market", {
        "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
        "top": top_sec, "bot": bot_sec,
        "market_name": m_cfg["name"],
        "headlines": market_context
    })
    
    st.markdown(f"""
    <div class='market-box'>
    <b>MARKET PULSE ({s_date} - {e_date}) | REGIME: {regime}</b><br>
    {market_text}
    </div>
    """, unsafe_allow_html=True)

    # --- 2. Sector Rotation (Correct Sync) ---
    st.subheader(f"SECTOR ROTATION ({s_date} - {e_date})")
    
    # (1) Read Chart Selection First
    click_sec = None
    sel = st.session_state.get("sector_chart", None)
    try:
        # dict case
        if isinstance(sel, dict) and sel.get("selection", {}).get("points"):
            click_sec = sel["selection"]["points"][0].get("y")
        # object case
        elif sel and hasattr(sel, "selection") and sel.selection and sel.selection.get("points"):
            click_sec = sel.selection["points"][0].get("y")
    except: pass

    if click_sec: st.session_state.selected_sector = click_sec

    # (2) Build Colors based on State
    selected = st.session_state.selected_sector
    colors = ["#333"] * len(sdf)
    if selected and selected in sdf["Sector"].values:
        pos = sdf.index.get_loc(sdf[sdf["Sector"] == selected].index[0])
        colors[pos] = "#00f2fe"

    fig = px.bar(sdf, x="RS", y="Sector", orientation='h', title=f"Relative Strength ({lookback_key})")
    fig.update_traces(marker_color=colors)
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', font_family="Orbitron")
    
    st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sector_chart")
    
    # (3) Responsive Buttons with Checkmark
    st.write("SELECT SECTOR:")
    cols = st.columns(2)
    valid_sectors = set(m_cfg["sectors"].keys())
    
    for i, s in enumerate(m_cfg["sectors"].keys()):
        label = f"✅ {s}" if s == st.session_state.selected_sector else s
        if cols[i%2].button(label, key=f"btn_{s}", use_container_width=True):
            st.session_state.selected_sector = s
            st.rerun()
            
    # Validate Sector
    if st.session_state.selected_sector not in valid_sectors:
        st.session_state.selected_sector = list(m_cfg["sectors"].keys())[0] if valid_sectors else None
        
    target_sector = st.session_state.selected_sector
    
    # Navigation Link
    if target_sector:
        st.caption(f"Current: **{target_sector}** → [Jump to Analysis](#sector_anchor)")
        if click_sec: st.toast(f"Sector Selected: {target_sector}", icon="🦅")

    # --- 3. Sector Forensic ---
    st.markdown(f"<div id='sector_anchor'></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader(f"SECTOR FORENSIC: {target_sector}")
    
    if not target_sector: return

    stock_list = m_cfg["stocks"].get(target_sector, [])
    if not stock_list:
        st.warning(f"No stocks mapped for {target_sector}. Check configuration.")
        return

    full_list = [bench] + stock_list
    
    cache_key = f"{market_key}_{target_sector}_{lookback_key}"
    if cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner(f"ANALYZING {target_sector}..."):
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
            
    if not results:
        st.warning("NO VALID DATA FOR SECTOR STOCKS.")
        return
        
    df = pd.DataFrame(results)
    # Dynamic Scoring
    w_rs = weight_mom
    w_acc = 0.8 - w_rs
    df["Apex"] = w_rs * calculate_zscore(df["RS"]) + w_acc * calculate_zscore(df["Accel"]) + 0.2 * calculate_zscore(df["Ret"])
    
    # Confidence Score
    df["Conf"] = 80 + (calculate_zscore(df["Apex"]).abs() * 5).clip(0, 15)
    
    df = df.sort_values("Apex", ascending=False)
    
    # --- 4. 5-AGENT SECTOR COUNCIL ---
    st.markdown("##### 🦅 5-AGENT SECTOR COUNCIL")
    top5_names = ", ".join(df.head(5)["Name"].tolist())
    sec_ai_raw = generate_ai_content("sector_debate", {
        "sec": target_sector,
        "count": len(df),
        "top5": top5_names,
        "avg_rs": f"{df['RS'].mean():.2f}"
    })
    st.markdown(parse_agent_debate(sec_ai_raw), unsafe_allow_html=True)

    # --- 5. Leaderboard ---
    st.markdown("##### LEADERBOARD")
    event = st.dataframe(
        df[["Name", "Conf", "Apex", "RS", "Accel", "HighDist", "1W", "1M", "12M"]],
        column_config={
            "Conf": st.column_config.ProgressColumn("Confidence", format="%.0f", min_value=0, max_value=100),
            "Apex": st.column_config.NumberColumn(format="%.2f"),
            "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
            "Accel": st.column_config.NumberColumn(format="%.2f"),
            "HighDist": st.column_config.NumberColumn("High%", format="%.1f%%"),
            "1W": st.column_config.NumberColumn(format="%.1f%%"),
            "1M": st.column_config.NumberColumn(format="%.1f%%"),
            "12M": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table"
    )
        
    # --- 6. Deep Dive (News & Report) ---
    top = df.iloc[0] # Default
    is_default = True
    try:
        # Robust Selection
        if hasattr(event, "selection") and event.selection:
            sel_rows = event.selection.get("rows", [])
            if sel_rows:
                top = df.iloc[sel_rows[0]]
                is_default = False
    except: pass

    st.divider()
    lbl = f"{top['Name']} (Default: Top Ranked)" if is_default else top['Name']
    st.markdown(f"### 🦅 DEEP DIVE: {lbl}")
    
    # Fetch sorted news & Fundamental Data for Report
    news_items, _ = get_news_consolidated(top["Ticker"], top["Name"], limit_each=10)
    fund_data = get_fundamental_data(top["Ticker"])
    
    report_txt = generate_ai_content("stock_report", {
        "name": top["Name"], 
        "ticker": top["Ticker"],
        "fundamentals": fund_data
    })
    
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        st.markdown(f"<div class='report-box'><b>ANALYST REPORT</b><br>{report_txt}</div>", unsafe_allow_html=True)
    with nc2:
        st.caption("INTEGRATED NEWS FEED (Newest → Oldest)")
        for n in news_items[:20]:
            dt = datetime.fromtimestamp(n["pub"]).strftime("%Y/%m/%d %H:%M") if n["pub"] else "N/A"
            st.markdown(f"- {dt} [{n['src']}] [{n['title']}]({n['link']})")

if __name__ == "__main__":
    main()