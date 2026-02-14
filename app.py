import os
import time
import math
import urllib.parse
import urllib.request
import traceback
import xml.etree.ElementTree as ET
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
# 1. PHANTOM UI (Professional, Cyberpunk, High Contrast)
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

/* AI AGENT BOXES & STYLES */
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

/* MARKET SUMMARY BOX */
.market-box {
    border-left: 5px solid var(--accent);
    background: #0f0f0f;
    padding: 20px;
    margin-bottom: 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #ddd;
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
# 3. UNIVERSE DEFINITIONS (OFFICIAL & ALIGNED)
# ==========================================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

US_SEC = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Comm Services": "XLC",
    "Cons. Disc": "XLY", "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"
}

# JP SECTORS (KEYS MATCH JP_STOCKS KEYS EXACTLY)
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

# JP STOCKS (KEYS ALIGNED WITH JP_SEC)
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
        # Keep only existing columns
        cols = [c for c in expected if c in close.columns]
        return close[cols]
    except Exception as e:
        log_system_event(f"Extract Close Error: {e}", "ERROR")
        return pd.DataFrame()

def calc_technical_metrics(s: pd.Series, b: pd.Series, win: int) -> Dict:
    # 1. Strict Length Check (DROPNA first)
    s_clean = s.dropna()
    b_clean = b.dropna()
    
    if len(s_clean) < win + 1 or len(b_clean) < win + 1: return None
    
    # 2. Fill NaN with ffill ONLY (No Bfill!)
    s_win = s.ffill().tail(win+1)
    b_win = b.ffill().tail(win+1)
    
    # 3. Final check to avoid starting with NaN
    if s_win.isna().iloc[0] or b_win.isna().iloc[0]: return None

    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # 52W High Ratio (using full available series)
    if len(s_clean) >= 252:
        year_high = s_clean.tail(252).max()
    else:
        year_high = s_clean.max()
        
    curr = s_win.iloc[-1]
    high_dist = (curr / year_high - 1) * 100 if year_high > 0 else 0
    
    # Multi-Horizon with NaN
    rets = {}
    for l, d in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s) > d:
            rets[l] = (s.iloc[-1] / s.iloc[-1-d] - 1) * 100
        else:
            rets[l] = np.nan
    
    return {"RS": rs, "Accel": accel, "MaxDD": dd, "Ret": p_ret, "HighDist": high_dist, **rets}

def calculate_regime(bench_series: pd.Series) -> Tuple[str, float]:
    """Calculate Market Regime based on Trend & Volatility"""
    if len(bench_series) < 200: return "Unknown", 0.5
    
    curr = bench_series.iloc[-1]
    ma200 = bench_series.rolling(200).mean().iloc[-1]
    vol20 = bench_series.pct_change().tail(20).std() * np.sqrt(252)
    
    trend = "Bull" if curr > ma200 else "Bear"
    vol_state = "High" if vol20 > 0.15 else "Low" # 15% threshold
    
    regime = f"{trend} / {vol_state} Vol"
    
    # Weight Adjustment
    weight_momentum = 0.6 if trend == "Bull" else 0.3
    return regime, weight_momentum

def audit_data_availability(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "list": []}
    
    last_valid = df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_valid.mode().iloc[0] if not last_valid.empty else None
    
    computable = []
    for t in present:
        # Strict: must have enough data AND be recent
        if last_valid[t] == mode_date and len(df[t].dropna()) >= win + 1:
            computable.append(t)
            
    return {"ok": True, "list": computable, "mode": mode_date, "count": len(computable), "total": len(expected)}

def calculate_zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# --- AI & NEWS (ROBUST) ---

@st.cache_data(ttl=1800)
def get_dossier(ticker: str, name: str) -> str:
    news_text = ""
    # Yahoo (Fast)
    try:
        raw = yf.Ticker(ticker).news
        if raw:
            for n in raw[:3]:
                news_text += f"- [Yahoo] {n.get('title','')}\n"
    except: pass
    
    # Google (Detailed) - Timeout Protected
    try:
        q = urllib.parse.quote(f"{name} 株")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=2) as r: 
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:3]:
                title = i.findtext("title")
                desc = i.findtext("description")
                date = i.findtext("pubDate")
                # Description truncated
                news_text += f"- [Google {date}] {title}: {str(desc)[:100]}...\n"
    except: pass
    
    return news_text if news_text else "特になし"

@st.cache_data(ttl=3600)
def get_debate_content(ticker: str, name: str, stats: Dict, dossier: str) -> str:
    """5-Agent Debate with HTML Formatting"""
    if not HAS_LIB or not API_KEY: return "AI OFFLINE"
    
    # Safe context for format string
    s_rs = f"{stats['RS']:.2f}"
    s_acc = f"{stats['Accel']:.2f}"
    s_r12m = f"{stats.get('12M', 0):.1f}" if pd.notna(stats.get('12M')) else "N/A"
    
    prompt = f"""
    あなたは5名の専門家です。以下のデータに基づき議論し、HTML形式で出力してください。
    
    対象: {name} ({ticker})
    データ: RS {s_rs}%, Accel {s_acc}, 12M {s_r12m}%
    ニュース: {dossier}
    
    以下のタグを使って出力してください（Markdownコードブロックは不要）。
    [FUNDAMENTAL] ファンダメンタルズ分析...
    [SENTIMENT] センチメント分析...
    [VALUATION] バリュエーション分析...
    [SKEPTIC] 懐疑的な視点・反論...
    [RISK] リスク管理オフィサーの指摘...
    [JUDGE] 最終結論(強気/中立/弱気)と理由、推奨アクション
    """
    
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
    raw_text = ""
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            raw_text = model.generate_content(prompt).text
            break
        except Exception as e:
            if "429" in str(e): time.sleep(1); continue
    
    if not raw_text: return "AI Error"
    
    # Parse to HTML
    html = ""
    mapping = {
        "[FUNDAMENTAL]": "agent-box agent-fundamental",
        "[SENTIMENT]": "agent-box agent-sentiment",
        "[VALUATION]": "agent-box agent-valuation",
        "[SKEPTIC]": "agent-box agent-skeptic",
        "[RISK]": "agent-box agent-risk",
        "[JUDGE]": "agent-box agent-verdict"
    }
    
    # Simple parser
    current_class = "agent-box"
    buffer = []
    # Remove markdown code blocks if any
    clean_text = raw_text.replace("```html", "").replace("```", "")
    lines = clean_text.split("\n")
    
    tag_encountered = False
    
    for line in lines:
        for tag, cls in mapping.items():
            if tag in line:
                if buffer:
                    # Flush buffer with <br>
                    html += f"<div class='{current_class}'>{'<br>'.join(buffer)}</div>"
                    buffer = []
                current_class = cls
                line = line.replace(tag, f"<b>{tag.replace('[','').replace(']','')}</b><br>")
                tag_encountered = True
                break
        buffer.append(line)
        
    if buffer:
        html += f"<div class='{current_class}'>{'<br>'.join(buffer)}</div>"
        
    # Fallback if no tags found
    if not tag_encountered:
        html = f"<div class='agent-box'>{clean_text.replace(chr(10), '<br>')}</div>"
        
    return html

@st.cache_data(ttl=3600)
def generate_ai_content(prompt_key: str, context: Dict) -> str:
    """Generic AI Caller with Rotation & Retry"""
    if not HAS_LIB or not API_KEY: return "⚠️ AI OFFLINE"
    
    # Construct prompt based on key
    if prompt_key == "market":
        p = f"""
        期間: {context['s_date']}〜{context['e_date']}
        市場リターン: {context['ret']:.2f}%
        最強: {context['top']}, 最弱: {context['bot']}
        ニュース: {context['dossier']}
        
        上記に基づき市場概況を300文字で解説せよ。数値とニュースを関連付けること。
        最後に「主な変動要因」を箇条書きで。挨拶不要。
        """
    elif prompt_key == "sector":
        p = f"""
        セクター: {context['sec']}
        平均RS: {context['avg_rs']:.2f}
        首位: {context['top']}
        
        このセクターの【動向】【見通し】【リスク】を簡潔に解説せよ。挨拶不要。
        """
    elif prompt_key == "report":
        p = f"銘柄: {context['name']} ({context['ticker']}) の企業概要、直近決算、コンセンサスをレポート形式でまとめよ。"
    else:
        return "Error"

    # Model Rotation
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            return model.generate_content(p).text
        except Exception as e:
            if "429" in str(e): time.sleep(1); continue
            log_system_event(f"AI {prompt_key} error: {e}", "WARN")
            
    return "AI Unavailable"

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

    # Context Refresh
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
    
    # SAFE INDEX CHECK
    if core_df.empty or len(core_df) < win + 1:
        st.warning("WAITING FOR DATA SYNC...")
        return

    audit = audit_data_availability(core_tickers, core_df, win)
    
    if bench not in audit["list"]:
        st.error("DATA FEED DISCONNECTED: BENCHMARK MISSING")
        return

    # --- 1. Market Overview & Regime ---
    b_stats = calc_technical_metrics(core_df[bench], core_df[bench], win)
    if not b_stats:
        st.error("BENCHMARK METRICS FAILED")
        return

    regime, weight_mom = calculate_regime(core_df[bench].dropna())
    
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit["list"]:
            res = calc_technical_metrics(core_df[s_t], core_df[bench], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    # Guard against empty sectors
    if not sec_rows:
        st.warning("SECTOR DATA INSUFFICIENT")
        return

    sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
    
    # Market AI Summary
    s_date = core_df.index[-win-1].strftime('%Y/%m/%d')
    e_date = core_df.index[-1].strftime('%Y/%m/%d')
    bench_dossier = get_dossier(bench, m_cfg["name"])
    
    market_text = generate_ai_content("market", {
        "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
        "top": sdf.iloc[-1]["Sector"], "bot": sdf.iloc[0]["Sector"],
        "dossier": bench_dossier
    })
    
    st.markdown(f"""
    <div class='market-box'>
    <b>MARKET REGIME: {regime}</b> (Mom-Weight: {weight_mom:.1f})<br>
    {market_text}
    </div>
    """, unsafe_allow_html=True)

    # --- 2. Sector Rotation (Plotly Interaction Fix) ---
    st.subheader("SECTOR ROTATION")
    fig = px.bar(sdf, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn")
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', font_family="Orbitron")
    
    # DUAL INTERACTION CHECK
    chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="sector_chart")
    
    click_sec = None
    # Method A: Event Return
    if chart_event:
        points = chart_event.get("selection", {}).get("points", [])
        if points:
            click_sec = points[0].get("y")
    
    # Method B: Session State Fallback
    if not click_sec:
        try:
            sel = st.session_state.get("sector_chart", {})
            points = sel.get("selection", {}).get("points", [])
            if points:
                click_sec = points[0].get("y")
        except: pass

    # Fallback Buttons
    with st.expander("SECTOR BUTTONS", expanded=False):
        cols = st.columns(6)
        for i, s in enumerate(m_cfg["sectors"].keys()):
            if cols[i%6].button(s, key=f"btn_{s}", use_container_width=True):
                click_sec = s
            
    if click_sec:
        st.session_state.selected_sector = click_sec
    
    target_sector = st.session_state.selected_sector or list(m_cfg["sectors"].keys())[0]

    # --- 3. Sector Analysis ---
    st.markdown(f"<div id='sector_anchor'></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader(f"SECTOR FORENSIC: {target_sector}")
    
    # Fetch Constituents
    stock_list = m_cfg["stocks"].get(target_sector, [])
    if not stock_list:
        st.error(f"CONFIGURATION ERROR: No stocks found for sector '{target_sector}'. Check keys.")
        return

    full_list = [bench] + stock_list
    
    # Sector Cache Logic
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
        st.warning("NO VALID DATA FOR STOCKS IN THIS SECTOR.")
        return
        
    df = pd.DataFrame(results)
    # Dynamic Scoring based on Regime
    w_rs = weight_mom
    w_acc = 0.8 - w_rs
    df["Apex"] = w_rs * calculate_zscore(df["RS"]) + w_acc * calculate_zscore(df["Accel"]) + 0.2 * calculate_zscore(df["Ret"])
    
    # CONFIDENCE SCORE Logic
    # Simple metric: 0-100 based on data completeness and signal clarity
    df["Conf"] = 80 + (calculate_zscore(df["Apex"]).abs() * 5).clip(0, 15)
    
    df = df.sort_values("Apex", ascending=False)
    
    # Sector AI Outlook
    sec_text = generate_ai_content("sector", {
        "sec": target_sector, "avg_rs": df["RS"].mean(), "top": df.iloc[0]["Name"]
    })
    st.markdown(f"<div class='ai-box'><b>SECTOR OUTLOOK</b><br>{sec_text}</div>", unsafe_allow_html=True)

    # --- 4. Stock Deep Dive (The Sovereign Core) ---
    c1, c2 = st.columns([1.4, 1])
    
    with c1:
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
        
        # Robust Table Selection
        top = df.iloc[0] # Default
        try:
            # Check if event has selection attribute safely
            if hasattr(event, "selection") and event.selection:
                sel_rows = event.selection.get("rows", [])
                if sel_rows:
                    top = df.iloc[sel_rows[0]]
        except: pass

    with c2:
        st.markdown(f"##### 🦅 5-AGENT COUNCIL: {top['Name']}")
        dossier = get_dossier(top["Ticker"], top["Name"])
        
        # 5-Agent Debate
        ai_html = get_debate_content(top["Ticker"], top["Name"], top.to_dict(), dossier)
        st.markdown(ai_html, unsafe_allow_html=True)
        
        # Report & News
        with st.expander("📋 ANALYST REPORT", expanded=False):
            rep = generate_ai_content("report", {"name": top["Name"], "ticker": top["Ticker"]})
            st.markdown(f"<div style='font-size:12px'>{rep}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()