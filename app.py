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
    initial_sidebar_state="expanded", # サイドバーを展開してアイコンバグを防ぐ
    page_icon="🦅"
)

# Global Error Boundary
def error_boundary(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
            st.code(traceback.format_exc())
            # Stop execution safely without crashing the whole app
            return None
    return wrapper

# ==========================================
# 1. PHANTOM UI DESIGN (CSS)
# ==========================================
st.markdown("""
<style>
/* FONTS */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Noto+Sans+JP:wght@400;700&display=swap');

:root {
  --bg: #000000;
  --panel: #080808;
  --card: #101010;
  --border: #333333;
  --accent: #00f2fe;
  --text: #e0e0e0;
}

/* GLOBAL RESET */
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; }
* { font-family: 'Orbitron', 'Noto Sans JP', sans-serif !important; letter-spacing: 0.5px !important; }

/* HIDE DEFAULT ELEMENTS THAT CAUSE GLITCHES */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* BRANDING */
h1, h2, h3, .brand {
  background: linear-gradient(90deg, #fff, #00f2fe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900 !important;
  text-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
  text-transform: uppercase;
}

/* CONTAINERS */
.deck { background: var(--panel); border: 1px solid var(--accent); padding: 20px; margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 242, 254, 0.15); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 15px; margin-bottom: 10px; }

/* TABLE STYLING (NEON CYBERPUNK) */
div[data-testid="stDataFrame"] { background-color: #000 !important; border: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] * { background-color: #000 !important; color: #e0e0e0 !important; font-family: 'Orbitron', monospace !important; }
[data-testid="stHeader"] { border-bottom: 2px solid var(--accent) !important; }
[data-testid="stHeader"] * { color: var(--accent) !important; font-weight: 900 !important; }

/* INPUTS & SELECTBOX (FIX WHITE-ON-WHITE) */
div[data-baseweb="select"] > div { background-color: #111 !important; border-color: #444 !important; color: #fff !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #000 !important; border: 1px solid #444 !important; }
div[data-baseweb="option"] { color: #fff !important; }
li[data-baseweb="option"]:hover, li[aria-selected="true"] { background-color: #222 !important; color: #00f2fe !important; }
.stSelectbox label { color: #888 !important; }
.stProgress > div > div > div > div { background-color: var(--accent) !important; }

/* BUTTONS */
button {
  background-color: #000 !important;
  color: var(--accent) !important;
  border: 1px solid #333 !important;
  border-radius: 0px !important;
  font-weight: 800 !important;
  text-transform: uppercase;
  transition: all 0.3s;
}
button:hover { border-color: var(--accent) !important; box-shadow: 0 0 15px var(--accent) !important; color: #fff !important; }

/* METRICS */
.kpi { border-left: 4px solid var(--border); background: var(--panel); padding: 10px; margin-bottom: 10px; }
.kpi-val { font-size: 24px; color: var(--accent); font-weight: 700; text-shadow: 0 0 10px rgba(0,242,254,0.4); }
.kpi-lbl { font-size: 10px; color: #888; text-transform: uppercase; }
.status-ok { border-color: #238636 !important; }
.status-ng { border-color: #da3633 !important; }

/* AI BOX */
.ai-box { border: 1px dashed var(--accent); background: rgba(0,242,254,0.05); padding: 20px; margin-top: 15px; line-height: 1.8; font-size: 13px; color: #e0e0e0; }

/* UTILS */
.muted { color: #888 !important; font-size: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTH & SETUP
# ==========================================
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

# AI Library Setup
try:
    import google.generativeai as genai
    HAS_LIB = True
    if API_KEY: genai.configure(api_key=API_KEY)
except Exception:
    HAS_LIB = False

def check_access():
    if not APP_PASS: return True
    if st.session_state.get("auth", False): return True
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br><h3 style='text-align:center'>SECURITY GATE</h3>", unsafe_allow_html=True)
        with st.form("auth"):
            p = st.text_input("PASSCODE", type="password")
            if st.form_submit_button("UNLOCK SYSTEM", use_container_width=True):
                if p == APP_PASS:
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("ACCESS DENIED")
    return False

if not check_access(): st.stop()

# ==========================================
# 3. UNIVERSE DEFINITIONS (COMPLETE)
# ==========================================
LOOKBACKS = {"1W": 5, "1M": 21, "3M": 63, "12M": 252}
FETCH_PERIOD = "24mo"

# SECTORS
US_SEC = {"Tech":"XLK", "Health":"XLV", "Fin":"XLF", "Comm":"XLC", "Disc":"XLY", "Staples":"XLP", "Ind":"XLI", "Energy":"XLE", "Mat":"XLB", "Util":"XLU", "RE":"XLRE"}
JP_SEC = {"通信":"1626.T", "電機":"1631.T", "自動車":"1621.T", "医薬":"1632.T", "銀行":"1623.T", "金融":"1624.T", "商社":"1622.T", "機械":"1630.T", "エネ":"1617.T", "建設":"1618.T", "素材":"1619.T", "食品":"1633.T", "電力":"1628.T", "不動産":"1625.T", "鉄鋼":"1629.T", "サービス":"1627.T", "産機":"1620.T"}

# STOCKS
US_STOCKS = {
    "Tech": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","AMD","QCOM","TXN","INTU","IBM","NOW","AMAT","MU","LRCX","ADI","KLAC","SNPS","CDNS","PANW","CRWD","ANET","PLTR"],
    "Comm": ["GOOGL","META","NFLX","DIS","CMCSA","TMUS","VZ","T","CHTR","WBD","LYV","EA","TTWO","OMC","IPG"],
    "Health": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","PFE","ISRG","DHR","VRTX","GILD","REGN","BMY","CVS","CI","SYK","BSX","MDT","ZTS","HCA","MCK"],
    "Fin": ["JPM","BAC","WFC","V","MA","AXP","GS","MS","BLK","C","SCHW","SPGI","PGR","CB","MMC","KKR","BX","TRV","AFL","MET","PRU","ICE","COF"],
    "Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","CMG","MAR","HLT","YUM","LULU","GM","F","ROST","ORLY","AZO","DHI","LEN"],
    "Staples": ["PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","KMB","GIS","KHC","KR","STZ","EL","TGT","DG","ADM","SYY"],
    "Ind": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","WM","NSC","CSX","GD","NOC","TDG","PCAR","FDX","CTAS"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","WMB","HAL","BKR","DVN","HES","FANG","TRGP","OKE"],
    "Mat": ["LIN","APD","SHW","FCX","ECL","NEM","DOW","DD","NUE","MLM","VMC","CTVA","PPG","ALB","CF","MOS"],
    "Util": ["NEE","DUK","SO","AEP","SRE","EXC","XEL","D","PEG","ED","EIX","WEC","AWK","ES","PPL","ETR"],
    "RE": ["PLD","AMT","CCI","EQIX","SPG","PSA","O","WELL","DLR","AVB","EQR","VICI","CSGP","SBAC","IRM"],
}
JP_STOCKS = {
    "通信": ["9432.T","9433.T","9434.T","9984.T","4689.T","4755.T","9613.T","9602.T","4385.T","6098.T","3659.T","3765.T"],
    "電機": ["8035.T","6857.T","6146.T","6920.T","6758.T","6501.T","6723.T","6981.T","6954.T","7741.T","6702.T","6503.T","6752.T","7735.T","6861.T"],
    "自動車": ["7203.T","7267.T","6902.T","7201.T","7269.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T","9005.T"],
    "医薬": ["4502.T","4568.T","4519.T","4503.T","4507.T","4523.T","4578.T","4151.T","4528.T","4506.T"],
    "銀行": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T","8331.T","8354.T"],
    "金融": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T","8570.T"],
    "商社": ["8001.T","8031.T","8058.T","8053.T","8002.T","8015.T","3382.T","9983.T","8267.T","2914.T","7453.T","3092.T"],
    "機械": ["6301.T","7011.T","7012.T","6367.T","6273.T","6113.T","6473.T","6326.T"],
    "エネ": ["1605.T","5020.T","9501.T","3407.T","4005.T"],
    "建設": ["1925.T","1928.T","1801.T","1802.T","1812.T","5201.T","5332.T"],
    "素材": ["4063.T","4452.T","4188.T","4901.T","4911.T","4021.T","4631.T","3402.T"],
    "食品": ["2801.T","2802.T","2269.T","2502.T","2503.T","2201.T","2002.T"],
    "電力": ["9501.T","9503.T","9531.T","9532.T"],
    "不動産": ["8801.T","8802.T","8830.T","3289.T","3003.T","3231.T"],
    "鉄鋼": ["5401.T","5411.T","5713.T","5406.T","5711.T","5802.T"],
    "サービス": ["4661.T","9735.T","4324.T","2127.T","6028.T","2412.T","4689.T"],
    "産機": ["6146.T","6460.T","6471.T","6268.T"]
}

MARKETS = {
    "🇺🇸 US": {"bench": "SPY", "name": "S&P 500", "sectors": US_SEC, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": "1306.T", "name": "TOPIX", "sectors": JP_SEC, "stocks": JP_STOCKS},
}

# FULL NAME DB
NAME_DB = {
    "SPY":"S&P500","1306.T":"TOPIX","XLK":"Tech","XLV":"Health","XLF":"Fin","XLC":"Comm","XLY":"ConsDisc","XLP":"Staples","XLI":"Indust","XLE":"Energy","XLB":"Material","XLU":"Utility","XLRE":"RealEst",
    "1626.T":"情報通信","1631.T":"電機精密","1621.T":"自動車","1632.T":"医薬品","1623.T":"銀行","1624.T":"金融他","1622.T":"商社小売","1630.T":"機械","1617.T":"エネ資源","1618.T":"建設資材","1619.T":"素材化学","1633.T":"食品","1628.T":"電力ガス","1625.T":"不動産","1629.T":"鉄鋼非鉄","1627.T":"サービス","1620.T":"産業機械",
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

def get_name(t: str) -> str:
    return NAME_DB.get(t, t)

# ==========================================
# 4. CORE ENGINES
# ==========================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    # Remove duplicates and None
    tickers = tuple(dict.fromkeys([t for t in tickers if t]))
    frames = []
    chunk = 50 # Smaller chunk size for reliability
    for i in range(0, len(tickers), chunk):
        c = tickers[i:i+chunk]
        try:
            r = yf.download(" ".join(c), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not r.empty: frames.append(r)
        except: continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

def extract_close(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0): close = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(1): close = df.xs("Close", axis=1, level=1)
            else: return pd.DataFrame()
        else: return pd.DataFrame()
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        # Keep only existing columns
        keep = [c for c in expected if c in close.columns]
        return close[keep]
    except: return pd.DataFrame()

def calc_multi_horizon(s: pd.Series) -> Dict[str, float]:
    res = {}
    for label, d in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s) > d:
            res[label] = (s.iloc[-1] / s.iloc[-1-d] - 1) * 100
        else: res[label] = np.nan
    return res

def calc_metrics(s: pd.Series, b: pd.Series, win: int) -> Dict:
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
    
    s_short, b_short = s.tail(6).dropna(), b.tail(6).dropna()
    stable = "⚠️"
    if len(s_short)==6 and len(b_short)==6:
        rs_s = (s_short.iloc[-1]/s_short.iloc[0]-1) - (b_short.iloc[-1]/b_short.iloc[0]-1)
        if np.sign(rs_s) == np.sign(rs): stable = "✅"
    
    return {"RS": rs, "Accel": accel, "MaxDD": dd, "Stable": stable, "Ret": p_ret}

def audit_gate(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "list": []}
    last = df[present].apply(lambda x: x.last_valid_index())
    mode = last.mode().iloc[0] if not last.mode().empty else None
    computable = []
    for t in present:
        if last[t] == mode and df[t].tail(win+1).notna().sum() >= win+1:
            computable.append(t)
    return {"ok": True, "list": computable, "mode": mode, "count": len(computable), "total": len(expected)}

def zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# ==========================================
# 5. AI & NEWS ENGINE
# ==========================================
def call_ai(ticker: str, name: str, stats: Dict) -> str:
    # 1. Check Libraries
    if not HAS_LIB or not API_KEY:
        return "⚠️ AI OFFLINE: CHECK KEYS"

    # 2. Retry Logic for 429 Errors
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Using 2.0-flash as primary per user log
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            prompt = f"""
            あなたはプロのファンドマネージャーです。
            以下の銘柄について3名のエージェント（モメンタム、リスク、マクロ）として議論し、結論を出してください。
            
            銘柄: {name} ({ticker})
            指標: RS {stats['RS']:.2f}% (市場比), 加速 {stats['Accel']:.2f}, 最大下落 {stats['MaxDD']:.2f}%
            騰落: 1W {stats.get('1W',0):.1f}%, 1M {stats.get('1M',0):.1f}%, 12M {stats.get('12M',0):.1f}%
            
            出力形式:
            【モメンタム】...
            【リスク】...
            【結論】(強気/中立/弱気) 理由1行
            """
            
            response = model.generate_content(prompt)
            if response.text: return response.text
            
        except Exception as e:
            # Retry on 429
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(5) # Wait 5s for free tier reset
                continue
            
            # Return error on final attempt
            if attempt == max_retries - 1:
                return f"⚠️ AI BUSY (429): {str(e)}"
    
    return "AI ERROR"

@st.cache_data(ttl=1800)
def fetch_news(ticker: str, name: str) -> Tuple[List[dict], List[dict]]:
    y, g = [], []
    try:
        raw = yf.Ticker(ticker).news
        if raw: y = [{"title": n.get("title",""), "link": n.get("link","")} for n in raw[:4]]
    except: pass
    try:
        q = urllib.parse.quote(f"{name} 株")
        with urllib.request.urlopen(f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja", timeout=4) as r:
            root = ET.fromstring(r.read())
            g = [{"title": i.findtext("title"), "link": i.findtext("link")} for i in root.findall(".//item")[:4]]
    except: pass
    return y, g

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
@error_boundary
def main():
    st.markdown("<h2 class='brand'>ALPHALENS SOVEREIGN</h2>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### SYSTEM STATUS")
        ai_st = "ONLINE" if HAS_LIB and API_KEY else "OFFLINE"
        c = "#238636" if ai_st == "ONLINE" else "#da3633"
        st.markdown(f"**AI ENGINE**: <span style='color:{c}'>{ai_st}</span>", unsafe_allow_html=True)
        st.caption("Model: gemini-2.0-flash")

    # Header Controls
    with st.container():
        st.markdown("<div class='deck'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
        with c1: market_key = st.selectbox("MARKET", list(MARKETS.keys()))
        with c2: lookback_key = st.selectbox("WINDOW", list(LOOKBACKS.keys()), index=1)
        with c3: st.caption(f"PERIOD: {FETCH_PERIOD}"); st.progress(100)
        with c4: 
            st.write("")
            sync = st.button("SYNC", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # 1. Sync Data
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if sync or "core_df" not in st.session_state or st.session_state.get("last_m") != market_key:
        with st.spinner("SYNCING MARKET DATA..."):
            raw = fetch_bulk_cached(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    audit_res = audit_gate(core_tickers, core_df, win)
    
    if bench not in audit_res["list"]:
        st.error("DATA FEED DISCONNECTED")
        return

    col1, col2 = st.columns(2)
    with col1: st.markdown(f"<div class='kpi status-ok'><div class='kpi-lbl'>DATA HEALTH</div><div class='kpi-val'>{audit_res['count']}/{audit_res['total']}</div></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='kpi status-ok'><div class='kpi-lbl'>LATEST DATE</div><div class='kpi-val'>{str(audit_res['mode']).split()[0]}</div></div>", unsafe_allow_html=True)

    # 2. Sector Overview
    b_stats = calc_metrics(core_df[bench], core_df[bench], win)
    
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit_res["list"]:
            res = calc_metrics(core_df[s_t], core_df[bench], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    if not sec_rows:
        st.warning("NO SECTOR DATA")
        return

    sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
    sdf_chart = pd.concat([sdf, pd.DataFrame([{"Sector":"MARKET", "RS":0}])], ignore_index=True).sort_values("RS")
    
    st.subheader("SECTOR ROTATION")
    fig = px.bar(sdf_chart, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn", title=f"RS ({lookback_key})")
    fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', font_family="Orbitron")
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    click_sec = event["selection"]["points"][0]["y"] if event and event.get("selection", {}).get("points") else None
    
    # 3. Drill Down
    cols = st.columns(6)
    btn_sec = None
    for i, s in enumerate(m_cfg["sectors"].keys()):
        if cols[i%6].button(s, key=f"b_{s}", use_container_width=True): btn_sec = s
            
    target_sector = btn_sec or click_sec or st.session_state.get("target_sector", list(m_cfg["sectors"].keys())[0])
    st.session_state.target_sector = target_sector
    
    st.markdown("---")
    st.subheader(f"FORENSIC: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    cache_key = f"{market_key}_{target_sector}"
    if cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner("SCANNING SECTOR ASSETS..."):
            raw_s = fetch_bulk_cached(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_gate(full_list, sec_df, win)
    
    results = []
    for t in [x for x in s_audit["list"] if x != bench]:
        stats = calc_metrics(sec_df[t], sec_df[bench], win)
        if stats:
            stats.update(calc_multi_horizon(sec_df[t]))
            stats["Ticker"] = t
            stats["Name"] = get_name(t)
            results.append(stats)
            
    if not results:
        st.warning("NO STOCKS FOUND.")
        return
        
    df = pd.DataFrame(results)
    df["RS_z"] = zscore(df["RS"])
    df["Acc_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["Apex"] = 0.6*df["RS_z"] + 0.25*df["Acc_z"] - 0.15*df["DD_z"]
    df = df.sort_values("Apex", ascending=False).reset_index(drop=True)
    df["Verdict"] = df.apply(lambda r: "STRONG" if r["RS"]>0 and r["Accel"]>0 and r["Stable"]=="✅" else "WATCH" if r["RS"]>0 else "AVOID", axis=1)

    # 4. Table
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("##### LEADERBOARD")
        event_table = st.dataframe(
            df[["Name", "Verdict", "Apex", "RS", "Accel", "1W", "1M", "3M", "12M"]],
            column_config={
                "Apex": st.column_config.NumberColumn(format="%.2f"),
                "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
                "Accel": st.column_config.NumberColumn(format="%.2f"),
                "1W": st.column_config.NumberColumn(format="%.1f%%"),
                "1M": st.column_config.NumberColumn(format="%.1f%%"),
                "3M": st.column_config.NumberColumn(format="%.1f%%"),
                "12M": st.column_config.NumberColumn(format="%.1f%%"),
            },
            hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
        )
        
    sel_rows = event_table.selection.get("rows", [])
    top = df.iloc[sel_rows[0]] if sel_rows else df.iloc[0]
    
    # 5. AI & News
    with c2:
        st.markdown(f"##### AI INTELLIGENCE: {top['Name']}")
        ai_txt = call_ai(top["Ticker"], top["Name"], top.to_dict())
        st.markdown(f"<div class='ai-box'>{ai_txt}</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader(f"INTELLIGENCE FEED: {top['Name']}")
    yn, gn = fetch_news(top["Ticker"], top["Name"])
    
    n1, n2 = st.columns(2)
    with n1:
        st.caption("YAHOO FINANCE")
        if not yn: st.write("NO DATA")
        for n in yn: st.markdown(f"- [{n['title']}]({n['link']})")
    with n2:
        st.caption("GOOGLE NEWS")
        if not gn: st.write("NO DATA")
        for n in gn: st.markdown(f"- [{n['title']}]({n['link']})")

if __name__ == "__main__":
    main()