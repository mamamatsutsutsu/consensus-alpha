import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# =========================
# 1. Phantom UI Configuration (ALL ORBITRON)
# =========================
st.set_page_config(page_title="AlphaLens Sovereign", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* FONT IMPORT */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=IBM+Plex+Sans+JP:wght@400;700&display=swap');

/* --- PHANTOM DARK THEME --- */
:root {
  --bg-app: #000000;
  --bg-panel: #050505;
  --bg-card: #0a0a0a;
  --border: #333333;
  --accent: #00f2fe;
  --text-main: #e0e0e0;
  --text-dim: #888888;
}

/* GLOBAL FONT FORCE (ALL ORBITRON) */
html, body, [class*="css"], div, button, span, p, h1, h2, h3, h4, h5, h6, input, select, textarea {
  font-family: 'Orbitron', 'IBM Plex Sans JP', sans-serif !important;
  letter-spacing: 1px !important;
  color: var(--text-main) !important;
}

/* APP BACKGROUND */
.stApp { background-color: var(--bg-app) !important; }

/* BRANDING */
.brand-box { text-align: center; margin-bottom: 20px; padding: 20px 0; border-bottom: 1px solid var(--border); }
.brand-title {
  font-size: 36px; font-weight: 900;
  background: linear-gradient(90deg, #FFF 0%, #00f2fe 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  text-shadow: 0 0 25px rgba(0, 242, 254, 0.6);
}

/* CONTAINERS */
.deck {
  background: var(--bg-panel);
  border: 1px solid var(--accent);
  border-radius: 0px; 
  padding: 15px; margin-bottom: 20px;
  box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 15px; margin-bottom: 10px;
}

/* TABLE STYLING (NEON) */
div[data-testid="stDataFrame"] {
  background-color: #000000 !important;
  border: 1px solid var(--accent) !important;
  box-shadow: 0 0 10px rgba(0, 242, 254, 0.1);
}
div[data-testid="stDataFrame"] * {
  font-family: 'Orbitron', monospace !important;
  color: #e0e0e0 !important;
}
[data-testid="stHeader"] {
  background-color: #0a0a0a !important;
  border-bottom: 1px solid var(--accent) !important;
}

/* BUTTONS (HIGH VISIBILITY) */
div.stButton > button {
  background-color: #000000 !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0px !important;
  font-weight: 700 !important;
  padding: 0.5rem 1rem !important;
  text-transform: uppercase;
}
div.stButton > button:hover, div.stButton > button:active {
  background-color: var(--accent) !important;
  color: #000000 !important;
  border-color: #ffffff !important;
  box-shadow: 0 0 20px var(--accent) !important;
}

/* METRICS */
.kpi-box {
  background: #050505; border: 1px solid var(--border);
  padding: 10px; text-align: center; margin-bottom: 5px;
}
.kpi-val { font-size: 18px; color: var(--accent); font-weight: 700; text-shadow: 0 0 5px var(--accent); }
.kpi-label { font-size: 10px; color: var(--text-dim); }

/* AI BOX */
.ai-box {
  border: 1px dashed var(--accent);
  background: rgba(0, 242, 254, 0.05);
  padding: 15px;
  font-size: 12px;
  line-height: 1.6;
}

/* SIDEBAR DIAGNOSTIC */
.diag-box {
  border: 1px solid #da3633;
  background: rgba(218, 54, 51, 0.1);
  padding: 10px;
  font-size: 10px;
  margin-bottom: 10px;
}
.diag-ok { border-color: #238636; background: rgba(35, 134, 54, 0.1); }

</style>
""", unsafe_allow_html=True)

# =========================
# 2. Secrets & AI Setup (With Diagnostics)
# =========================
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

# Library Check
try:
    import google.generativeai as genai
    HAS_GENAI_LIB = True
except ImportError:
    HAS_GENAI_LIB = False

# Setup GenAI if possible
if HAS_GENAI_LIB and API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except: pass

def check_auth():
    if not APP_PASS: return True
    if st.session_state.get("auth", False): return True
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.form("login"):
            st.markdown("<h3 style='text-align:center;'>ACCESS CONTROL</h3>", unsafe_allow_html=True)
            pwd = st.text_input("PASSCODE", type="password")
            if st.form_submit_button("ENTER SYSTEM", use_container_width=True):
                if pwd == APP_PASS:
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("DENIED")
    return False

if not check_auth(): st.stop()

# =========================
# 3. Master Universe
# =========================
LOOKBACKS = {"1W": 5, "1M": 21, "3M": 63, "12M": 252}
FETCH_PERIOD = "24mo"

US_SECTOR_ETF = {"Tech": "XLK", "Health": "XLV", "Fin": "XLF", "Comm": "XLC", "Disc": "XLY", "Staples": "XLP", "Ind": "XLI", "Energy": "XLE", "Mat": "XLB", "Util": "XLU", "RE": "XLRE"}
JP_SECTOR_ETF = {"情報通信": "1626.T", "電機精密": "1631.T", "自動車": "1621.T", "医薬品": "1632.T", "銀行": "1623.T", "金融他": "1624.T", "商社小売": "1622.T", "機械": "1630.T", "エネ": "1617.T", "建設資材": "1618.T", "素材化学": "1619.T", "食品": "1633.T", "電力ガス": "1628.T", "不動産": "1625.T", "鉄鋼非鉄": "1629.T", "サービス": "1627.T", "産業機械": "1620.T"}

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
    "情報通信": ["9432.T","9433.T","9434.T","9984.T","4689.T","4755.T","9613.T","9602.T","4385.T","6098.T","3659.T","3765.T"],
    "電機精密": ["8035.T","6857.T","6146.T","6920.T","6758.T","6501.T","6723.T","6981.T","6954.T","7741.T","6702.T","6503.T","6752.T","7735.T","6861.T"],
    "自動車": ["7203.T","7267.T","6902.T","7201.T","7269.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T","9005.T"],
    "医薬品": ["4502.T","4568.T","4519.T","4503.T","4507.T","4523.T","4578.T","4151.T","4528.T","4506.T"],
    "銀行": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T","8331.T","8354.T"],
    "金融他": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T","8570.T"],
    "商社小売": ["8001.T","8031.T","8058.T","8053.T","8002.T","8015.T","3382.T","9983.T","8267.T","2914.T","7453.T","3092.T"],
    "機械": ["6301.T","7011.T","7012.T","6367.T","6273.T","6113.T","6473.T","6326.T"],
    "エネ": ["1605.T","5020.T","9501.T","3407.T","4005.T"],
    "建設資材": ["1925.T","1928.T","1801.T","1802.T","1812.T","5201.T","5332.T"],
    "素材化学": ["4063.T","4452.T","4188.T","4901.T","4911.T","4021.T","4631.T","3402.T"],
    "食品": ["2801.T","2802.T","2269.T","2502.T","2503.T","2201.T","2002.T"],
    "電力ガス": ["9501.T","9503.T","9531.T","9532.T"],
    "不動産": ["8801.T","8802.T","8830.T","3289.T","3003.T","3231.T"],
    "鉄鋼非鉄": ["5401.T","5411.T","5713.T","5406.T","5711.T","5802.T"],
    "サービス": ["4661.T","9735.T","4324.T","2127.T","6028.T","2412.T","4689.T"],
    "産業機械": ["6146.T","6460.T","6471.T","6268.T"]
}

MARKETS = {
    "🇺🇸 US": {"bench": "SPY", "name": "S&P 500", "sectors": US_SECTOR_ETF, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": "1306.T", "name": "TOPIX", "sectors": JP_SECTOR_ETF, "stocks": JP_STOCKS},
}

NAME_DB = {
    "SPY":"S&P500","1306.T":"TOPIX","XLK":"Tech","XLV":"Health","XLF":"Financial","XLC":"Comm","XLY":"ConsDisc","XLP":"Staples","XLI":"Indust","XLE":"Energy","XLB":"Material","XLU":"Utility","XLRE":"RealEst",
    "1626.T":"情報通信","1631.T":"電機精密","1621.T":"自動車","1632.T":"医薬品","1623.T":"銀行","1624.T":"金融他","1622.T":"商社小売","1630.T":"機械","1617.T":"エネ資源","1618.T":"建設資材","1619.T":"素材化学","1633.T":"食品","1628.T":"電力ガス","1625.T":"不動産","1629.T":"鉄鋼非鉄","1627.T":"サービス","1620.T":"産業機械",
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","GOOGL":"Alphabet","META":"Meta","AMZN":"Amazon","TSLA":"Tesla","AVGO":"Broadcom","ORCL":"Oracle","CRM":"Salesforce","ADBE":"Adobe","AMD":"AMD","QCOM":"Qualcomm","TXN":"Texas Inst","NFLX":"Netflix","DIS":"Disney","CMCSA":"Comcast","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T",
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
    "6146.T":"ディスコ","6460.T":"セガサミー","6471.T":"日本精工","6268.T":"ナブテスコ","2801.T":"キッコーマン","2802.T":"味の素"
}

def get_display_name(t: str) -> str: return NAME_DB.get(t, t)

# =========================
# 4. Engine
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk(tickers: Tuple[str, ...]) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t]))
    frames = []
    chunk = 80
    for i in range(0, len(tickers), chunk):
        c = tickers[i:i+chunk]
        try:
            r = yf.download(" ".join(c), period=FETCH_PERIOD, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
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
        keep = [c for c in expected if c in close.columns]
        return close[keep]
    except: return pd.DataFrame()

def calc_stats(s: pd.Series, b: pd.Series, win: int) -> Dict:
    # 1. Base Integrity
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    # 2. Main Metrics
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # Stable
    s_short, b_short = s.tail(6).dropna(), b.tail(6).dropna()
    stable = "⚠️"
    if len(s_short)==6 and len(b_short)==6:
        rs_s = (s_short.iloc[-1]/s_short.iloc[0]-1) - (b_short.iloc[-1]/b_short.iloc[0]-1)
        if np.sign(rs_s) == np.sign(rs): stable = "✅"
    
    # 3. Multi-Horizon Returns (Robust)
    rets = {}
    for label, days in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s) > days:
            rets[label] = (s.iloc[-1]/s.iloc[-1-days]-1)*100
        else:
            rets[label] = np.nan
            
    return {"RS": rs, "Accel": accel, "MaxDD": dd, "Stable": stable, "Ret": p_ret, **rets}

def audit(expected: List[str], df: pd.DataFrame, win: int):
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

# =========================
# AI & News (Robust)
# =========================
@st.cache_data(ttl=1800)
def get_news_robust(ticker: str, name: str) -> Tuple[List[dict], List[dict]]:
    y_news = []
    try:
        raw = yf.Ticker(ticker).news
        if raw and isinstance(raw, list):
            for n in raw[:4]:
                y_news.append({
                    "title": n.get("title", "No Title"),
                    "link": n.get("link", "#"),
                    "src": n.get("publisher", "Yahoo")
                })
    except: pass
    
    g_news = []
    try:
        q = urllib.parse.quote(f"{name} 株")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=4) as r:
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:4]:
                g_news.append({
                    "title": i.findtext("title"),
                    "link": i.findtext("link"),
                    "src": "Google"
                })
    except: pass
    return y_news, g_news

def check_ai_health():
    # Diagnostic for Sidebar
    if not HAS_GENAI_LIB: return "LIBRARY_MISSING"
    if not API_KEY: return "KEY_MISSING"
    return "ONLINE"

def call_gemini(ticker: str, name: str, stats: Dict) -> str:
    # 1. Try Gemini
    if HAS_GENAI_LIB and API_KEY:
        try:
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"""
            プロの投資家として、以下の銘柄を議論（モメンタム、リスク、マクロ視点）し、結論を出せ。
            日本語で、断定的に。
            
            銘柄: {name} ({ticker})
            指標: RS {stats['RS']:.2f}% (市場比), Accel {stats['Accel']:.2f}, DD {stats['MaxDD']:.2f}%
            騰落: 1W {stats.get('1W','N/A')}%, 1M {stats.get('1M','N/A')}%, 12M {stats.get('12M','N/A')}%
            
            形式:
            【モメンタム】...
            【リスク】...
            【結論】(強気/中立/弱気) 理由1行
            """
            resp = model.generate_content(prompt)
            if resp and resp.text: return resp.text
        except Exception as e:
            return f"⚠️ AI ERROR: {str(e)}"
            
    # 2. Fallback
    v = "強気" if stats['RS']>0 and stats['Accel']>0 else "中立"
    msg = "LIB MISSING" if not HAS_GENAI_LIB else "KEY MISSING"
    return f"※AI UNAVAILABLE ({msg}). RULE-BASED:\nTREND: {v}\nRS: {stats['RS']:.2f}% | 12M: {stats['12M']:.1f}%"

# =========================
# 5. Main UI
# =========================
def main():
    # Sidebar: Diagnostic
    with st.sidebar:
        st.markdown("### 🤖 SYSTEM STATUS")
        status = check_ai_health()
        color = "#238636" if status == "ONLINE" else "#da3633"
        st.markdown(f"<div style='border:1px solid {color}; padding:10px; color:{color}; font-weight:bold; text-align:center;'>{status}</div>", unsafe_allow_html=True)
        if status != "ONLINE":
            st.caption("Check requirements.txt (google-generativeai) or secrets.toml (GEMINI_API_KEY).")

    # Header
    st.markdown("<div class='brand-box'><div class='brand-title'>ALPHALENS</div><div class='brand-sub'>COMMAND CENTER v36.0</div></div>", unsafe_allow_html=True)

    # Deck
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
        with st.spinner("INITIATING DATA SYNC..."):
            raw = fetch_bulk(tuple(core_tickers))
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    audit_res = audit(core_tickers, core_df, win)
    
    if bench not in audit_res["list"]:
        st.error("SYSTEM HALT: BENCHMARK DATA MISSING")
        st.stop()

    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div class='kpi-box status-green'><div class='kpi-label'>SYSTEM HEALTH</div><div class='kpi-val'>{audit_res['count']}/{audit_res['total']}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi-box status-green'><div class='kpi-label'>DATA MODE</div><div class='kpi-val'>{str(audit_res['mode']).split()[0]}</div></div>", unsafe_allow_html=True)

    # 2. Market Overview
    b_stats = calc_stats(core_df[bench], core_df[bench], win)
    
    sec_data = []
    for s_name, s_tk in m_cfg["sectors"].items():
        if s_tk in audit_res["list"]:
            res = calc_stats(core_df[s_tk], core_df[bench], win)
            if res:
                res["Sector"] = s_name
                sec_data.append(res)
    
    sdf = pd.DataFrame(sec_data).sort_values("RS", ascending=True)
    sdf_chart = pd.concat([sdf, pd.DataFrame([{"Sector": "MARKET", "RS": 0, "Ret": b_stats["Ret"]}])], ignore_index=True).sort_values("RS")
    
    st.subheader("SECTOR ROTATION")
    fig = px.bar(sdf_chart, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn", title=f"RS ({lookback_key})")
    fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3', font_family="Orbitron")
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
            raw_s = fetch_bulk(tuple(full_list))
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit(full_list, sec_df, win)
    
    results = []
    for t in [x for x in s_audit["list"] if x != bench]:
        stats = calc_stats(sec_df[t], sec_df[bench], win)
        if stats:
            stats.update(calc_stats(sec_df[t], sec_df[bench], win)) # Re-calc for safety
            stats["Ticker"] = t
            stats["Name"] = get_display_name(t)
            # Add multi horizon
            stats.update(calc_returns_multi(sec_df[t]))
            results.append(stats)
            
    if not results:
        st.warning("NO DATA.")
        st.stop()
        
    df = pd.DataFrame(results)
    df["RS_z"] = zscore(df["RS"])
    df["Acc_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["Apex"] = 0.6*df["RS_z"] + 0.25*df["Acc_z"] - 0.15*df["DD_z"]
    df = df.sort_values("Apex", ascending=False).reset_index(drop=True)
    df["Verdict"] = df.apply(lambda r: "STRONG" if r["RS"]>0 and r["Accel"]>0 and r["Stable"]=="✅" else "WATCH" if r["RS"]>0 else "AVOID", axis=1)

    # 4. Table & AI
    c1, c2 = st.columns([1.6, 1])
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
    
    with c2:
        st.markdown(f"##### AI INTELLIGENCE: {top['Name']}")
        ai_txt = call_gemini(top["Ticker"], top["Name"], top.to_dict())
        st.markdown(f"<div class='ai-box'>{ai_txt}</div>", unsafe_allow_html=True)
        
    # 5. News
    st.markdown("---")
    st.subheader(f"INTELLIGENCE FEED: {top['Name']}")
    yn, gn = get_news_robust(top["Ticker"], top["Name"])
    
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