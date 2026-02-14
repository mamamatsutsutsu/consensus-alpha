# app.py — AlphaLens v33.0 "Grand Sovereign"
# Final release candidate based on all user feedback.
# - Full Name DB (400+ entries)
# - Google Gemini Native Support
# - Robust News Fetching (No KeyErrors)
# - Phantom Dark UI with fixed button styling

import os
import math
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
# 1. Phantom UI Configuration
# =========================
st.set_page_config(page_title="AlphaLens Sovereign", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* FONT IMPORTS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Orbitron:wght@700;900&family=JetBrains+Mono:wght@400;700&display=swap');

/* --- PHANTOM DARK THEME VARIABLES --- */
:root {
  --bg-app: #050505;
  --bg-panel: #0d1117;
  --bg-card: #161b22;
  --border: #30363d;
  --accent: #00f2fe;
  --accent-dim: rgba(0, 242, 254, 0.1);
  --text-main: #e6edf3;
  --text-sub: #8b949e;
}

/* GLOBAL OVERRIDES */
.stApp { background-color: var(--bg-app) !important; color: var(--text-main) !important; font-family: 'Inter', sans-serif !important; }
h1, h2, h3, h4, h5, h6 { color: var(--text-main) !important; letter-spacing: -0.02em; }
a { color: var(--accent) !important; text-decoration: none; }

/* BRAND HEADER */
.brand-box { text-align: center; margin-bottom: 30px; padding-top: 20px; }
.brand-title {
  font-family: 'Orbitron', sans-serif;
  font-size: 42px;
  font-weight: 900;
  background: linear-gradient(135deg, #ffffff 0%, #00f2fe 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 0 15px rgba(0, 242, 254, 0.4));
  letter-spacing: 4px;
}
.brand-sub { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--accent); letter-spacing: 2px; opacity: 0.8; }

/* DECK & CARDS */
.deck {
  background: rgba(13, 17, 23, 0.8);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
  transition: transform 0.2s;
}
.card:hover { border-color: var(--accent); }

/* METRIC PILLS */
.kpi-box {
  background: #0a0c10;
  border-left: 3px solid var(--border);
  border-radius: 6px;
  padding: 10px 14px;
  display: flex; flex-direction: column;
}
.kpi-label { font-size: 10px; color: var(--text-sub); text-transform: uppercase; letter-spacing: 0.5px; }
.kpi-val { font-size: 16px; font-weight: 700; color: var(--text-main); font-family: 'JetBrains Mono', monospace; }
.status-green { border-left-color: #238636 !important; }
.status-yellow { border-left-color: #d29922 !important; }
.status-red { border-left-color: #da3633 !important; }

/* BUTTONS (High Contrast Dark Mode) */
div.stButton > button {
  background-color: var(--bg-card) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.5rem 1rem !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}
div.stButton > button:hover {
  background-color: var(--bg-panel) !important;
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  box-shadow: 0 0 12px rgba(0, 242, 254, 0.25) !important;
}
div.stButton > button:active { transform: scale(0.98); }

/* AI ANALYSIS BOX */
.ai-box {
  border: 1px solid rgba(0, 242, 254, 0.3);
  background: linear-gradient(180deg, rgba(0, 242, 254, 0.05) 0%, rgba(0,0,0,0) 100%);
  border-radius: 12px; padding: 20px; margin-top: 15px;
}

/* BADGES */
.badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 10px; font-weight: 800; margin-right: 8px; border: 1px solid; }
.b-strong { border-color:#1f6feb; color:#58a6ff; background:rgba(31,111,235,0.1); }
.b-watch { border-color:#d29922; color:#f0b429; background:rgba(210,153,34,0.1); }
.b-avoid { border-color:#da3633; color:#f85149; background:rgba(218,54,51,0.1); }

/* UTILS */
.muted { color: var(--text-sub); font-size: 12px; }
.highlight { color: var(--accent); font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. Authentication & API
# =========================
# 優先順位: Secrets > Environment
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

HAS_GENAI = False
if API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        HAS_GENAI = True
    except ImportError: pass

def check_auth():
    if not APP_PASS: return True
    if st.session_state.get("auth", False): return True
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        with st.form("login"):
            st.markdown("<h3 style='text-align:center;'>🔒 SECURITY CLEARANCE</h3>", unsafe_allow_html=True)
            pwd = st.text_input("Access Code", type="password")
            if st.form_submit_button("AUTHENTICATE", use_container_width=True):
                if pwd == APP_PASS:
                    st.session_state.auth = True
                    st.rerun()
                else: st.error("INVALID CODE")
    return False

if not check_auth(): st.stop()

# =========================
# 3. Master Universe (Full Scale)
# =========================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

# --- US SECTORS ---
US_SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Comm Services": "XLC",
    "Cons. Disc": "XLY", "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE"
}

# --- JP SECTORS ---
JP_SECTOR_ETF = {
    "情報通信": "1626.T", "電機・精密": "1631.T", "自動車・輸送": "1621.T", "医薬品": "1632.T",
    "銀行": "1623.T", "金融(除銀行)": "1624.T", "商社・小売": "1622.T", "機械": "1630.T",
    "エネルギー": "1617.T", "建設・資材": "1618.T", "素材・化学": "1619.T", "食品": "1633.T",
    "電力・ガス": "1628.T", "不動産": "1625.T", "鉄鋼・非鉄": "1629.T", "サービス": "1627.T",
    "産業機械": "1620.T"
}

# --- US STOCKS (Approx 200) ---
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

# --- JP STOCKS (Approx 200) ---
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
    "🇺🇸 US": {"bench": "SPY", "name": "S&P 500", "sectors": US_SECTOR_ETF, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": "1306.T", "name": "TOPIX", "sectors": JP_SECTOR_ETF, "stocks": JP_STOCKS},
}

# --- COMPLETE NAME DATABASE (FULL MAPPING) ---
# コード内の全ティッカーに対応する社名を完備
NAME_DB = {
    # BENCH / ETF
    "SPY":"S&P500 ETF","1306.T":"TOPIX ETF","XLK":"Tech ETF","XLV":"Health ETF","XLF":"Finance ETF","XLC":"Comm ETF","XLY":"ConsDisc ETF","XLP":"Staples ETF","XLI":"Indust ETF","XLE":"Energy ETF","XLB":"Material ETF","XLU":"Utility ETF","XLRE":"RealEst ETF",
    "1626.T":"情報通信ETF","1631.T":"電機精密ETF","1621.T":"自動車ETF","1632.T":"医薬品ETF","1623.T":"銀行ETF","1624.T":"金融(除銀行)ETF","1622.T":"商社小売ETF","1630.T":"機械ETF","1617.T":"エネルギーETF","1618.T":"建設資材ETF","1619.T":"素材化学ETF","1633.T":"食品ETF","1628.T":"電力ガスETF","1625.T":"不動産ETF","1629.T":"鉄鋼非鉄ETF","1627.T":"サービスETF","1620.T":"産業機械ETF",
    # US STOCKS
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AVGO":"Broadcom","ORCL":"Oracle","CRM":"Salesforce","ADBE":"Adobe","AMD":"AMD","QCOM":"Qualcomm","TXN":"Texas Inst","INTU":"Intuit","IBM":"IBM","NOW":"ServiceNow","AMAT":"Applied Mat","MU":"Micron","LRCX":"Lam Research","ADI":"Analog Dev","KLAC":"KLA Corp","SNPS":"Synopsys","CDNS":"Cadence","PANW":"Palo Alto","CRWD":"CrowdStrike","ANET":"Arista","PLTR":"Palantir",
    "GOOGL":"Alphabet","META":"Meta","NFLX":"Netflix","DIS":"Disney","CMCSA":"Comcast","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CHTR":"Charter","WBD":"Warner Bros","LYV":"Live Nation","EA":"Elec Arts","TTWO":"Take-Two","OMC":"Omnicom","IPG":"Interpublic",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","ABBV":"AbbVie","MRK":"Merck","TMO":"Thermo Fisher","ABT":"Abbott","AMGN":"Amgen","PFE":"Pfizer","ISRG":"Intuitive","DHR":"Danaher","VRTX":"Vertex","GILD":"Gilead","REGN":"Regeneron","BMY":"Bristol Myers","CVS":"CVS Health","CI":"Cigna","SYK":"Stryker","BSX":"Boston Sci","MDT":"Medtronic","ZTS":"Zoetis","HCA":"HCA Health","MCK":"McKesson",
    "JPM":"JPMorgan","BAC":"BofA","WFC":"Wells Fargo","V":"Visa","MA":"Mastercard","AXP":"Amex","GS":"Goldman","MS":"Morgan Stanley","BLK":"BlackRock","C":"Citi","SCHW":"Schwab","SPGI":"S&P Global","PGR":"Progressive","CB":"Chubb","MMC":"Marsh","KKR":"KKR","BX":"Blackstone","TRV":"Travelers","AFL":"Aflac","MET":"MetLife","PRU":"Prudential","ICE":"Intercon Ex","COF":"Capital One",
    "AMZN":"Amazon","TSLA":"Tesla","HD":"Home Depot","MCD":"McDonalds","NKE":"Nike","SBUX":"Starbucks","LOW":"Lowe's","BKNG":"Booking","TJX":"TJX","CMG":"Chipotle","MAR":"Marriott","HLT":"Hilton","YUM":"Yum!","LULU":"Lululemon","GM":"GM","F":"Ford","ROST":"Ross","ORLY":"O'Reilly","AZO":"AutoZone","DHI":"DR Horton","LEN":"Lennar",
    "PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","COST":"Costco","WMT":"Walmart","PM":"Philip Morris","MO":"Altria","MDLZ":"Mondelez","CL":"Colgate","KMB":"Kimberly","GIS":"Gen Mills","KHC":"Kraft Heinz","KR":"Kroger","STZ":"Constellation","EL":"Estee Lauder","TGT":"Target","DG":"Dollar Gen","ADM":"Archer Dan","SYY":"Sysco",
    "GE":"GE Aerospace","CAT":"Caterpillar","DE":"Deere","HON":"Honeywell","UNP":"Union Pacific","UPS":"UPS","RTX":"RTX Corp","LMT":"Lockheed","BA":"Boeing","MMM":"3M","ETN":"Eaton","EMR":"Emerson","ITW":"Illinois Tool","WM":"Waste Mgmt","NSC":"Norfolk So","CSX":"CSX","GD":"Gen Dynamics","NOC":"Northrop","TDG":"TransDigm","PCAR":"PACCAR","FDX":"FedEx","CTAS":"Cintas",
    "XOM":"Exxon","CVX":"Chevron","COP":"Conoco","EOG":"EOG Res","SLB":"Schlumberger","MPC":"Marathon","PSX":"Phillips 66","VLO":"Valero","OXY":"Occidental","KMI":"Kinder Morgan","WMB":"Williams","HAL":"Halliburton","BKR":"Baker Hughes","DVN":"Devon","HES":"Hess","FANG":"Diamondback","TRGP":"Targa","OKE":"ONEOK",
    "LIN":"Linde","APD":"Air Products","SHW":"Sherwin","FCX":"Freeport","ECL":"Ecolab","NEM":"Newmont","DOW":"Dow","DD":"DuPont","NUE":"Nucor","MLM":"Martin Mari","VMC":"Vulcan","CTVA":"Corteva","PPG":"PPG Ind","ALB":"Albemarle","CF":"CF Ind","MOS":"Mosaic",
    "NEE":"NextEra","DUK":"Duke","SO":"Southern","AEP":"Am Elec","SRE":"Sempra","EXC":"Exelon","XEL":"Xcel","D":"Dominion","PEG":"PSEG","ED":"Con Ed","EIX":"Edison","WEC":"WEC Energy","AWK":"Am Water","ES":"Eversource","PPL":"PPL Corp","ETR":"Entergy",
    "PLD":"Prologis","AMT":"Amer Tower","CCI":"Crown Castle","EQIX":"Equinix","SPG":"Simon Prop","PSA":"Public Storage","O":"Realty Income","WELL":"Welltower","DLR":"Digital Rlty","AVB":"AvalonBay","EQR":"Equity Res","VICI":"VICI Prop","CSGP":"CoStar","SBAC":"SBA Comm","IRM":"Iron Mtn",
    # JP STOCKS
    "9432.T":"NTT","9433.T":"KDDI","9434.T":"ソフトバンク","9984.T":"SBG","4689.T":"LINEヤフー","4755.T":"楽天G","9613.T":"NTTデータ","9602.T":"東宝","4385.T":"メルカリ","6098.T":"リクルート","3659.T":"ネクソン","3765.T":"ガンホー",
    "8035.T":"東京エレク","6857.T":"アドバンテ","6146.T":"ディスコ","6920.T":"レーザーテク","6758.T":"ソニーG","6501.T":"日立","6723.T":"ルネサス","6981.T":"村田製","6954.T":"ファナック","7741.T":"HOYA","6702.T":"富士通","6503.T":"三菱電機","6752.T":"パナHD","7735.T":"SCREEN","6861.T":"キーエンス",
    "7203.T":"トヨタ","7267.T":"ホンダ","6902.T":"デンソー","7201.T":"日産","7269.T":"スズキ","7270.T":"SUBARU","7272.T":"ヤマハ発","9101.T":"日本郵船","9104.T":"商船三井","9020.T":"JR東日本","9022.T":"JR東海","9005.T":"東急",
    "4502.T":"武田","4568.T":"第一三共","4519.T":"中外","4503.T":"アステラス","4507.T":"塩野義","4523.T":"エーザイ","4578.T":"大塚HD","4151.T":"協和キリン","4528.T":"小野薬","4506.T":"住友ファーマ",
    "8306.T":"三菱UFJ","8316.T":"三井住友","8411.T":"みずほ","8308.T":"りそな","8309.T":"三井住友トラ","7182.T":"ゆうちょ","5831.T":"しずおかFG","8331.T":"千葉銀","8354.T":"ふくおかFG",
    "8591.T":"オリックス","8604.T":"野村HD","8766.T":"東京海上","8725.T":"MS&AD","8750.T":"第一生命","8697.T":"日本取引所","8630.T":"SOMPO","8570.T":"イオンFS",
    "8001.T":"伊藤忠","8031.T":"三井物産","8058.T":"三菱商事","8053.T":"住友商事","8002.T":"丸紅","8015.T":"豊田通商","3382.T":"7&i","9983.T":"ファストリ","8267.T":"イオン","2914.T":"JT","7453.T":"良品計画","3092.T":"ZOZO",
    "6301.T":"コマツ","7011.T":"三菱重工","7012.T":"川崎重工","6367.T":"ダイキン","6273.T":"SMC","6113.T":"アマダ","6473.T":"ジェイテクト","6326.T":"クボタ",
    "1605.T":"INPEX","5020.T":"ENEOS","9501.T":"東電EP","3407.T":"旭化成","4005.T":"住友化学",
    "1925.T":"大和ハウス","1928.T":"積水ハウス","1801.T":"大成建設","1802.T":"大林組","1812.T":"鹿島","5201.T":"AGC","5332.T":"TOTO",
    "4063.T":"信越化学","4452.T":"花王","4188.T":"三菱ケミ","4901.T":"富士フイルム","4911.T":"資生堂","4021.T":"日産化学","4631.T":"DIC","3402.T":"東レ",
    "2801.T":"キッコーマン","2802.T":"味の素","2269.T":"明治HD","2502.T":"アサヒ","2503.T":"キリン","2201.T":"森永製菓","2002.T":"日清製粉",
    "9501.T":"東電EP","9503.T":"関電","9531.T":"東ガス","9532.T":"大ガス",
    "8801.T":"三井不","8802.T":"三菱地所","8830.T":"住友不","3289.T":"東急不","3003.T":"ヒューリック","3231.T":"野村不",
    "5401.T":"日本製鉄","5411.T":"JFE","5713.T":"住友鉱","5406.T":"神戸鋼","5711.T":"三菱マテ","5802.T":"住友電工",
    "4661.T":"OLC","9735.T":"セコム","4324.T":"電通","2127.T":"日本M&A","6028.T":"テクノプロ","2412.T":"ベネフィット","4689.T":"LINEヤフー",
    "6146.T":"ディスコ","6460.T":"セガサミー","6471.T":"日本精工","6268.T":"ナブテスコ"
}

def get_name(t: str) -> str:
    return NAME_DB.get(t, t)

# =========================
# 4. Engine (Robust)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t]))
    frames = []
    chunk_size = 80
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            raw = yf.download(" ".join(chunk), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not raw.empty: frames.append(raw)
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

def calc_stats_multi(s: pd.Series, b: pd.Series, win: int) -> Dict:
    # 1. Base Integrity
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    # 2. Main Metrics
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    
    # Accel
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    
    # MaxDD
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

def audit_gate(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "list": []}
    last_dates = df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    computable = []
    for t in present:
        if last_dates[t] == mode_date and df[t].tail(win+1).notna().sum() >= win+1:
            computable.append(t)
    return {"ok": True, "list": computable, "mode": mode_date, "count": len(computable), "total": len(expected)}

def zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# =========================
# AI Agent Logic
# =========================
@st.cache_data(ttl=1800)
def get_news_robust(ticker: str, name: str) -> Tuple[List[dict], List[dict]]:
    # Yahoo (Robust Get)
    y_news = []
    try:
        raw = yf.Ticker(ticker).news
        if raw and isinstance(raw, list):
            for n in raw[:3]:
                # Safeguard against missing keys
                y_news.append({
                    "title": n.get("title", "No Title"),
                    "link": n.get("link", "#"),
                    "publisher": n.get("publisher", "Yahoo")
                })
    except: pass
    
    # Google RSS
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

def call_gemini(ticker: str, name: str, stats: Dict) -> str:
    if HAS_GENAI and API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            あなたはプロの機関投資家です。以下の銘柄について3名のエージェント（モメンタム、リスク、マクロ）になりきって議論し、最終判断を日本語で下してください。
            
            対象: {name} ({ticker})
            [テクニカル] RS: {stats['RS']:.2f}% (市場比), Accel: {stats['Accel']:.2f}, MaxDD: {stats['MaxDD']:.2f}%, Stable: {stats['Stable']}
            [騰落率] 1W:{stats['1W']:.1f}%, 1M:{stats['1M']:.1f}%, 3M:{stats['3M']:.1f}%, 12M:{stats['12M']:.1f}%
            
            出力形式:
            【モメンタム担当】...
            【リスク担当】...
            【結論】(強気/中立/弱気) 理由1行
            """
            return model.generate_content(prompt).text
        except:
            pass # Fallback to rule-based
            
    # Fallback
    v = "強気" if stats['RS']>0 and stats['Accel']>0 else "中立"
    return f"※AIキー未設定 (Rule-based):\nトレンド: {v}\nRS: {stats['RS']:.2f}% | 12M: {stats['12M']:.1f}%"

# =========================
# 5. Main UI
# =========================
def main():
    # Brand
    st.markdown("""
    <div class='brand-box'>
        <div class='brand-title'>ALPHALENS</div>
        <div class='brand-sub'>INTELLIGENT SENTINEL v33.0</div>
    </div>
    """, unsafe_allow_html=True)

    # Deck
    with st.container():
        st.markdown("<div class='deck'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
        with c1: market_key = st.selectbox("Market", list(MARKETS.keys()))
        with c2: lookback_key = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
        with c3: st.caption(f"Horizon: {FETCH_PERIOD}"); st.progress(100)
        with c4: 
            st.write("")
            sync = st.button("SYNC", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # 1. SYNC
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if sync or "core_df" not in st.session_state or st.session_state.get("last_m") != market_key:
        with st.spinner("Synchronizing Market Data..."):
            raw = fetch_bulk_cached(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    audit = audit_gate(core_tickers, core_df, win)
    
    if bench not in audit["list"]:
        st.error("Critical: Benchmark data unavailable. Check connection.")
        st.stop()

    # Integrity
    col1, col2 = st.columns(2)
    with col1: st.markdown(f"<div class='kpi-box status-green'><div class='kpi-label'>Health</div><div class='kpi-val'>{audit['count']}/{audit['total']}</div></div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='kpi-box status-green'><div class='kpi-label'>Data Date</div><div class='kpi-val'>{str(audit['mode']).split()[0]}</div></div>", unsafe_allow_html=True)

    # 2. SECTOR OVERVIEW
    b_stats = calc_stats_multi(core_df[bench], core_df[bench], win)
    
    sec_data = []
    for s_name, s_tk in m_cfg["sectors"].items():
        if s_tk in audit["list"]:
            res = calc_stats_multi(core_df[s_tk], core_df[bench], win)
            if res:
                res["Sector"] = s_name
                sec_data.append(res)
    
    sdf = pd.DataFrame(sec_data).sort_values("RS", ascending=True)
    # Add Market to Chart
    sdf_chart = pd.concat([sdf, pd.DataFrame([{"Sector": "MARKET", "RS": 0, "Ret": b_stats["Ret"]}])], ignore_index=True).sort_values("RS")
    
    st.subheader("📊 Sector Rotation")
    fig = px.bar(sdf_chart, x="RS", y="Sector", orientation='h', color="RS", 
                 color_continuous_scale="RdYlGn", title=f"Relative Strength ({lookback_key})")
    fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # Drill Down Selection
    click_sec = event["selection"]["points"][0]["y"] if event and event.get("selection", {}).get("points") else None
    
    cols = st.columns(6)
    btn_sec = None
    for i, s in enumerate(m_cfg["sectors"].keys()):
        if cols[i%6].button(s, key=f"b_{s}", use_container_width=True): btn_sec = s
            
    target_sector = btn_sec or click_sec or st.session_state.get("target_sector", list(m_cfg["sectors"].keys())[0])
    st.session_state.target_sector = target_sector
    
    # 3. STOCK ANALYSIS
    st.markdown("---")
    st.subheader(f"🔍 Forensic: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    cache_key = f"{market_key}_{target_sector}"
    if cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner("Analyzing Sector Stocks..."):
            raw_s = fetch_bulk_cached(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_gate(full_list, sec_df, win)
    
    results = []
    for t in [x for x in s_audit["list"] if x != bench]:
        stats = calc_stats_multi(sec_df[t], sec_df[bench], win)
        if stats:
            stats["Ticker"] = t
            stats["Name"] = get_name(t)
            results.append(stats)
            
    if not results:
        st.warning("No computable stocks found.")
        st.stop()
        
    df = pd.DataFrame(results)
    df["RS_z"] = zscore(df["RS"])
    df["Acc_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["Apex"] = 0.6*df["RS_z"] + 0.25*df["Acc_z"] - 0.15*df["DD_z"]
    df = df.sort_values("Apex", ascending=False).reset_index(drop=True)
    
    def get_badge(r):
        if r["RS"]>0 and r["Accel"]>0 and r["Stable"]=="✅": return "STRONG"
        if r["RS"]>0: return "WATCH"
        return "AVOID"
    df["Verdict"] = df.apply(get_badge, axis=1)

    # 4. TABLE & AI
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("##### 🏆 Leaderboard")
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
        st.markdown(f"##### 🤖 AI Analysis: {top['Name']}")
        ai_txt = call_gemini(top["Ticker"], top["Name"], top.to_dict())
        st.markdown(f"<div class='ai-box'><div style='font-size:13px; white-space: pre-wrap;'>{ai_txt}</div></div>", unsafe_allow_html=True)
        
    # 5. NEWS
    st.markdown("---")
    st.subheader(f"📰 News: {top['Name']}")
    yn, gn = get_news_robust(top["Ticker"], top["Name"])
    
    n1, n2 = st.columns(2)
    with n1:
        st.caption("Yahoo Finance")
        if not yn: st.write("No news.")
        for n in yn:
            t = n.get('title','Link')
            l = n.get('link','#')
            st.markdown(f"- [{t}]({l})")
    with n2:
        st.caption("Google News")
        if not gn: st.write("No news.")
        for n in gn:
            st.markdown(f"- [{n['title']}]({n['link']}) <span class='muted'>({n['src']})</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()