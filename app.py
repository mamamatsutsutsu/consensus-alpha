import os
import time
import math
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
# Page & CSS
# =========================
st.set_page_config(page_title="AlphaLens Sovereign", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* FORCE DARK THEME */
:root { --bg0:#0b0f14; --bg1:#0d1117; --bg2:#161b22; --bg3:#1c2128; --bd:#30363d; --tx:#e6edf3; --mut:#8b949e; }
html, body, .stApp { background: var(--bg1) !important; color: var(--tx) !important; font-family: "Inter", sans-serif !important; }
.deck { background: var(--bg2); padding: 15px; border-radius: 12px; border: 1px solid var(--bd); margin-bottom: 20px; }
.card { background: var(--bg3); border: 1px solid var(--bd); border-radius: 12px; padding: 15px; margin-bottom: 12px; }
.metric-box { background: var(--bg0); border-left: 4px solid var(--bd); border-radius: 8px; padding: 10px; margin-bottom: 10px; }
.status-green { border-left-color: #238636 !important; }
.status-yellow { border-left-color: #d29922 !important; }
.status-red { border-left-color: #da3633 !important; }
.ai-box { border: 1px dashed #7ee787; background: rgba(35,134,54,0.05); padding: 15px; border-radius: 10px; margin-top: 10px; }
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; border:1px solid; margin-right:5px; }
.badge-strong { border-color:#1f6feb; color:#58a6ff; background:rgba(31,111,235,0.1); }
.badge-watch { border-color:#d29922; color:#f0b429; background:rgba(210,153,34,0.1); }
.badge-avoid { border-color:#da3633; color:#f85149; background:rgba(218,54,51,0.1); }
.muted { color: var(--mut); font-size: 12px; }
h1, h2, h3 { color: var(--tx) !important; }
/* Login Form */
.login-box { max-width: 400px; margin: 100px auto; padding: 40px; background: var(--bg2); border-radius: 20px; border: 1px solid var(--bd); text-align: center; }
</style>
""", unsafe_allow_html=True)

# =========================
# 🔐 AUTHENTICATION & API SETUP
# =========================
# 1. Get Secrets (Support both names)
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

# 2. Google Gemini Setup
try:
    import google.generativeai as genai
    HAS_GENAI = True
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
except ImportError:
    HAS_GENAI = False

# 3. Simple Login Logic
def check_password():
    """Returns True if user is authenticated."""
    # If no password set in secrets, assume open access
    if not APP_PASS:
        return True
        
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Login UI
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔒 AlphaLens Access")
    pwd = st.text_input("Enter Access Password", type="password", key="login_pwd")
    if st.button("Unlock", type="primary", use_container_width=True):
        if pwd == APP_PASS:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.markdown("</div>", unsafe_allow_html=True)
    return False

# =========================
# STOP HERE IF NOT LOGGED IN
# =========================
if not check_password():
    st.stop()

# =========================
# Configuration
# =========================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

US_BENCH, JP_BENCH = "SPY", "1306.T"

US_SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Cons. Disc": "XLY",
    "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE", "Materials": "XLB",
    "Utilities": "XLU", "Real Estate": "XLRE"
}
JP_TOPIX17_ETF = {
    "エネルギー": "1617.T", "建設・資材": "1618.T", "素材・化学": "1619.T", "産業機械": "1620.T",
    "自動車・輸送": "1621.T", "商社・小売": "1622.T", "銀行": "1623.T", "金融（除銀行）": "1624.T",
    "不動産": "1625.T", "情報通信": "1626.T", "サービス": "1627.T", "電力・ガス": "1628.T",
    "電機・精密": "1631.T", "医薬品": "1632.T", "食品": "1633.T"
}

# --- UNIVERSE DEFINITION ---
US_STOCKS = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTU","IBM","AMD","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","SNPS","CDNS","NOW","PANW","CRWD","ANET"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","VRTX","BMY","GILD","PFE","REGN","SYK","BSX","MDT","ZTS","HCA","CVS","CI","ELV"],
    "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","PNC","USB","TFC","MMC","AIG","MET","PRU","AFL","CB","ICE","SPGI","V","MA","BRK-B"],
    "Cons. Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ROST","GM","F","MAR","HLT","EBAY","CMG","YUM","LULU","DHI","LEN","RCL","ABNB","ORLY","AZO"],
    "Cons. Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","KR","TGT","EL","HSY","STZ","KDP","WBA","MNST"],
    "Industrials": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","NSC","WM","FDX","NOC","GD","PCAR","ROK","CSX","ODFL"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","HAL","BKR","DVN","HES","APA","FANG","WMB","TRGP","OKE"],
    "Materials": ["LIN","APD","SHW","ECL","FCX","NEM","DOW","DD","NUE","VMC","MLM","ALB","CF","MOS","IP","CTVA"],
    "Utilities": ["NEE","DUK","SO","EXC","AEP","SRE","XEL","D","ED","PEG","EIX","PCG","AWK","WEC","ES","PPL"],
    "Real Estate": ["AMT","PLD","CCI","EQIX","SPG","O","PSA","WELL","DLR","AVB","EQR","VTR","IRM","VICI","SBAC"],
}

JP_STOCKS = {
    "電機・精密": ["8035.T","6857.T","6146.T","6920.T","6723.T","6758.T","6501.T","6981.T","6954.T","7741.T","6702.T","6503.T","6752.T","7735.T","6861.T","6971.T","6645.T"],
    "銀行": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T","8331.T","8354.T"],
    "自動車・輸送": ["7203.T","7267.T","6902.T","7201.T","7269.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T"],
    "商社・小売": ["8001.T","8031.T","8058.T","8053.T","8002.T","8015.T","3382.T","9983.T","8267.T","2914.T","7453.T","3092.T"],
    "情報通信": ["9432.T","9433.T","9434.T","9984.T","4689.T","6098.T","4755.T","4385.T","9613.T","9602.T","2413.T","3659.T"],
    "素材・化学": ["4063.T","4452.T","4005.T","4188.T","4901.T","4911.T","3407.T","4021.T","4631.T","3402.T"],
    "医薬品": ["4502.T","4568.T","4519.T","4503.T","4507.T","4523.T","4578.T","4151.T"],
    "産業機械": ["6301.T","7011.T","7012.T","6367.T","6273.T","6113.T","6473.T","6326.T"],
    "エネルギー": ["1605.T","5020.T","9501.T","9503.T","9531.T","9532.T"],
    "金融（除銀行）": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T"],
    "不動産": ["8801.T","8802.T","8830.T","3289.T","3003.T"],
    "鉄鋼・非鉄": ["5401.T","5411.T","5713.T","5406.T","5711.T","5802.T"],
    "サービス": ["4661.T","9735.T","4324.T","2127.T","6028.T","2412.T"],
    "食品": ["2801.T","2802.T","2269.T","2502.T","2503.T","2201.T"],
}

MARKETS = {
    "🇺🇸 US": {"bench": US_BENCH, "name": "S&P 500", "sectors": US_SECTOR_ETF, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": JP_BENCH, "name": "TOPIX", "sectors": JP_TOPIX17_ETF, "stocks": JP_STOCKS},
}

# --- NAME DATABASE ---
NAME_DB = {
    "SPY":"S&P500 ETF","1306.T":"TOPIX ETF","XLK":"Tech ETF","XLV":"Health ETF","XLF":"Fin ETF","XLY":"Cons Disc","XLP":"Staples","XLI":"Industrials","XLE":"Energy","XLB":"Materials","XLU":"Utilities","XLRE":"Real Estate",
    "1617.T":"エネ・資源","1618.T":"建設・資材","1619.T":"素材・化学","1620.T":"産業機械","1621.T":"自動車・輸送","1622.T":"商社・小売","1623.T":"銀行","1624.T":"金融(除銀行)","1625.T":"不動産","1626.T":"情報通信","1627.T":"サービス","1628.T":"電力・ガス","1631.T":"電機・精密","1632.T":"医薬品","1633.T":"食品",
    # JP
    "8035.T":"東京エレク","6857.T":"アドバンテ","6146.T":"ディスコ","6920.T":"レーザーテク","6723.T":"ルネサス","6758.T":"ソニーG","6501.T":"日立","6981.T":"村田製","6954.T":"ファナック","7741.T":"HOYA","6702.T":"富士通","6503.T":"三菱電機",
    "8306.T":"三菱UFJ","8316.T":"三井住友","8411.T":"みずほ","7203.T":"トヨタ","7267.T":"ホンダ","6902.T":"デンソー","7201.T":"日産","8001.T":"伊藤忠","8031.T":"三井物産","8058.T":"三菱商事","9983.T":"ファストリ","3382.T":"7&i","8267.T":"イオン",
    "9432.T":"NTT","9433.T":"KDDI","9984.T":"SBG","4689.T":"LINEヤフー","6098.T":"リクルート","4063.T":"信越化学","4452.T":"花王","4502.T":"武田薬品","4568.T":"第一三共","4519.T":"中外製薬","1605.T":"INPEX","5020.T":"ENEOS","8801.T":"三井不","8802.T":"三菱地所",
    "5401.T":"日本製鉄","6301.T":"コマツ","7011.T":"三菱重工","4661.T":"OLC","9735.T":"セコム","2801.T":"キッコーマン","2802.T":"味の素","2914.T":"JT","7974.T":"任天堂","7182.T":"ゆうちょ","8591.T":"オリックス",
    # US
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon","TSLA":"Tesla","GOOGL":"Alphabet","META":"Meta","AVGO":"Broadcom","ORCL":"Oracle","CRM":"Salesforce","ADBE":"Adobe","AMD":"AMD","QCOM":"Qualcomm","NFLX":"Netflix","INTC":"Intel",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","JPM":"JPMorgan","BAC":"BofA","WFC":"Wells Fargo","V":"Visa","MA":"Mastercard","BRK-B":"Berkshire","GS":"Goldman",
    "PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","HD":"Home Depot","MCD":"McDonalds","NKE":"Nike","SBUX":"Starbucks",
    "XOM":"Exxon","CVX":"Chevron","GE":"GE Aero","CAT":"Caterpillar","BA":"Boeing","LMT":"Lockheed","RTX":"RTX Corp","DE":"Deere",
    "LIN":"Linde","NEE":"NextEra","AMT":"Amer. Tower","PLD":"Prologis","DIS":"Disney","CMCSA":"Comcast","T":"AT&T","VZ":"Verizon",
}

# =========================
# Core Logic
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t and isinstance(t, str)]))
    frames = []
    chunk_size = 80
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            raw = yf.download(" ".join(chunk), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not raw.empty: frames.append(raw)
        except: continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

@st.cache_data(ttl=86400)
def get_name(ticker: str) -> str:
    return NAME_DB.get(ticker, ticker)

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

def zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

def vol_annualized(series: pd.Series, win: int) -> float:
    s = series.tail(win + 1)
    if len(s) < win + 1: return np.nan
    r = s.pct_change().dropna()
    if r.empty: return np.nan
    return float(r.std(ddof=0) * np.sqrt(252) * 100.0)

def calc_stats(s: pd.Series, b: pd.Series, win: int) -> Dict:
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    s_short, b_short = s.tail(6).dropna(), b.tail(6).dropna()
    stable = "⚠️"
    if len(s_short)==6 and len(b_short)==6:
        rs_short = (s_short.iloc[-1]/s_short.iloc[0]-1) - (b_short.iloc[-1]/b_short.iloc[0]-1)
        if np.sign(rs_short) == np.sign(p_ret - b_ret): stable = "✅"
    
    vol = vol_annualized(s, win)
    
    return {"RS": p_ret - b_ret, "Accel": accel, "MaxDD": dd, "Stable": stable, "Ret": p_ret, "Vol": vol}

def audit_gatekeeper(expected: List[str], close_df: pd.DataFrame, win: int) -> Dict:
    exp = len(expected)
    if close_df.empty: return {"computable": 0, "expected": exp, "computable_list": [], "mode_date": None}
    
    present = [t for t in expected if t in close_df.columns]
    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    
    computable = []
    for t in present:
        if last_dates[t] == mode_date and close_df[t].tail(win+1).notna().sum() >= win+1:
            computable.append(t)
            
    return {"computable": len(computable), "expected": exp, "computable_list": computable, "mode_date": mode_date}

# =========================
# AI & News Logic
# =========================
@st.cache_data(ttl=1800)
def get_yahoo_news(ticker: str) -> List[dict]:
    try: return yf.Ticker(ticker).news[:5]
    except: return []

@st.cache_data(ttl=1800)
def get_google_rss(query: str) -> List[dict]:
    try:
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=5) as resp:
            root = ET.fromstring(resp.read())
        return [{"title": i.findtext("title"), "link": i.findtext("link"), "source": i.find("source").text} for i in root.findall(".//item")[:5]]
    except: return []

def call_gemini(ticker: str, name: str, stats: Dict) -> str:
    if GEMINI_KEY and HAS_GENAI:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            あなたはプロの投資アナリストです。以下のデータに基づき、3つの視点（モメンタム、リスク、マクロ）で議論し、最終投資判断を日本語で下してください。
            銘柄: {name} ({ticker})
            RS: {stats['RS']:.2f}%, Accel: {stats['Accel']:.2f}
            MaxDD: {stats['MaxDD']:.2f}%, Vol: {stats['Vol']:.2f}%
            出力:
            【モメンタム】...
            【リスク】...
            【結論】(強気/中立/弱気) 理由1行
            """
            return model.generate_content(prompt).text
        except Exception as e:
            return f"AI Error: {str(e)}\n(ルールベース分析を表示)"
    
    # Fallback
    view = "強気" if stats['RS']>0 and stats['Accel']>0 and stats['MaxDD']<15 else "弱気" if stats['RS']<0 else "中立"
    return f"※AIキー未設定。ルールベース分析:\nトレンドは{'強い' if stats['RS']>0 else '弱い'}、リスクは{'許容範囲' if stats['MaxDD']<15 else '高め'}。\n判定: {view}"

# =========================
# Main UI
# =========================
def main():
    if not GEMINI_KEY:
        st.warning("⚠️ GEMINI_API_KEY / GOOGLE_API_KEY が未設定です。AI機能は制限されます。")

    st.markdown("<h2 class='brand'>AlphaLens Sovereign</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='deck'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
        with c1: market_key = st.selectbox("Market", list(MARKETS.keys()))
        with c2: lookback_key = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
        with c3: st.caption(f"Fetch: {FETCH_PERIOD}"); st.progress(100)
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
        with st.spinner("Market Data Sync..."):
            raw = fetch_bulk_cached(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.core_df
    audit = audit_gatekeeper(core_tickers, core_df, win)
    
    if bench not in audit["computable_list"]:
        st.error(f"Critical: Benchmark {bench} missing/insufficient data.")
        st.stop()
        
    st.markdown(f"<div class='cert'>GATEKEEPER: PASSED | Health {audit['computable']}/{audit['expected']}</div>", unsafe_allow_html=True)

    # 2. Market Overview
    b_stats = calc_stats(core_df[bench], core_df[bench], win)
    
    sec_rows = []
    for s_name, s_ticker in m_cfg["sectors"].items():
        if s_ticker in audit["computable_list"]:
            stats = calc_stats(core_df[s_ticker], core_df[bench], win)
            if stats: sec_rows.append({"Sector": s_name, "Ticker": s_ticker, "Return": stats["Ret"], "RS": stats["RS"]})
            
    sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
    sdf_chart = pd.concat([sdf, pd.DataFrame([{"Sector": "MARKET", "Ticker": bench, "Return": b_stats["Ret"], "RS": 0}])], ignore_index=True).sort_values("RS")
    
    c1, c2 = st.columns([1, 3])
    with c1: st.metric(m_cfg["name"], f"{b_stats['Ret']:+.2f}%")
    with c2: st.info(f"**Market Insight**: {m_cfg['name']} is moving {b_stats['Ret']:+.2f}%. {len(sdf[sdf['RS']>0])} sectors outperforming.")

    fig = px.bar(sdf_chart, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn", title=f"Relative Strength vs {bench}")
    fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # 3. Drill Down
    click_sec = event["selection"]["points"][0]["y"] if event and event.get("selection", {}).get("points") else None
    cols = st.columns(6)
    btn_sec = None
    for i, s in enumerate(m_cfg["sectors"].keys()):
        if cols[i%6].button(s, key=f"btn_{s}", use_container_width=True): btn_sec = s
            
    target_sector = btn_sec or click_sec or st.session_state.get("target_sector", list(m_cfg["sectors"].keys())[0])
    st.session_state.target_sector = target_sector
    
    st.markdown("---")
    st.subheader(f"🔍 Drilldown: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    sec_cache_key = f"{market_key}_{target_sector}_{datetime.now().hour}"
    if sec_cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner(f"Analyzing stocks..."):
            raw_s = fetch_bulk_cached(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = sec_cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_gatekeeper(full_list, sec_df, win)
    
    if bench not in s_audit["computable_list"]:
        st.warning("Benchmark missing in sector data.")
        st.stop()
        
    results = []
    for t in [x for x in s_audit["computable_list"] if x != bench]:
        stats = calc_stats(sec_df[t], sec_df[bench], win)
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
    
    df["Verdict"] = df.apply(lambda r: "STRONG" if r["RS"]>0 and r["Accel"]>0 and r["Stable"]=="✅" else "WATCH" if r["RS"]>0 else "AVOID", axis=1)

    # 4. Analysis
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("##### 🏆 Leaderboard")
        event_table = st.dataframe(
            df[["Name", "Ticker", "Verdict", "Apex", "RS", "Accel", "MaxDD", "Stable"]],
            column_config={
                "Apex": st.column_config.NumberColumn(format="%.2f"),
                "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
                "Accel": st.column_config.NumberColumn(format="%.2f"),
                "MaxDD": st.column_config.NumberColumn(format="%.2f%%"),
            },
            hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
        )
        
    sel_rows = event_table.selection.get("rows", [])
    top = df.iloc[sel_rows[0]] if sel_rows else df.iloc[0]

    with c2:
        st.markdown(f"##### 🤖 AI Analysis: {top['Name']}")
        ai_msg = call_gemini(top["Ticker"], top["Name"], {"RS":top["RS"], "Accel":top["Accel"], "MaxDD":top["MaxDD"], "Stable":top["Stable"], "Vol":top["Vol"]})
        st.markdown(f"<div class='ai-box'><div style='font-size:13px; white-space: pre-wrap;'>{ai_msg}</div></div>", unsafe_allow_html=True)
        
    # 5. News
    st.markdown("---")
    st.subheader(f"📰 News: {top['Name']}")
    
    n1, n2 = st.columns(2)
    with n1:
        st.caption("Yahoo Finance")
        ynews = get_yahoo_news(top["Ticker"])
        if not ynews: st.write("No news.")
        for n in ynews: st.markdown(f"- [{n['title']}]({n['link']})")
            
    with n2:
        st.caption("Google News (RSS)")
        gnews = get_google_rss(f"{top['Name']} 株")
        if not gnews: st.write("No news.")
        for n in gnews: st.markdown(f"- [{n['title']}]({n['link']}) <span class='muted'>({n['source']})</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()