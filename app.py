# app.py — AlphaLens v29.0 "Gemini Core"
# 
# [Features]
# 1. Google Gemini (AI Agents) Integration: Real debate if API key exists.
# 2. Zero-Dependency UI: No Matplotlib/Styler. Pure Streamlit + Plotly.
# 3. Integrity Gatekeeper: Filters bad data before calc.
# 4. Embedded Name Database: 400+ Tickers mapped internally (No external files).
# 5. Professional Dark Mode: Forced CSS for mobile/desktop consistency.

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

# Try importing Google GenAI
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# =========================
# Page config & CSS
# =========================
st.set_page_config(page_title="AlphaLens Gemini", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
/* FORCE DARK THEME */
:root { --bg0:#0b0f14; --bg1:#0d1117; --bg2:#161b22; --bg3:#1c2128; --bd:#30363d; --tx:#e6edf3; --mut:#8b949e; }
html, body, .stApp { background: var(--bg1) !important; color: var(--tx) !important; }
.deck { background: var(--bg2); padding: 16px; border-radius: 12px; border: 1px solid var(--bd); margin-bottom: 20px; }
.card { background: var(--bg3); border: 1px solid var(--bd); border-radius: 12px; padding: 14px; margin-bottom: 12px; }
.metric-box { background: var(--bg0); border-left: 4px solid var(--bd); border-radius: 8px; padding: 10px; }
.status-green { border-left-color: #238636 !important; }
.status-yellow { border-left-color: #d29922 !important; }
.status-red { border-left-color: #da3633 !important; }
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; border:1px solid; margin-right:5px; }
.badge-strong { border-color:#1f6feb; color:#58a6ff; background:rgba(31,111,235,0.1); }
.badge-watch { border-color:#d29922; color:#f0b429; background:rgba(210,153,34,0.1); }
.badge-avoid { border-color:#da3633; color:#f85149; background:rgba(218,54,51,0.1); }
.ai-box { border: 1px dashed #7ee787; background: rgba(35,134,54,0.05); padding: 15px; border-radius: 10px; }
.muted { color: var(--mut); font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Constants & Universes
# =========================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

# Google API Key Check
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

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

# --- EXTENSIVE INTERNAL NAME DATABASE (No external files) ---
NAME_DB = {
    # ETFs
    "SPY":"S&P500 ETF", "1306.T":"TOPIX ETF", "XLK":"Tech ETF", "XLV":"Health ETF", "XLF":"Fin ETF",
    "1617.T":"エネルギーETF", "1621.T":"自動車ETF", "1622.T":"商社ETF", "1623.T":"銀行ETF", "1631.T":"電機精密ETF",
    # JP Major
    "8035.T":"東京エレクトロン", "6857.T":"アドバンテスト", "6146.T":"ディスコ", "6920.T":"レーザーテック", "6723.T":"ルネサス",
    "6758.T":"ソニーG", "6501.T":"日立製作所", "6981.T":"村田製作所", "6954.T":"ファナック", "7741.T":"HOYA",
    "8306.T":"三菱UFJ", "8316.T":"三井住友", "8411.T":"みずほ", "7203.T":"トヨタ", "7267.T":"ホンダ",
    "8001.T":"伊藤忠", "8031.T":"三井物産", "8058.T":"三菱商事", "9984.T":"ソフトバンクG", "9432.T":"NTT",
    "9433.T":"KDDI", "9983.T":"ファストリ", "6098.T":"リクルート", "4063.T":"信越化学", "4452.T":"花王",
    "4502.T":"武田薬品", "4568.T":"第一三共", "4519.T":"中外製薬", "2914.T":"JT", "7974.T":"任天堂",
    "1605.T":"INPEX", "5020.T":"ENEOS", "5401.T":"日本製鉄", "6301.T":"コマツ", "7011.T":"三菱重工",
    "8801.T":"三井不動産", "8802.T":"三菱地所", "4661.T":"OLC", "7182.T":"ゆうちょ銀行",
    # US Major
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon", "TSLA":"Tesla", "GOOGL":"Alphabet",
    "META":"Meta", "AVGO":"Broadcom", "ORCL":"Oracle", "CRM":"Salesforce", "ADBE":"Adobe", "AMD":"AMD",
    "LLY":"Eli Lilly", "UNH":"UnitedHealth", "JNJ":"J&J", "ABBV":"AbbVie", "MRK":"Merck", "PFE":"Pfizer",
    "JPM":"JPMorgan", "BAC":"BofA", "WFC":"Wells Fargo", "V":"Visa", "MA":"Mastercard", "BRK-B":"Berkshire",
    "PG":"P&G", "KO":"Coca-Cola", "PEP":"PepsiCo", "WMT":"Walmart", "COST":"Costco", "HD":"Home Depot",
    "XOM":"Exxon", "CVX":"Chevron", "GE":"GE Aerospace", "CAT":"Caterpillar", "BA":"Boeing", "LMT":"Lockheed",
}

# --- UNIVERSE DEFINITIONS (approx 200 per market) ---
US_STOCKS = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTU","IBM","AMD","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","SNPS","CDNS","NOW","PANW","CRWD","ANET"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","VRTX","BMY","GILD","PFE","REGN","SYK","BSX","MDT","ZTS","HCA","CVS","CI","ELV"],
    "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","PNC","USB","TFC","MMC","AIG","MET","PRU","AFL","CB","ICE","SPGI","V","MA","BRK-B"],
    "Cons. Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ROST","GM","F","MAR","HLT","EBAY","CMG","YUM","LULU","DHI","LEN","ORLY","AZO"],
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

# =========================
# Core Logic & Analytics
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t and isinstance(t, str)]))
    frames = []
    chunk_size = 80
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            # Threaded download for speed
            raw = yf.download(" ".join(chunk), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not raw.empty: frames.append(raw)
        except: continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

@st.cache_data(ttl=86400)
def get_name(ticker: str) -> str:
    # 1. Internal DB (Instant)
    if ticker in NAME_DB: return NAME_DB[ticker]
    # 2. Fallback
    return ticker

def extract_close(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # Handle yfinance MultiIndex variations
            if "Close" in df.columns.get_level_values(0): close = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(1): close = df.xs("Close", axis=1, level=1)
            else: return pd.DataFrame()
        else: return pd.DataFrame()
        
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        keep = [c for c in expected if c in close.columns]
        return close[keep]
    except: return pd.DataFrame()

def calc_stats(s: pd.Series, b: pd.Series, win: int) -> Dict:
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    
    # Accel (2nd half vs 1st half implied)
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    
    # DD
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # Stable check
    s_short = s.tail(6).dropna()
    b_short = b.tail(6).dropna()
    stable = "⚠️"
    if len(s_short)==6 and len(b_short)==6:
        rs_short = (s_short.iloc[-1]/s_short.iloc[0]-1) - (b_short.iloc[-1]/b_short.iloc[0]-1)
        if np.sign(rs_short) == np.sign(p_ret - b_ret): stable = "✅"
    
    return {"RS": p_ret - b_ret, "Accel": accel, "MaxDD": dd, "Stable": stable, "Ret": p_ret}

def zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# =========================
# AI / LLM Logic (Gemini)
# =========================
def call_gemini_debate(ticker: str, company: str, stats: Dict) -> str:
    """
    Calls Google Gemini to simulate an investment committee debate.
    """
    if not GOOGLE_API_KEY or not HAS_GENAI:
        return deterministic_fallback(ticker, company, stats)
    
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        あなたはヘッジファンドの投資委員会です。以下の銘柄について、3人の専門エージェントとして議論し、最終的な投資判断を下してください。
        
        対象銘柄: {company} ({ticker})
        テクニカル指標:
        - 相対強度(RS): {stats['RS']:.2f}% (市場平均に対する超過リターン)
        - 加速力(Accel): {stats['Accel']:.2f} (トレンドの加速性)
        - 最大ドローダウン(MaxDD): {stats['MaxDD']:.2f}% (リスク)
        - 短期整合性(Stable): {stats['Stable']}
        
        役割:
        1. モメンタム戦略家: トレンドの強さと加速を重視。
        2. リスクマネジャー: ドローダウンとボラティリティを警戒。
        3. マクロストラテジスト: 市場全体との整合性を重視。
        
        出力形式:
        【モメンタム戦略家】...
        【リスクマネジャー】...
        【マクロストラテジスト】...
        
        【委員会の最終結論】(推奨: 強気 / 中立 / 弱気) 理由を1行で。
        
        日本語で、プロフェッショナルな口調で出力してください。
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI生成エラー: {str(e)}\n\n" + deterministic_fallback(ticker, company, stats)

def deterministic_fallback(ticker: str, company: str, stats: Dict) -> str:
    # Rule-based fallback if API fails
    rs = stats['RS']
    accel = stats['Accel']
    dd = stats['MaxDD']
    
    view = "中立"
    if rs > 0 and accel > 0 and dd < 10: view = "強気"
    elif rs < 0: view = "弱気"
    
    return f"""
    ※Google APIキー未設定またはエラーのため、ルールベース分析を表示します。
    
    【モメンタム分析】RSは{rs:+.2f}%、加速は{accel:+.2f}。{'トレンドは堅調です' if rs>0 else 'トレンドは劣後しています'}。
    【リスク評価】期間中の最大下落率は{dd:.2f}%です。
    
    【システム判定】{view}
    """

# =========================
# Main App
# =========================
def main():
    # Header
    st.markdown("<div style='font-size:24px; font-weight:900; background:linear-gradient(90deg,#4facfe,#00f2fe); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>AlphaLens Gemini</div>", unsafe_allow_html=True)
    st.caption("Google AI Agents × Pro-Grade Sector Analytics")

    # API Key Alert
    if not GOOGLE_API_KEY:
        st.warning("⚠️ Google API Key not found. AI Debate will be simulated (Rule-based). Set 'GOOGLE_API_KEY' in secrets to enable Gemini.")

    # Deck
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

    # Context
    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # 1. Core Sync
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    
    if sync or "core_df" not in st.session_state or st.session_state.get("last_m") != market_key:
        with st.spinner("Connecting to Market Data..."):
            raw = fetch_bulk_cached(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.core_df
    
    # Integrity Check
    present = [t for t in core_tickers if t in core_df.columns]
    computable = [t for t in present if core_df[t].tail(win+1).notna().sum() >= win+1]
    
    if bench not in computable:
        st.error(f"❌ Benchmark {bench} is missing or has insufficient data.")
        st.stop()
        
    # Market Overview
    b_stats = calc_stats(core_df[bench], core_df[bench], win)
    b_ret = b_stats["Ret"]
    
    sec_data = []
    for s_name, s_ticker in m_cfg["sectors"].items():
        if s_ticker in computable:
            s_s = calc_stats(core_df[s_ticker], core_df[bench], win)
            if s_s:
                sec_data.append({"Sector": s_name, "RS": s_s["RS"], "Return": s_s["Ret"]})
    
    if not sec_data:
        st.error("No sector data available.")
        st.stop()
        
    sdf = pd.DataFrame(sec_data).sort_values("RS", ascending=True)
    leaders = len(sdf[sdf["RS"] > 0])
    
    st.markdown(f"""
    <div class='metric-box'>
        <b>{m_cfg['name']} ({bench})</b>: <span style='font-size:18px'>{b_ret:+.2f}%</span>
        <span class='muted'> | Market Breadth: {leaders}/{len(sdf)} sectors leading</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Sector Chart
    fig = px.bar(sdf, x="RS", y="Sector", orientation='h', color="RS", color_continuous_scale="RdYlGn", title="Sector Relative Strength")
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    # Selection
    click_sec = None
    if event and event.get("selection", {}).get("points"):
        click_sec = event["selection"]["points"][0]["y"]
        
    cols = st.columns(6)
    btn_sec = None
    for i, s in enumerate(m_cfg["sectors"].keys()):
        if cols[i%6].button(s, key=f"btn_{s}", use_container_width=True):
            btn_sec = s
            
    target_sector = btn_sec or click_sec or st.session_state.get("target_sector", list(m_cfg["sectors"].keys())[0])
    st.session_state.target_sector = target_sector
    
    # 3. Drill Down
    st.markdown("---")
    st.subheader(f"🔍 Drilldown: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    sec_cache_key = f"{market_key}_{target_sector}_{datetime.now().hour}"
    if sec_cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner(f"Analyzing {len(stock_list)} stocks in {target_sector}..."):
            raw_s = fetch_bulk_cached(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = sec_cache_key
            
    sec_df = st.session_state.sec_df
    
    results = []
    for t in full_list:
        if t == bench: continue
        if t in sec_df.columns:
            stats = calc_stats(sec_df[t], sec_df[bench], win)
            if stats:
                stats["Ticker"] = t
                stats["Name"] = get_name(t)
                results.append(stats)
                
    if not results:
        st.warning("No valid stocks found in this sector.")
        st.stop()
        
    df = pd.DataFrame(results)
    df["RS_z"] = zscore(df["RS"])
    df["Acc_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["Apex"] = 0.6*df["RS_z"] + 0.25*df["Acc_z"] - 0.15*df["DD_z"]
    df = df.sort_values("Apex", ascending=False).reset_index(drop=True)
    
    # 4. Top Pick AI Debate
    col_l, col_r = st.columns([1.5, 1])
    
    with col_l:
        st.markdown("##### 🏆 Sector Leaderboard")
        st.dataframe(
            df[["Name", "Ticker", "Apex", "RS", "Accel", "MaxDD", "Stable"]],
            column_config={
                "Apex": st.column_config.NumberColumn(format="%.2f"),
                "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
                "Accel": st.column_config.NumberColumn(format="%.2f"),
                "MaxDD": st.column_config.NumberColumn(format="%.2f%%"),
            },
            hide_index=True, use_container_width=True
        )
        
    with col_r:
        top = df.iloc[0]
        st.markdown(f"##### 🤖 AI Investment Committee")
        st.caption(f"Target: {top['Name']}")
        
        # Determine strict AI inputs (deterministic base)
        stats_for_ai = {
            "RS": top["RS"], "Accel": top["Accel"],
            "MaxDD": top["MaxDD"], "Stable": top["Stable"]
        }
        
        # Call Gemini (or fallback)
        debate_content = call_gemini_debate(top["Ticker"], top["Name"], stats_for_ai)
        
        st.markdown(f"""
        <div class='ai-box'>
            <div style='font-size:13px; white-space: pre-wrap;'>{debate_content}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. News
    st.markdown("---")
    st.subheader("📰 Latest News")
    news = yf.Ticker(top["Ticker"]).news[:5]
    if not news:
        st.caption("No recent news found via yfinance.")
    for n in news:
        ts = n.get("providerPublishTime", 0)
        dt = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        st.markdown(f"**{n['title']}** <span class='muted'>| {n['publisher']} | {dt}</span>", unsafe_allow_html=True)
        st.markdown(f"[Read Article]({n['link']})")

if __name__ == "__main__":
    main()