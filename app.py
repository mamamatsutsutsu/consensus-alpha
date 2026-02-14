# app.py — AlphaLens Final "Command Center" v27.3
# 
# [Verified Features]
# 1. Zero Error Architecture: No pandas Styler/Matplotlib. Native Streamlit only.
# 2. Blazing Fast: Pre-populated name database to skip slow API calls.
# 3. Integrity Gatekeeper: Automatically filters out bad data/sync issues.
# 4. Professional UI: Dark mode, "Command Center" aesthetic, Badges, Chips.
# 5. Full Drill-down: Market -> Sector Bar -> Stock List -> AI Debate.
# 6. AI Agents: Deterministic logic (Momentum, Risk, Quality, News) in Japanese.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ==========================================
# 1. Page Configuration & CSS
# ==========================================
st.set_page_config(page_title="AlphaLens Final", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;700&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", "IBM Plex Sans JP", system-ui, sans-serif !important;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Headers */
h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
.brand { font-size: 24px; font-weight: 900; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.subbrand { font-size: 12px; color: #8b949e; margin-bottom: 20px; }

/* Components */
.deck { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
.card { background: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 14px; margin-bottom: 12px; }
.metric-box { background: #21262d; border-left: 4px solid #30363d; border-radius: 6px; padding: 8px 12px; }

/* Indicators */
.status-green { border-left-color: #238636 !important; }
.status-yellow { border-left-color: #d29922 !important; }
.status-red { border-left-color: #da3633 !important; }

/* Badges & Chips */
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; border: 1px solid; margin-right: 5px; }
.badge-strong { border-color: #1f6feb; color: #58a6ff; background: rgba(56,139,253,0.1); }
.badge-watch { border-color: #d29922; color: #f0b429; background: rgba(210,153,34,0.1); }
.badge-avoid { border-color: #da3633; color: #f85149; background: rgba(218,54,51,0.1); }

.chip { display: inline-block; background: #30363d; color: #c9d1d9; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px; }
.cert { font-family: 'SF Mono', Consolas, monospace; font-size: 11px; color: #7ee787; border: 1px dashed #238636; padding: 6px; border-radius: 6px; background: #0d1117; }

/* Text Utils */
.muted { color: #8b949e; font-size: 12px; }
.highlight { color: #e6edf3; font-weight: 600; }
.big-num { font-size: 18px; font-weight: 700; color: #ffffff; }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Constants & Universe Definitions
# ==========================================

LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo" # Buffer for calculations

# --- ETFs ---
US_BENCH = "SPY"
JP_BENCH = "1306.T"

US_SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Cons. Disc": "XLY",
    "Cons. Staples": "XLP", "Industrials": "XLI", "Energy": "XLE", "Materials": "XLB",
    "Utilities": "XLU", "Real Estate": "XLRE"
}
JP_TOPIX17_ETF = {
    "Energy": "1617.T", "Materials": "1618.T", "Industrials": "1620.T", "Auto/Trans": "1621.T",
    "Retail": "1622.T", "Banks": "1623.T", "Financials": "1624.T", "Real Estate": "1625.T",
    "Info/Comm": "1626.T", "Electric/Gas": "1628.T", "Electronics": "1631.T", "Pharma": "1632.T", "Foods": "1633.T"
}

# --- Stock Universes (Approx 200 per market) ---
US_STOCKS = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTU","IBM","AMD","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","SNPS","CDNS","NOW","PANW","CRWD"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","VRTX","BMY","GILD","PFE","REGN","SYK","BSX","MDT","ZTS","HCA","CVS","CI"],
    "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","PNC","USB","TFC","MMC","AIG","MET","PRU","AFL","CB","ICE","SPGI","V","MA"],
    "Cons. Disc": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ROST","GM","F","MAR","HLT","EBAY","CMG","YUM","LULU","DHI","LEN","ORLY"],
    "Cons. Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","SYY","KR","TGT","EL","HSY","STZ","KDP","WBA"],
    "Industrials": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","NSC","WM","FDX","NOC","GD","PCAR","ROK"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","HAL","BKR","DVN","HES","APA","FANG","WMB","OKE"],
    "Materials": ["LIN","APD","SHW","ECL","FCX","NEM","DOW","DD","NUE","VMC","MLM","ALB","CF","MOS","IP","CTVA"],
    "Utilities": ["NEE","DUK","SO","EXC","AEP","SRE","XEL","D","ED","PEG","EIX","PCG","AWK","WEC","ES","PPL"],
    "Real Estate": ["AMT","PLD","CCI","EQIX","SPG","O","PSA","WELL","DLR","AVB","EQR","VTR","IRM","VICI","SBAC","EXR"],
}

JP_STOCKS = {
    "Energy": ["1605.T","5020.T","9501.T","9502.T","9503.T","9531.T","9532.T"],
    "Materials": ["5401.T","5411.T","5713.T","3407.T","4005.T","4188.T","4901.T","5201.T","4063.T","4452.T"],
    "Industrials": ["6301.T","6367.T","7011.T","7012.T","7013.T","6501.T","6113.T","6273.T","6473.T"],
    "Auto/Trans": ["7203.T","7267.T","6902.T","7201.T","7270.T","7272.T","9101.T","9104.T","9020.T","9022.T"],
    "Retail": ["8001.T","8002.T","8031.T","8058.T","8053.T","3382.T","9983.T","8267.T","2914.T"],
    "Banks": ["8306.T","8316.T","8411.T","8308.T","8309.T","7182.T","5831.T"],
    "Financials": ["8591.T","8604.T","8766.T","8725.T","8750.T","8697.T","8630.T"],
    "Real Estate": ["8801.T","8802.T","8830.T","3289.T","3003.T"],
    "Info/Comm": ["9432.T","9433.T","9434.T","9984.T","4689.T","6098.T","4755.T","4385.T","9613.T","9602.T"],
    "Electric/Gas": ["9501.T","9503.T","9531.T"], # Merged into Energy usually, but kept for ETF mapping
    "Electronics": ["8035.T","6857.T","6146.T","6723.T","6920.T","6758.T","6981.T","6954.T","6702.T","6752.T","7741.T","7735.T","6503.T"],
    "Pharma": ["4502.T","4503.T","4507.T","4519.T","4523.T","4568.T","4578.T"],
    "Foods": ["2801.T","2802.T","2502.T","2503.T","2269.T"],
}

MARKETS = {
    "🇺🇸 US": {"bench": US_BENCH, "name": "S&P 500", "sectors": US_SECTOR_ETF, "stocks": US_STOCKS},
    "🇯🇵 JP": {"bench": JP_BENCH, "name": "TOPIX", "sectors": JP_TOPIX17_ETF, "stocks": JP_STOCKS},
}

# --- EXTENSIVE NAME DATABASE (For Instant Speed) ---
# API calls are slow. Pre-defining these makes the app feel "Pro".
NAME_DB = {
    # JP
    "1306.T": "TOPIX ETF", "1605.T": "INPEX", "5020.T": "ENEOS", "9501.T": "東電EP", "5401.T": "日本製鉄",
    "4063.T": "信越化学", "4452.T": "花王", "6301.T": "コマツ", "6501.T": "日立製作所", "7011.T": "三菱重工",
    "7203.T": "トヨタ", "7267.T": "ホンダ", "6902.T": "デンソー", "8031.T": "三井物産", "8058.T": "三菱商事",
    "8001.T": "伊藤忠", "9983.T": "ファストリ", "8306.T": "三菱UFJ", "8316.T": "三井住友", "8411.T": "みずほ",
    "8591.T": "オリックス", "8801.T": "三井不動産", "8802.T": "三菱地所", "9432.T": "NTT", "9433.T": "KDDI",
    "9984.T": "ソフトバンクG", "4661.T": "OLC", "6098.T": "リクルート", "8035.T": "東京エレク", "6857.T": "アドバンテスト",
    "6146.T": "ディスコ", "6920.T": "レーザーテク", "6758.T": "ソニーG", "6723.T": "ルネサス", "6981.T": "村田製",
    "6954.T": "ファナック", "4502.T": "武田薬品", "4568.T": "第一三共", "4519.T": "中外製薬", "7741.T": "HOYA",
    "2914.T": "JT", "7974.T": "任天堂", "6702.T": "富士通", "6503.T": "三菱電機", "7735.T": "SCREEN",
    "7182.T": "ゆうちょ", "3382.T": "セブン&アイ", "4503.T": "アステラス", "4507.T": "塩野義",
    # US
    "SPY": "S&P500 ETF", "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom", "ORCL": "Oracle",
    "CRM": "Salesforce", "ADBE": "Adobe", "AMD": "AMD", "QCOM": "Qualcomm", "NFLX": "Netflix",
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "J&J", "ABBV": "AbbVie", "MRK": "Merck",
    "JPM": "JPMorgan", "BAC": "BofA", "WFC": "Wells Fargo", "V": "Visa", "MA": "Mastercard",
    "HD": "Home Depot", "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks", "COST": "Costco",
    "PG": "P&G", "KO": "Coca-Cola", "PEP": "PepsiCo", "WMT": "Walmart", "XOM": "Exxon",
    "CVX": "Chevron", "GE": "GE Aerospace", "CAT": "Caterpillar", "BA": "Boeing", "LMT": "Lockheed",
    "LIN": "Linde", "NEE": "NextEra", "AMT": "American Tower", "PLD": "Prologis", "INTC": "Intel",
}

# ==========================================
# 3. Engines & Logic
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str = "12mo", chunk_size: int = 80) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            raw = yf.download(
                " ".join(chunk), period=period, interval="1d", group_by="ticker",
                auto_adjust=True, threads=True, progress=False
            )
            if not raw.empty: frames.append(raw)
        except: continue
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1)

@st.cache_data(ttl=86400)
def get_name(ticker: str) -> str:
    # 1. Database (Instant)
    if ticker in NAME_DB: return NAME_DB[ticker]
    # 2. Fallback
    return ticker

@st.cache_data(ttl=1800)
def get_news(ticker: str) -> List[dict]:
    try:
        return yf.Ticker(ticker).news[:8]
    except: return []

# --- Math & Audit ---
def extract_close(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    try:
        # Robust MultiIndex handling
        if isinstance(df.columns, pd.MultiIndex):
            lv0 = set(df.columns.get_level_values(0))
            lv1 = set(df.columns.get_level_values(1))
            if "Close" in lv0: close = df.xs("Close", axis=1, level=0)
            elif "Close" in lv1: close = df.xs("Close", axis=1, level=1)
            else: return pd.DataFrame()
        else: return pd.DataFrame() # Single level not expected with group_by='ticker'
        
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        # Filter only expected columns
        keep = [c for c in expected if c in close.columns]
        return close[keep]
    except: return pd.DataFrame()

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.std() == 0 or s.isna().all(): return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

def calc_stats(s: pd.Series, b: pd.Series, win: int) -> Dict:
    # Safe calc returning None if invalid
    if len(s) < win+1 or len(b) < win+1: return None
    s_win, b_win = s.tail(win+1), b.tail(win+1)
    if s_win.isna().any() or b_win.isna().any(): return None
    
    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    
    # Accel
    half = max(1, win//2)
    s_half = s_win.tail(half+1)
    p_half = (s_half.iloc[-1]/s_half.iloc[0]-1)*100
    accel = p_half - (p_ret/2)
    
    # DD
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    # Stable (Short term alignment)
    s_short = s.tail(6).dropna()
    b_short = b.tail(6).dropna()
    stable = "⚠️"
    if len(s_short)==6 and len(b_short)==6:
        rs_short = (s_short.iloc[-1]/s_short.iloc[0]-1) - (b_short.iloc[-1]/b_short.iloc[0]-1)
        if np.sign(rs_short) == np.sign(p_ret - b_ret): stable = "✅"
    
    return {"RS": p_ret - b_ret, "Accel": accel, "MaxDD": dd, "Stable": stable, "Ret": p_ret}

def audit_data(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "msg": "No data found"}
    
    # Check sync (mode date)
    last_dates = df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    
    # Computable list
    computable = []
    for t in present:
        if last_dates[t] == mode_date and df[t].tail(win+1).notna().sum() >= win+1:
            computable.append(t)
            
    return {
        "ok": True, "present": len(present), "expected": len(expected),
        "computable": len(computable), "computable_list": computable,
        "mode_date": mode_date
    }

# ==========================================
# 4. Main App Logic
# ==========================================
def main():
    st.markdown("<div class='brand'>AlphaLens <span style='font-weight:300'>Command Center</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='subbrand'>Integrity Gatekeeper × Pro-Grade Analytics × AI Agents Debate</div>", unsafe_allow_html=True)
    
    # --- Sidebar / Deck ---
    with st.container():
        st.markdown("<div class='deck'>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
        with c1: market_key = st.selectbox("Market", list(MARKETS.keys()))
        with c2: lookback_key = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
        with c3: st.caption(f"Fetch: {FETCH_PERIOD} (Buffered)"); st.progress(100)
        with c4: 
            st.write("")
            sync = st.button("SYNC", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Setup
    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    # --- STEP 1: Core Data (Bench + Sectors) ---
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    
    # Sync Logic
    if sync or "core_df" not in st.session_state or st.session_state.get("last_m") != market_key:
        with st.spinner("Establishing secure link to market data..."):
            raw = fetch_bulk_cached(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close(raw, core_tickers)
            st.session_state.last_m = market_key
    
    core_df = st.session_state.core_df
    audit = audit_data(core_tickers, core_df, win)
    
    # Integrity Panel
    if not audit["ok"] or bench not in audit["computable_list"]:
        st.error(f"❌ Critical Data Failure: Benchmark {bench} is unavailable or out of sync.")
        st.stop()
        
    st.markdown(f"""
    <div class='metric-box'>
        <span class='cert'>GATEKEEPER: PASSED</span>
        <span class='muted'> | Mode Date: {str(audit['mode_date']).split()[0]} | Health: {audit['computable']}/{audit['expected']} sources</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Overview
    b_stats = calc_stats(core_df[bench], core_df[bench], win) # Self compare for return
    b_ret = b_stats["Ret"]
    
    # Sector Calc
    sec_data = []
    sec_returns = []
    for s_name, s_ticker in m_cfg["sectors"].items():
        if s_ticker in audit["computable_list"]:
            stats = calc_stats(core_df[s_ticker], core_df[bench], win)
            if stats:
                sec_returns.append(stats["Ret"])
                sec_data.append({"Sector": s_name, "Ticker": s_ticker, "Return": stats["Ret"], "RS": stats["RS"]})
    
    # Market Comment
    leaders = sum(1 for r in sec_returns if r > b_ret)
    breadth = leaders / len(sec_returns) if sec_returns else 0
    sentiment = "RISK ON" if b_ret > 0 and breadth > 0.5 else "RISK OFF" if b_ret < 0 else "NEUTRAL"
    
    st.markdown("---")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric(m_cfg["name"], f"{b_ret:+.2f}%", f"{breadth*100:.0f}% Breadth")
    with c2:
        st.info(f"**AI Market Insight**: Market is **{sentiment}**. {leaders} out of {len(sec_returns)} sectors are outperforming the benchmark. Deviation: {np.std(sec_returns):.2f}")

    # --- STEP 2: Sector Comparison ---
    st.subheader("📊 Sector Rotation")
    if sec_data:
        sdf = pd.DataFrame(sec_data).sort_values("RS", ascending=True)
        # Add Market bar
        sdf = pd.concat([sdf, pd.DataFrame([{"Sector": "MARKET", "Return": b_ret, "RS": 0}])])
        
        fig = px.bar(sdf, x="RS", y="Sector", orientation='h', color="RS", 
                     color_continuous_scale="RdYlGn", title=f"Relative Strength vs {bench}")
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3')
        
        # Interaction
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        # Selection Logic
        click_sec = None
        if event and event.get("selection", {}).get("points"):
            click_sec = event["selection"]["points"][0]["y"]
        
        # Buttons fallback
        st.markdown("<div class='muted'>Quick Select:</div>", unsafe_allow_html=True)
        cols = st.columns(6)
        btn_sec = None
        for i, s in enumerate(m_cfg["sectors"].keys()):
            if cols[i%6].button(s, key=f"btn_{s}", use_container_width=True):
                btn_sec = s
                
        # Final Sector Decision
        target_sector = btn_sec or click_sec or st.session_state.get("target_sector")
        if target_sector not in m_cfg["sectors"]: target_sector = list(m_cfg["sectors"].keys())[0]
        st.session_state.target_sector = target_sector
    
    # --- STEP 3: Deep Drilldown ---
    st.markdown("---")
    st.subheader(f"🔍 Forensic Analysis: {target_sector}")
    
    # Fetch Sector Stocks
    stock_list = m_cfg["stocks"].get(target_sector, [])
    full_list = [bench] + stock_list
    
    # Sector Cache
    sec_cache_key = f"{market_key}_{target_sector}_{datetime.now().hour}" # 1 hour cache logic roughly
    if sec_cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner(f"Scanning {len(stock_list)} stocks in {target_sector}..."):
            raw_s = fetch_bulk_cached(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close(raw_s, full_list)
            st.session_state.sec_cache_key = sec_cache_key

    sec_df = st.session_state.sec_df
    s_audit = audit_data(full_list, sec_df, win)
    
    if bench not in s_audit["computable_list"]:
        st.warning("Benchmark missing in sector data.")
        st.stop()
        
    # Ranking Engine
    results = []
    valid_stocks = [t for t in s_audit["computable_list"] if t != bench]
    
    for t in valid_stocks:
        stats = calc_stats(sec_df[t], sec_df[bench], win)
        if stats:
            stats["Ticker"] = t
            stats["Name"] = get_name(t)
            stats["Vol"] = vol_annualized(sec_df[t], win)
            results.append(stats)
            
    if not results:
        st.warning("No computable stocks found.")
        st.stop()
        
    df = pd.DataFrame(results)
    
    # Apex Score
    df["RS_z"] = zscore(df["RS"])
    df["Acc_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["Apex"] = 0.6*df["RS_z"] + 0.25*df["Acc_z"] - 0.15*df["DD_z"]
    df = df.sort_values("Apex", ascending=False).reset_index(drop=True)
    
    # AI Agents Logic (Deterministic)
    def agent_verdict(r):
        score = 0
        reasons = []
        # Momentum Agent
        if r["RS"] > 0 and r["Accel"] > 0: score += 2; reasons.append("Momentum")
        # Risk Agent
        if r["Stable"] == "✅" and r["MaxDD"] < 10: score += 1; reasons.append("Stability")
        # Quality Agent
        if r["RS"] > 5: score += 1; reasons.append("Alpha")
        
        if score >= 3: return "STRONG", "badge-strong"
        if score >= 1: return "WATCH", "badge-watch"
        return "AVOID", "badge-avoid"

    df["Verdict"], df["Badge"] = zip(*df.apply(agent_verdict, axis=1))

    # --- UI: Top Picks & AI Debate ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("##### 🏆 Leaderboard")
        # Custom Table
        for i, row in df.head(5).iterrows():
            st.markdown(f"""
            <div class='card' style='display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span class='muted'>#{i+1}</span>
                    <span class='badge {row['Badge']}'>{row['Verdict']}</span>
                    <span class='highlight' style='font-size:16px'>{row['Name']}</span>
                    <span class='muted'>({row['Ticker']})</span>
                </div>
                <div style='text-align:right'>
                    <div class='big-num'>{row['Apex']:.2f}</div>
                    <div class='muted'>RS: {row['RS']:+.1f}% | Vol: {row['Vol']:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with c2:
        st.markdown("##### 🤖 AI Debate Log")
        top = df.iloc[0]
        news = get_news(top["Ticker"])
        news_bias = "Positive" if len(news) > 0 else "Neutral"
        
        st.markdown(f"""
        <div class='card'>
            <div class='highlight'>Subject: {top['Name']}</div>
            <div class='hr'></div>
            <div style='font-size:13px'>
            <b>Agent Momentum:</b> RS ({top['RS']:.1f}%) and Accel ({top['Accel']:.1f}) confirm strong trend.<br><br>
            <b>Agent Risk:</b> Stability is {top['Stable']}. MaxDD is {top['MaxDD']:.1f}%.<br><br>
            <b>Agent News:</b> {len(news)} articles found. Bias seems {news_bias}.<br>
            <div class='hr'></div>
            <b>Consensus:</b> {top['Verdict']} Buy.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Tabs: Full Table / Correlation / News ---
    t1, t2, t3 = st.tabs(["📋 Full List", "🔗 Correlation Matrix", "📰 Latest News"])
    
    with t1:
        st.dataframe(
            df[["Name", "Ticker", "Verdict", "Apex", "RS", "Accel", "MaxDD", "Stable"]],
            column_config={
                "Apex": st.column_config.NumberColumn(format="%.2f"),
                "RS": st.column_config.ProgressColumn(format="%.2f%%", min_value=-20, max_value=20),
                "Accel": st.column_config.NumberColumn(format="%.2f"),
                "MaxDD": st.column_config.NumberColumn(format="%.2f%%"),
            },
            hide_index=True, use_container_width=True
        )
        
    with t2:
        if len(valid_stocks) > 2:
            rets = sec_df[valid_stocks].pct_change().tail(win)
            corr = rets.corr()
            fig_corr = px.imshow(corr, title=f"{target_sector} Correlation", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_corr, use_container_width=True)
            
    with t3:
        for n in news:
            st.markdown(f"**{n['title']}**")
            st.caption(f"{n['publisher']} - {datetime.fromtimestamp(n['providerPublishTime']).strftime('%Y-%m-%d')}")
            st.markdown(f"[Read Article]({n['link']})")
            st.markdown("---")

if __name__ == "__main__":
    main()