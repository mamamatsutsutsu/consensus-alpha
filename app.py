import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from google import genai
import json
from io import StringIO
from urllib.parse import quote_plus
import time
import requests
import xml.etree.ElementTree as ET

# --- 1. PRO-MINIMALIST DESIGN (CSS) ---
st.set_page_config(page_title="AlphaLens v6.2", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3.5em; background-color: #238636; color: white; border: none; font-weight: bold; letter-spacing: 2px; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 15px; }
    .brief-content { line-height: 1.6; color: #8b949e; font-size: 0.9em; }
    .news-tag { display: inline-block; padding: 2px 8px; background-color: #21262d; border-radius: 4px; font-size: 0.75em; margin: 2px; border: 1px solid #30363d; }
    a { color: #58a6ff !important; text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("ALPHALENS / v6.2")

# --- 2. UTILITY & SECURITY ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("SYSTEM ERROR: MISSING_API_KEY")
    st.stop()
client = genai.Client(api_key=api_key)

def to_f(val):
    """徹底した数値正規化 (NaN/Numpy型をPython float/Noneへ変換)"""
    try:
        if val is None: return None
        f_val = float(val)
        return f_val if f_val == f_val else None # NaN check
    except: return None

# --- 3. SECTOR CATALOG (200 Assets scale) ---
SECTOR_CATALOG = {
    "US UNIVERSE": {
        "BIG TECH": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla"},
        "SEMI / AI": {"AVGO": "Broadcom", "AMD": "AMD", "ASML": "ASML", "TSM": "TSMC", "MU": "Micron", "LRCX": "Lam Research", "AMAT": "Applied Materials", "ARM": "Arm"},
        "FIN / PAY": {"JPM": "JP Morgan", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "AXP": "Amex"}
    },
    "JP UNIVERSE": {
        "TECH / SEMI": {"8035": "Tokyo Electron", "6857": "Advantest", "6758": "Sony", "6723": "Renesas", "6146": "DISCO", "6920": "Lasertec", "6501": "Hitachi", "6702": "Fujitsu"},
        "FINANCE": {"8306": "MUFG", "8316": "SMFG", "8411": "Mizuho", "8766": "Tokio Marine", "8591": "ORIX", "8604": "Nomura", "8725": "MS&AD"},
        "AUTO / HEAVY": {"7203": "Toyota", "7267": "Honda", "7011": "MHI", "7012": "KHI", "7013": "IHI", "6902": "Denso", "7201": "Nissan", "7270": "Subaru"}
    }
}

# --- 4. ROBUST ENGINES ---
@st.cache_data(ttl=3600)
def fetch_price_raw(ticker, suffix):
    candidates = [suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]
    for s in candidates:
        try:
            r = requests.get(f"https://stooq.com/q/d/l/?s={ticker.lower()}.{s}&i=d", timeout=10)
            df = pd.read_csv(StringIO(r.content.decode("utf-8")))
            if "Close" in df.columns and not df.empty:
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date").sort_index()
        except: continue
    return None

@st.cache_data(ttl=86400)
def fetch_fundamentals(ticker, suffix):
    yf_t = f"{ticker}.T" if suffix == "JP" else ticker
    try:
        info = yf.Ticker(yf_t).info
        return {"per": to_f(info.get("trailingPE")), "pbr": to_f(info.get("priceToBook"))}
    except: return {"per": None, "pbr": None}

@st.cache_data(ttl=900)
def fetch_news(ticker, name, suffix):
    lang, gl, ceid = ("en-US", "US", "US:en") if suffix == "US" else ("ja-JP", "JP", "JP:ja")
    q = quote_plus(f"{name} {ticker}")
    try:
        r = requests.get(f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={gl}&ceid={ceid}", timeout=5)
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in ET.fromstring(r.text).findall('.//item')[:3]]
    except: return []

# --- 5. CORE LOGIC ---
def get_alpha_score(m):
    score = 0
    if m.get("mom_12_1") is not None: score += m["mom_12_1"] * 0.4
    if m.get("vol") is not None: score += (30 - m["vol"]) * 0.2
    if m.get("per") is not None and m["per"] > 0: score += min((15 / m["per"]) * 10, 15)
    score += abs(m["dd"]) * -0.3
    return round(score, 2)

# --- 6. CORE INTERFACE ---
st.sidebar.markdown("### CORE_CONFIG")
universe_choice = st.sidebar.selectbox("UNIVERSE", list(SECTOR_CATALOG.keys()))
group_choice = st.sidebar.selectbox("GROUP", list(SECTOR_CATALOG[universe_choice].keys()))
span_choice = st.sidebar.selectbox("SPAN (Chart/DD)", ["3M", "1Y", "3Y"], index=1)

suffix = "US" if "US" in universe_choice else "JP"

if st.sidebar.button("EXECUTE CORE SCAN"):
    results = []
    days = {"3M": 63, "1Y": 252, "3Y": 756}[span_choice]
    
    with st.spinner("SCANNING ASSETS..."):
        for t, name in SECTOR_CATALOG[universe_choice][group_choice].items():
            df = fetch_price_raw(t, suffix)
            if df is not None:
                c = df["Close"].astype(float)
                ref = min(len(c), days + 1)
                
                # Metrics Calculation
                mom_12_1 = None
                if len(c) >= 252:
                    mom_12_1 = to_f(((c.iloc[-1]/c.iloc[-252]-1)*100) - ((c.iloc[-1]/c.iloc[-21]-1)*100))
                
                vol = to_f(c.pct_change().rolling(60).std().iloc[-1] * (252**0.5) * 100) if len(c)>60 else None
                dd = to_f(((c.iloc[-ref:] / c.iloc[-ref:].cummax() - 1) * 100).min())
                
                f = fetch_fundamentals(t, suffix)
                m = {
                    "ticker": t, "name": name, "price": to_f(c.iloc[-1]), 
                    "mom_12_1": mom_12_1, "vol": vol, "dd": dd, 
                    **f, "hist": c.iloc[-ref:]
                }
                m["score"] = get_alpha_score(m)
                m["news"] = fetch_news(t, name, suffix)
                results.append(m)
            time.sleep(0.05)

    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # NORMALIZED CHART
        st.markdown(f"### RELATIVE PERFORMANCE / {span_choice}")
        
        fig = go.Figure()
        for r in results[:10]:
            norm = (r["hist"] / r["hist"].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm, name=r["ticker"], line=dict(width=1.5)))
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown("### STRATEGIC BRIEF")
            # JSON Payload cleaning (Strict conversion for Gemini)
            payload = [{k: v for k, v in r.items() if k not in ['hist', 'news']} for r in results[:10]]
            prompt = f"""Role: Hedge fund quant analyst. Task: Identify alpha using the provided data.
            Rules: Use MOM_12_1 for momentum and PER for valuation. Highlight risks via DD.
            Format: Ranked list with 1-line rationales. No introduction. Data: {json.dumps(payload, ensure_ascii=False)}"""
            try:
                res = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
                st.markdown(f"<div class='metric-card brief-content'>{res.text}</div>", unsafe_allow_html=True)
            except: st.error("BRIEF GENERATION FAILED")

        with col2:
            st.markdown("### METRICS & INTEL")
            for r in results:
                with st.expander(f"{r['ticker']} / SCORE: {r['score']}", expanded=(r == results[0])):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("PRICE", f"{r['price']:.1f}")
                    c2.metric("MOM(12-1)", f"{r['mom_12_1']:.1f}%" if r['mom_12_1'] is not None else "N/A")
                    c3.metric(f"DD({span_choice})", f"{r['dd']:.1f}%", delta_color="inverse")
                    
                    st.write(f"**PER**: {r['per'] if r['per'] else 'N/A'} | **PBR**: {r['pbr'] if r['pbr'] else 'N/A'}")
                    for n in r["news"]:
                        st.markdown(f"<div class='news-tag'><a href='{n['link']}' target='_blank'>{n['title']}</a></div>", unsafe_allow_html=True)