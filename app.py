import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from google import genai
import json, threading, time, random
from io import StringIO
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# --- 1. DESIGN: OBSIDIAN CONTROL CENTER ---
st.set_page_config(page_title="AlphaLens v24.0", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&family=Orbitron:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #000000; color: #e6edf3; }
.header-kpi { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
.kpi-box { flex: 1; min-width: 140px; background: #0a0a0a; padding: 12px; border-radius: 8px; border: 1px solid #bc13fe; text-align: center; }
.kpi-label { font-size: 0.6em; color: #bc13fe; text-transform: uppercase; letter-spacing: 1px; }
.kpi-val { font-size: 1.1em; font-weight: bold; color: #00f2ff; font-family: 'Orbitron'; }
.stButton>button { width: 100%; border-radius: 6px; height: 3.2em; background: #0a0a0a; color: #00f2ff; border: 1px solid #00f2ff; font-weight: bold; transition: 0.25s; }
.stButton>button:hover { background: #00f2ff; color: #000; box-shadow: 0 0 20px #00f2ff; }
.sync-btn>div>button { background: #bc13fe !important; color: white !important; border: none !important; }
.glass-card { background: rgba(10, 10, 10, 0.98); padding: 22px; border-radius: 12px; border: 1px solid #222; border-left: 6px solid #00f2ff; margin-bottom: 20px; line-height: 1.8; }
.news-tag { display: block; padding: 10px; background: #0a0a0a; border-radius: 6px; font-size: 0.85em; margin: 6px 0; border: 1px solid #333; text-decoration: none !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. CORE UTILITY ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.error("GEMINI_API_KEY MISSING"); st.stop()
client = genai.Client(api_key=api_key)

def calc_ret(series: pd.Series, days: int):
    """厳密な期間計算エンジン"""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) <= days: return None
        prev = float(s.iloc[-(days+1)])
        now = float(s.iloc[-1])
        if prev == 0: return None
        return (now/prev - 1.0) * 100.0
    except: return None

def stream_ai_response(prompt, placeholder):
    full = ""
    try:
        stream = client.models.generate_content(model="gemini-flash-latest", contents=prompt, stream=True)
        for chunk in stream:
            part = getattr(chunk, "text", "") or ""
            full += part
            if len(full) % 200 < len(part):
                placeholder.markdown(f"<div class='glass-card' style='border-left-color:#bc13fe;'>{full}▌</div>", unsafe_allow_html=True)
        placeholder.markdown(f"<div class='glass-card'>{full}</div>", unsafe_allow_html=True)
        return full
    except:
        res = client.models.generate_content(model="gemini-flash-latest", contents=prompt).text
        placeholder.markdown(f"<div class='glass-card'>{res}</div>", unsafe_allow_html=True)
        return res

@st.cache_data(ttl=1800)
def fetch_bulk_yf(tickers, suffix, chunk_size=80):
    yf_tickers = [f"{t}.T" if suffix == "JP" else t for t in tickers]
    # ベンチマークも追加取得
    bench = "SPY" if suffix == "US" else "1306.T"
    yf_tickers.append(bench)
    
    frames = []
    for i in range(0, len(yf_tickers), chunk_size):
        chunk = yf_tickers[i:i+chunk_size]
        try:
            df = yf.download(" ".join(chunk), period="14mo", interval="1d", group_by='ticker', threads=True, progress=False, auto_adjust=True)
            if not df.empty: frames.append(df)
            time.sleep(0.3)
        except: continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

def get_ticker_close(data: pd.DataFrame, ticker: str):
    """多層Indexを考慮した終値ベクトル抽出"""
    try:
        if ticker in data.columns.get_level_values(0):
            return data[ticker]['Close'].dropna()
        if ("Close", ticker) in data.columns:
            return data[("Close", ticker)].dropna()
        return None
    except: return None

# --- 3. MASTER CATALOG (418 Assets) ---
SECTOR_CATALOG = {
    "US MARKET": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Alphabet","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","VRT":"Vertiv","SMCI":"Supermicro"},
        "Software / SaaS / Cyber": {"SNOW":"Snowflake","PLTR":"Palantir","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","FTNT":"Fortinet","ZS":"Zscaler","OKTA":"Okta","TEAM":"Atlassian","ADSK":"Autodesk","SHOP":"Shopify","NET":"Cloudflare"},
        "Financials": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","AXP":"Amex","BLK":"BlackRock","C":"Citigroup"},
        "Healthcare / Bio": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo","BMY":"Bristol","AMGN":"Amgen","ISRG":"Intuitive"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar","DE":"Deere","MMM":"3M"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","KMI":"KinderMorgan","MPC":"Marathon","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo"},
        "Consumer / Media": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","MCD":"McDonald's","TGT":"Target","DIS":"Disney","PARA":"Paramount"},
        "Materials / REITs": {"LIN":"Linde","APD":"AirProducts","SHW":"Sherwin","ECL":"Ecolab","FCX":"Freeport","PLD":"Prologis","AMT":"AmericanTower","PSA":"PublicStorage","O":"RealtyIncome"}
    },
    "JP MARKET": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","7735":"スクリン","6526":"ソシオネ","6963":"ローム","7751":"キヤノン","6981":"村田製","6762":"TDK","6861":"キーエンス"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ","6098":"リクルート","4385":"メルカリ","3923":"ラクス","9613":"NTTデータ"},
        "重工業/防衛/建機": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC","6305":"日立建機","6113":"アマダ","6473":"ジェイテクト","6326":"クボタ"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","7205":"日野自"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO"},
        "総合商社/エネルギー": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX","5020":"ENEOS","1518":"三井松島"},
        "必需品/医薬/化学": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","4911":"資生堂","4901":"富士フイルム","4502":"武田薬品","4568":"第一三共"},
        "不動産/鉄道/電力": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組","1928":"積水ハウス","1925":"大和ハウス","9022":"JR東海","9020":"JR東日本","9101":"日本郵船","9104":"商船三井","9501":"東京電力","9503":"関西電力"}
    }
}

# --- 4. CONTROL CENTER FLOW ---

if 'bulk_data' not in st.session_state: st.session_state.bulk_data = None
if 'market' not in st.session_state: st.session_state.market = "US MARKET"
if 'sector' not in st.session_state: st.session_state.sector = None

st.title("ALPHALENS v24.0 // OMNI-CONTROL")

# Header KPI (Transparency)
market = st.session_state.market
suffix = "US" if "US" in market else "JP"
all_tickers = []
for gs in SECTOR_CATALOG[market].values(): all_tickers.extend(list(gs.keys()))

st.markdown(f"""
<div class="header-kpi">
    <div class="kpi-box"><div class="kpi-label">Market Cluster</div><div class="kpi-val">{market.split()[0]}</div></div>
    <div class="kpi-box"><div class="kpi-label">Control Target</div><div class="kpi-val">{len(all_tickers)} Assets</div></div>
    <div class="kpi-box"><div class="kpi-label">Bench</div><div class="kpi-val">{'SPY' if suffix=='US' else 'TOPIX'}</div></div>
</div>
""", unsafe_allow_html=True)

# 5. COMMAND BAR: SYNC & WINDOW
c_cmd1, c_cmd2, c_cmd3 = st.columns([1,1,2])
with c_cmd1:
    if st.button("🇺🇸 SET US", use_container_width=True): 
        st.session_state.market = "US MARKET"; st.session_state.sector = None; st.rerun()
with c_cmd2:
    if st.button("🇯🇵 SET JP", use_container_width=True): 
        st.session_state.market = "JP MARKET"; st.session_state.sector = None; st.rerun()
with c_cmd3:
    window_ui = st.radio("ANALYSIS WINDOW", ["1W", "1M", "3M"], index=1, horizontal=True, label_visibility="collapsed")
    win_days = {"1W": 5, "1M": 21, "3M": 63}[window_ui]

st.markdown("<div class='sync-btn'>", unsafe_allow_html=True)
if st.button("🔄 SYNC QUANTUM DATA (ALL CONSTITUENTS)"):
    with st.status("Establishing Multi-Packet Connection...", expanded=True) as status:
        st.session_state.bulk_data = fetch_bulk_yf(tuple(sorted(set(all_tickers))), suffix)
        status.update(label="Sync Successful", state="complete", expanded=False)
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.bulk_data is None:
    st.info("Please press [SYNC DATA] to initiate the管制塔.")
    st.stop()

# 6. PULSE ENGINE (TRUE VECTOR CALC)
data = st.session_state.bulk_data
# Extract Close Matrix (Vectorization)
if "Close" in data.columns.get_level_values(0):
    close_matrix = data["Close"].copy()
else:
    # 銘柄数が少ない場合のフォールバック
    close_matrix = pd.DataFrame({t: get_ticker_close(data, t) for t in data.columns.get_level_values(0).unique()})

# Calculate Benchmark Return
bench_t = "SPY" if suffix == "US" else "1306.T"
bench_px = get_ticker_close(data, bench_t)
bench_ret = calc_ret(bench_px, win_days) if bench_px is not None else 0.0

pulse_rows = []
missing = set()
for g_name, tickers in SECTOR_CATALOG[market].items():
    rets = []
    for t in tickers.keys():
        t_key = f"{t}.T" if suffix == "JP" else t
        if t_key in close_matrix.columns:
            r = calc_ret(close_matrix[t_key], win_days)
            if r is not None: rets.append(r)
            else: missing.add(t)
        else: missing.add(t)
    if rets:
        avg = sum(rets)/len(rets)
        pulse_rows.append({"Sector": g_name, "Return": avg, "Excess": avg - bench_ret, "N": len(rets)})

pulse_df = pd.DataFrame(pulse_rows).sort_values("Excess", ascending=False)

# 7. SECTOR PULSE (RS FIXED)
st.write(f"### 🔥 SECTOR RELATIVE STRENGTH (vs Bench: {bench_ret:+.2f}%)")
fig = px.bar(pulse_df, x="Excess", y="Sector", orientation='h', color="Excess", color_continuous_scale="RdYlGn", hover_data=["Return", "N"])
st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

if st.button("🧠 GENERATE MACRO INTELLIGENCE (AI Stream)"):
    ai_c = st.empty()
    stream_ai_response(f"{market}市場のセクター状況(Bench対比)から地合いと戦略を日本語1000文字で。Data: {pulse_df.to_json()}", ai_c)

# 8. SECTOR CHIPS
st.write("### 📂 ACTIVATE SECTOR DATA-SET")
chips = pulse_df["Sector"].tolist()[:9]
s_cols = st.columns(3)
for i, s_name in enumerate(chips):
    if s_cols[i%3].button(f"💠 {s_name}"): st.session_state.sector = s_name

st.divider()

# 9. SECTOR DRILL DOWN
if st.session_state.sector:
    sel_sec = st.session_state.sector
    st.subheader(f"📍 {sel_sec} UNIVERSE")
    
    if st.button(f"▶︎ GENERATE {sel_sec} CONTEXT (AI)"):
        ai_sec_c = st.empty()
        stream_ai_response(f"{sel_sec}セクターの最新状況をマクロ、需給、地政学含め日本語500文字で分析。挨拶不要。", ai_sec_c)

    # Matrix Table (Instant)
    target_map = SECTOR_CATALOG[market][sel_sec]
    results = []
    for t, n in target_map.items():
        t_key = f"{t}.T" if suffix == "JP" else t
        if t_key in close_matrix.columns:
            px_series = close_matrix[t_key].dropna()
            ret = calc_ret(px_series, win_days)
            results.append({"Name":n,"Ticker":t,"Price":px_series.iloc[-1],"Return":ret,"df":px_series})
    
    if results:
        df_disp = pd.DataFrame(results).drop(columns=['df']).sort_values("Return", ascending=False)
        df_disp["Return"] = df_disp["Return"].map(lambda x: f"{x:+.1f}%" if x is not None else "N/A")
        st.dataframe(df_disp.set_index("Name"), use_container_width=True)

    # 10. RANKING (Consistent Logic)
    if st.button(f"🔍 EXECUTE {sel_sec} DEEP QUANT RANKING"):
        with st.status("Aligning Evaluation Factors...", expanded=True) as status:
            scored = []
            for r in results:
                c = r["df"]
                # Dynamic Momentum aligned with window
                if window_ui == "1W": mom = (calc_ret(c, 21) or 0) - (r['Return'] or 0)
                elif window_ui == "1M": mom = (calc_ret(c, 63) or 0) - (r['Return'] or 0)
                else: mom = (calc_ret(c, 252) or 0) - (r['Return'] or 0)
                
                score = round(mom*0.5 + (r['Return'] or 0)*0.5, 2)
                scored.append({"Name":r['Name'], "Score":score, "Mom_Factor":mom, f"Ret_{window_ui}":r['Return']})
            
            ranked = sorted(scored, key=lambda x: x['Score'], reverse=True)
            st.write("### 📈 ALPHA RANKING")
            st.dataframe(pd.DataFrame(ranked).set_index("Name"), use_container_width=True)
            
            ai_r_c = st.empty()
            stream_ai_response(f"Quant Analystとして格付け結果から投資推奨理由とリスクを日本語分析。Data: {json.dumps(ranked[:5], ensure_ascii=False)}", ai_r_c)
            status.update(label="Ranking Complete", state="complete")