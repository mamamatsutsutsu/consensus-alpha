import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from google import genai
import json, re, threading, time, random
from io import StringIO
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# --- 1. DESIGN: OBSIDIAN STABLE SYSTEM ---
st.set_page_config(page_title="AlphaLens v22.1", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&family=Inter:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #010101; color: #e6edf3; }
    
    /* Stable Radio Toggles (No :has) */
    .stRadio>div { display: flex; gap: 8px; flex-wrap: wrap; }
    .stRadio label { background: #0a0a0a; padding: 10px 18px; border-radius: 6px; border: 1px solid #30363d; cursor: pointer; color: #8b949e; transition: 0.3s; }

    /* Obsidian Glass Cards */
    .glass-card { background: rgba(10, 10, 10, 0.95); padding: 22px; border-radius: 12px; border: 1px solid #1f1f1f; margin-bottom: 20px; border-left: 6px solid #00f2ff; line-height: 1.8; color: #e6edf3; }
    .ai-streaming { border-left-color: #bc13fe; }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; background: #0a0a0a; 
        color: #00f2ff; border: 1px solid #00f2ff; font-weight: bold; font-size: 0.9em; 
        transition: 0.25s; letter-spacing: 1px;
    }
    .stButton>button:hover { background: #00f2ff; color: #010101; box-shadow: 0 0 20px rgba(0, 242, 255, 0.4); }
    
    .action-btn>div>button { 
        background: linear-gradient(135deg, #bc13fe, #7000ff) !important; 
        color: white !important; height: 4.2em !important; border: none !important;
    }
    
    .news-tag { display: block; padding: 10px; background: #0a0a0a; border-radius: 6px; font-size: 0.85em; margin: 6px 0; border: 1px solid #30363d; text-decoration: none !important; }
    .kpi-box { flex: 1; min-width: 130px; background: #0a0a0a; padding: 12px; border-radius: 8px; border: 1px solid #bc13fe; text-align: center; }
    .kpi-label { font-size: 0.6em; color: #bc13fe; text-transform: uppercase; }
    .kpi-val { font-size: 1em; font-weight: bold; color: #00f2ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE (FAIL-SAFE STREAMING) ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.error("GEMINI_API_KEY MISSING"); st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()
def _get_sess():
    if getattr(_thread_local, "session", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "AlphaLens/22.1"})
        _thread_local.session = s
    return _thread_local.session

def stream_ai_response(prompt, placeholder, model="gemini-flash-latest"):
    """ストリーミング描画 + 失敗時フォールバック"""
    full = ""
    try:
        # Stream試行
        try:
            stream = client.models.generate_content(model=model, contents=prompt, stream=True)
            for chunk in stream:
                part = getattr(chunk, "text", "") or ""
                if not part: continue
                full += part
                # 更新頻度を抑えて固まりを防止
                if len(full) % 150 < len(part):
                    placeholder.markdown(f"<div class='glass-card ai-streaming'>{full}▌</div>", unsafe_allow_html=True)
            placeholder.markdown(f"<div class='glass-card'>{full}</div>", unsafe_allow_html=True)
            return full
        except Exception:
            # フォールバック
            res = client.models.generate_content(model=model, contents=prompt).text
            placeholder.markdown(f"<div class='glass-card'>{res}</div>", unsafe_allow_html=True)
            return res
    except Exception as e:
        placeholder.error(f"AI Engine Offline: {e}")
        return None

# Precision Analytics
def calc_ret(series: pd.Series, days: int):
    try:
        if series is None or series.empty: return None
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) < days + 1: return None
        prev = float(s.iloc[-(days+1)])
        now = float(s.iloc[-1])
        if prev == 0: return None
        return (now/prev - 1.0) * 100.0
    except: return None

def fetch_px_single(t, suffix):
    sess = _get_sess()
    sym = t.replace(".", "-").lower()
    cands = ["brk-b", "brk.b", sym] if t.upper() == "BRK-B" else [sym]
    exts = [suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]
    for cand in cands:
        for s in exts:
            try:
                url = f"https://stooq.com/q/d/l/?s={cand}.{s}&i=d"
                r = sess.get(url, timeout=(3, 5))
                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    if not df.empty and "Date" in df.columns:
                        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                        return t, df.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")[["Close"]]
            except: continue
    return t, None

@st.cache_data(ttl=1800)
def fetch_px_batch(tickers_tuple, suffix):
    tickers = tuple(sorted(tickers_tuple))
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers}
        for f in as_completed(futs):
            t, df = f.result()
            if df is not None: out[t] = df
    return out

@st.cache_data(ttl=21600)
def fetch_fund_one(ticker, suffix):
    try:
        tk = yf.Ticker(f"{ticker}.T" if suffix == "JP" else ticker)
        fi = tk.fast_info
        mcap = getattr(fi, "market_cap", None) if not isinstance(fi, dict) else fi.get("market_cap")
        # PER/PBRは info が取れたときのみ。fast_infoを優先して落ちにくくする
        per = pbr = None
        try:
            info = tk.info
            per, pbr = info.get("trailingPE"), info.get("priceToBook")
            summary = info.get("longBusinessSummary", "")
        except: summary = ""
        return {"per": per, "pbr": pbr, "mcap": mcap or info.get("marketCap"), "summary": summary}
    except: return {"per": None, "pbr": None, "mcap": None, "summary": ""}

# --- 3. MASTER CATALOG (418 Assets) ---
SECTOR_CATALOG = {
    "US MARKET": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Alphabet","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","VRT":"Vertiv","SMCI":"Supermicro"},
        "Software / SaaS": {"SNOW":"Snowflake","PLTR":"Palantir","NOW":"ServiceNow","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","TEAM":"Atlassian","ADSK":"Autodesk","SHOP":"Shopify","NET":"Cloudflare"},
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
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO","8750":"第一生命"},
        "総合商社/エネルギー": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX","5020":"ENEOS","1518":"三井松島"},
        "必需品/医薬/化学": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","4911":"資生堂","4901":"富士フイルム","4502":"武田薬品","4568":"第一三共"},
        "不動産/鉄道/電力": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","9022":"JR東海","9020":"JR東日本","9501":"東京電力","9503":"関西電力"}
    }
}

# --- 4. UI STATE ---
if 'market' not in st.session_state: st.session_state.market = "US MARKET"
if 'sector' not in st.session_state: st.session_state.sector = None

st.title("ALPHALENS v22.1 // OBSIDIAN SENTINEL")

# --- HEADER: KPI & SELECTION ---
c_top1, c_top2 = st.columns([1.4, 1])
with c_top1:
    market_ui = st.radio("SELECT MARKET", ["US MARKET", "JP MARKET"], horizontal=True, label_visibility="collapsed")
    if market_ui != st.session_state.market:
        st.session_state.market = market_ui; st.session_state.sector = None; st.rerun()

with c_top2:
    window_ui = st.radio("TIMEFRAME", ["1W (5d)", "1M (21d)", "3M (63d)"], index=1, horizontal=True, label_visibility="collapsed")
    win_days = 5 if "1W" in window_ui else (21 if "1M" in window_ui else 63)

market = st.session_state.market
suffix = "US" if "US" in market else "JP"

# --- 5. PULSE: SECTOR STRENGTH ---
st.write(f"### 🔥 MARKET PULSE ({window_ui})")
all_t = []
for gs in SECTOR_CATALOG[market].values(): all_t.extend(list(gs.keys()))
pulse_tickers = tuple(sorted(set(all_t)))

with st.status("Synchronizing Quantum Assets...", expanded=False) as status:
    st.write("Connecting to Stooq protocol...")
    px_map = fetch_px_batch(pulse_tickers, suffix)
    if not px_map:
        status.update(label="Sync Failed", state="error"); st.error("No Data. Refresh."); st.stop()
    
    st.write("Processing sector momentum...")
    pulse_rows = []
    for g_name, tickers in SECTOR_CATALOG[market].items():
        rets = [calc_ret(px_map[t]["Close"], win_days) for t in tickers.keys() if t in px_map]
        rets = [r for r in rets if r is not None]
        if rets: pulse_rows.append({"Sector": g_name, "Return": (sum(rets)/len(rets)), "N": len(rets)})
    
    pulse_df = pd.DataFrame(pulse_rows)
    if pulse_df.empty: status.update(label="No Data Found", state="error"); st.stop()
    status.update(label="Sync Complete", state="complete")

pulse_df = pulse_df.sort_values("Return", ascending=False)
fig = px.bar(pulse_df, x="Return", y="Sector", orientation='h', color="Return", color_continuous_scale="RdYlGn", hover_data=["N"])
st.plotly_chart(fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

# --- 6. MARKET INTELLIGENCE (Optional for Speed) ---
st.write(f"### ✨ {market.split()[0]} INTELLIGENCE STREAM")
if st.button("▶︎ INITIATE MACRO AI ANALYSIS (Streaming)"):
    ai_container = st.empty()
    prompt_market = f"{market}市場全体のセクター状況に基づき、現在の資金循環と戦略を日本語1000文字で論理的に。挨拶不要。Data: {pulse_df.to_json()}"
    stream_ai_response(prompt_market, ai_container)
else:
    st.caption("AI macro report is optional (run to save initial load time)")

# --- 7. SECTOR SELECTOR ---
st.write("### 📂 INITIATE SECTOR FLOW")
cols = st.columns(3)
for i, s_name in enumerate(pulse_df["Sector"].tolist()[:9]):
    if cols[i%3].button(f"💠 {s_name}"): st.session_state.sector = s_name

st.divider()

# --- 8. DRILL DOWN ---
if st.session_state.sector:
    sel_sec = st.session_state.sector
    st.subheader(f"📍 {sel_sec} CORE DATA")
    
    # AI Sector Context (Streaming)
    st.write("Building Sector Insight...")
    sector_ai_container = st.empty()
    # セクター分析も「挨拶不要」「本題から」を徹底
    stream_ai_response(f"{sel_sec}の現状を需給・テクニカル・地政学含め日本語500文字以上で詳細分析せよ。挨拶不要。", sector_ai_container)

    # Stock List
    target_map = SECTOR_CATALOG[market][sel_sec]
    results = []
    for t, n in target_map.items():
        if t in px_map:
            c = px_map[t]["Close"]
            ret = calc_ret(c, win_days)
            results.append({"Name":n,"Ticker":t,"Price":float(c.iloc[-1]),"Return":ret,"df":c})
    
    if results:
        df_disp = pd.DataFrame(results).drop(columns=['df']).sort_values("Return", ascending=False, na_position='last')
        df_disp["Return"] = df_disp["Return"].map(lambda x: f"{x:+.1f}%" if x is not None else "N/A")
        st.dataframe(df_disp.set_index("Name"), use_container_width=True)

    # --- 9. DEEP RANKING ---
    st.divider()
    st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
    if st.button(f"🔍 EXECUTE {sel_sec} QUANT RANKING"):
        with st.status("Quant Engine Running...", expanded=True) as status:
            scored = []
            for r in results:
                c = r["df"].astype(float)
                r252, r21, r63 = calc_ret(c, 252), calc_ret(c, 21), calc_ret(c, 63)
                mom = (r252 - r21) if (r252 is not None and r21 is not None) else 0.0
                vol = (c.pct_change().rolling(60).std().iloc[-1]*252**0.5*100) if len(c)>=61 else 30
                score = round(mom*0.5 + (r['Return'] or 0)*0.3 + (30-vol)*0.2, 2)
                scored.append({"Name": r["Name"], "Ticker": r["Ticker"], "Score": score, "Momentum": mom, "Volatility": vol})
            
            ranked = sorted(scored, key=lambda x: x['Score'], reverse=True)
            st.dataframe(pd.DataFrame(ranked).set_index("Name"), use_container_width=True)
            
            status.write("Final AI Recommendation...")
            rank_ai_container = st.empty()
            stream_ai_response(f"Quant Analystとして上位銘柄の投資理由とリスクを日本語で詳細分析。Data: {json.dumps(ranked[:5], ensure_ascii=False)}", rank_ai_container)
            status.update(label="Analysis Complete", state="complete")
    st.markdown("</div>", unsafe_allow_html=True)