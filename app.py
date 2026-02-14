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

# --- 1. SETTINGS & CSS ---
st.set_page_config(page_title="AlphaLens v10.1", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Fira Code', monospace; background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.2em; background: #21262d; color: #58a6ff; border: 1px solid #30363d; font-weight: bold; margin-bottom: 8px; }
    .kpi-container { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 20px; }
    .kpi-box { flex: 1; text-align: center; padding: 12px; background: #161b22; border-radius: 8px; border: 1px solid #30363d; }
    .kpi-label { font-size: 0.7em; color: #8b949e; text-transform: uppercase; }
    .kpi-value { font-size: 1.1em; font-weight: bold; color: #7aa2f7; }
    .sector-card { background: #161b22; padding: 12px; border-radius: 8px; border: 1px solid #30363d; margin-bottom: 8px; }
    .glass-card { background: rgba(33, 38, 45, 0.9); padding: 15px; border-radius: 8px; border-left: 4px solid #58a6ff; margin-bottom: 15px; }
    .metric-sub { font-size: 0.72em; color: #8b949e; }
    .news-tag { display: block; padding: 6px 10px; background: #0d1117; border-radius: 4px; font-size: 0.8em; margin: 4px 0; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE UTILITY ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.error("API KEY MISSING"); st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()
def _get_sess():
    if getattr(_thread_local, "session", None) is None:
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"User-Agent": "AlphaLens/10.1"})
    return _thread_local.session

def to_f(val):
    try:
        v = float(val)
        return v if v == v else None
    except: return None

# --- 3. MASTER CATALOG ---
# 本番運用では外部JSONからロードすることを推奨（ここでは代表銘柄を網羅した400規模構造を維持）
SECTOR_CATALOG = {
    "US MARKET": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Google","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","QCOM":"Qualcomm","VRT":"Vertiv"},
        "Software / SaaS": {"SNOW":"Snowflake","PLTR":"Palantir","NOW":"ServiceNow","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","ADSK":"Autodesk"},
        "Financials / Banking": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","BLK":"BlackRock","PYPL":"PayPal"},
        "Healthcare / Bio": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","ISRG":"Intuitive"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo"},
        "Consumer Disc": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","TGT":"Target"},
        "Communication / Media": {"DIS":"Disney","CHTR":"Charter","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CMCSA":"Comcast","PARA":"Paramount","WBD":"WarnerBros"},
        "REITs / Materials": {"PLD":"Prologis","AMT":"AmericanTower","EQIX":"Equinix","O":"RealtyIncome","LIN":"Linde","APD":"AirProducts","FCX":"Freeport"}
    },
    "JP MARKET": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","7735":"スクリン","6526":"ソシオネ"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ"},
        "重工業/防衛/建機": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7270":"SUBARU","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8630":"SOMPO"},
        "総合商社": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX"},
        "必需品/医薬": {"2802":"味の素","2914":"JT","2502":"アサヒ","4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4503":"アステラス","4901":"富士フイルム"},
        "建設/不動産": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組"}
    }
}

# --- 4. ENGINE (TRUE ACCURACY) ---

def fetch_px_single(t, suffix, max_retry=2):
    sess = _get_sess()
    sym = t.replace(".", "-").lower()
    cands = ["brk-b", "brk.b", sym] if t.upper() == "BRK-B" else [sym]
    exts = [suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]

    for attempt in range(max_retry + 1):
        for cand in cands:
            for s in exts:
                try:
                    url = f"https://stooq.com/q/d/l/?s={cand}.{s}&i=d"
                    r = sess.get(url, timeout=(3, 5))
                    if r.status_code != 200 or not r.text: continue
                    df = pd.read_csv(StringIO(r.text))
                    if df.empty or ("Date" not in df.columns) or ("Close" not in df.columns): continue
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
                    if not df.empty: return t, df.set_index("Date")[["Close"]]
                except: continue
        time.sleep((0.2 * (2 ** attempt)) + random.random() * 0.2)
    return t, None

@st.cache_data(ttl=1800)
def fetch_px_batch_cached(tickers_tuple, suffix):
    out, miss = {}, []
    # スマホ環境を考慮し並列数を調整（max_workers=6）
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers_tuple}
        for f in as_completed(futs):
            t, df = f.result()
            if df is not None: out[t] = df
            else: miss.append(t)
    return out, miss

@st.cache_data(ttl=1800)
def fetch_news_one(name, ticker, suffix):
    lang, gl, ceid = ("en-US","US","US:en") if suffix=="US" else ("ja-JP","JP","JP:ja")
    try:
        q = quote_plus(f"{name} {ticker}")
        url = f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={gl}&ceid={ceid}"
        items = ET.fromstring(requests.get(url, timeout=5).text).findall('.//item')[:3]
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in items]
    except: return []

def alpha_scoring(r):
    mom = r["mom"] if r["mom"] is not None else 0.0
    ret = r["ret"] if r["ret"] is not None else -99.0
    vol = r["vol"] if r["vol"] is not None else 30.0
    dd  = r["dd"]  if r["dd"]  is not None else 0.0
    accel = r["accel"] if r["accel"] is not None else 0.0
    # $$Alpha Score = (Mom_{12-1} \times 0.45) + (Ret_{3M} \times 0.2) + ((30 - Vol) \times 0.15) - |DD| \times 0.2 + Accel \times 0.05$$
    return round(mom*0.45 + ret*0.2 + (30-vol)*0.15 - abs(dd)*0.2 + accel*0.05, 2)

# --- 5. UI FLOW: ABSOLUTE SENTINEL ---

if 'market' not in st.session_state: st.session_state.market = "US MARKET"
if 'sector' not in st.session_state: st.session_state.sector = None

st.title("ALPHALENS // SENTINEL.v10.1")

# 1. HEADER KPI
m_cols = st.columns(2)
if m_cols[0].button("🇺🇸 US MARKET"): 
    st.session_state.market = "US MARKET"; st.session_state.sector = None
if m_cols[1].button("🇯🇵 JP MARKET"): 
    st.session_state.market = "JP MARKET"; st.session_state.sector = None

market = st.session_state.market
suffix = "US" if "US" in market else "JP"

# Pulse Data Fetch
all_pulse_needed = []
for gs in SECTOR_CATALOG[market].values(): all_pulse_needed.extend(list(gs.keys())[:4])
pulse_tickers = tuple(sorted(set(all_pulse_needed)))

with st.spinner("Syncing Pulse Radar..."):
    batch_dfs, miss_list = fetch_px_batch_cached(pulse_tickers, suffix)

n_ok = len(pulse_tickers) - len(miss_list)
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-box"><div class="kpi-label">Market</div><div class="kpi-value">{market.split()[0]}</div></div>
    <div class="kpi-box"><div class="kpi-label">Data Health</div><div class="kpi-value">{n_ok}/{len(pulse_tickers)}</div></div>
    <div class="kpi-box"><div class="kpi-label">Horizon</div><div class="kpi-value">1M Pulse</div></div>
</div>
""", unsafe_allow_html=True)

with st.expander("Diagnostics (Data Integrity Audit)", expanded=False):
    st.caption(f"Missed tickers in pulse: {len(miss_list)}")
    if miss_list: st.write(miss_list)

# 2. MARKET PULSE
rows = []
for g_name, tickers in SECTOR_CATALOG[market].items():
    rets = []
    for t in list(tickers.keys())[:4]:
        df = batch_dfs.get(t)
        if df is not None and len(df) > 21:
            r = (df["Close"].iloc[-1]/df["Close"].iloc[-22]-1)*100
            if pd.notna(r): rets.append(float(r))
    rows.append({
        "Sector": g_name, 
        "1M_Ret": (sum(rets)/len(rets)) if rets else None, 
        "N": len(rets)
    })

pulse_df = pd.DataFrame(rows).sort_values("1M_Ret", ascending=False, na_position="last")

st.write("### 🔥 SECTOR STRENGTH (1M)")

fig = px.bar(pulse_df.dropna(subset=["1M_Ret"]), x="1M_Ret", y="Sector", orientation='h', color="1M_Ret", color_continuous_scale="RdYlGn", hover_data=["N"])
st.plotly_chart(fig.update_layout(height=380, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

if not pulse_df.empty and pulse_df["N"].min() < 3:
    st.warning("⚠️ N<3 sectors detected. Data may be skewed.")

# Quick Access Chips
st.write("### ⚡ QUICK DRILL-DOWN")
top_3 = pulse_df["Sector"].tolist()[:3]
worst_1 = pulse_df["Sector"].tolist()[-1:]
chip_cols = st.columns(4)
for i, s_name in enumerate(top_3):
    if chip_cols[i].button(f"➡️ {s_name}"): st.session_state.sector = s_name
if chip_cols[3].button(f"🩸 {worst_1[0]}"): st.session_state.sector = worst_1[0]

# 3. SECTOR LEADERBOARD
st.divider()
selected_sec = st.selectbox("Or choose from list", ["---"] + list(SECTOR_CATALOG[market].keys()), 
                            index=0 if not st.session_state.sector else (["---"]+list(SECTOR_CATALOG[market].keys())).index(st.session_state.sector),
                            key="sector_select")

if selected_sec != "---":
    st.session_state.sector = selected_sec
    group_map = SECTOR_CATALOG[market][selected_sec]
    with st.spinner(f"Scanning {selected_sec}..."):
        detail_tickers = tuple(sorted(group_map.keys()))
        detail_batch, _ = fetch_px_batch_cached(detail_tickers, suffix)
        
        results = []
        for t, df in detail_batch.items():
            c = df["Close"].astype(float)
            # 3M リターン計算 (厳密に63営業日+1)
            ref_3m = 63 + 1
            ret = None
            if len(c) >= ref_3m:
                r_prev, r_now = float(c.iloc[-ref_3m]), float(c.iloc[-1])
                ret = ((r_now / r_prev) - 1) * 100 if r_prev != 0 else None
            
            mom = (((c.iloc[-1]/c.iloc[-252]-1)*100)-((c.iloc[-1]/c.iloc[-21]-1)*100)) if len(c)>=252 else None
            r21, r63 = (c.iloc[-1]/c.iloc[-22]-1)*100 if len(c)>=22 else None, (c.iloc[-1]/c.iloc[-64]-1)*100 if len(c)>=64 else None
            accel = (r21 - (r63/3)) if (r21 is not None and r63 is not None) else 0.0
            vol = (c.pct_change().rolling(60).std().iloc[-1]*(252**0.5)*100)
            vol = float(vol) if pd.notna(vol) else None
            dd = ((c.iloc[-64:]/c.iloc[-64:].cummax()-1)*100).min() if len(c)>=64 else None
            
            res = {"name":group_map[t], "ticker":t, "price":float(c.iloc[-1]), "ret":ret, "mom":mom, "accel":accel, "vol":vol, "dd":to_f(dd)}
            res["score"] = alpha_scoring(res)
            results.append(res)

    st.write(f"### 📊 {selected_sec} LEADERBOARD (by Score)")
    sorted_res = sorted(results, key=lambda x: x['score'], reverse=True)
    
    def render_card(r):
        ret_txt = f"{r['ret']:+.1f}%" if isinstance(r.get('ret'), (int,float)) else "N/A"
        mom_txt = f"{r['mom']:+.1f}%" if isinstance(r.get('mom'), (int,float)) else "N/A"
        accel_txt = f"{r['accel']:+.1f}" if isinstance(r.get('accel'), (int,float)) else "N/A"
        st.markdown(f"""
        <div class="sector-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <b>{r['name']} ({r['ticker']})</b>
                <span style="color:#7aa2f7; font-weight:bold;">Score: {r['score']}</span>
            </div>
            <div class="metric-sub">PX: {r['price']:.1f} | 3M Ret: {ret_txt} | MOM: {mom_txt} | Accel: {accel_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    for r in sorted_res[:10]: render_card(r)
    if len(sorted_res) > 10:
        with st.expander("Show more assets"):
            for r in sorted_res[10:]: render_card(r)

    # 4. DEEP ANALYSIS (FUSED)
    if st.button(f"🔍 ANALYZE {selected_sec} (AI × NEWS × QUANT)"):
        with st.spinner("AI Fusing Strategic Intel..."):
            top_3 = sorted_res[:3]
            news_bundle = []
            for r in top_3:
                news = fetch_news_one(r['name'], r['ticker'], suffix)
                news_bundle.append({"ticker":r['ticker'], "name":r['name'], "titles":[n['title'] for n in news[:2]], "links":news})
            
            sector_row = pulse_df[pulse_df["Sector"] == selected_sec].iloc[0].to_dict() if not pulse_df.empty else {}
            ai_payload = {
                "sector": selected_sec,
                "sector_pulse": sector_row,
                "top3": [{k:v for k,v in r.items()} for r in top_3],
                "news": [{"ticker":x['ticker'], "headlines":x['titles']} for x in news_bundle]
            }
            prompt = (
                f"あなたはヘッジファンドのQuant×裁量アナリスト。{selected_sec}セクターの地合い（市場内順位）を踏まえ、上位3銘柄を比較・分析せよ。\n"
                f"制約: ①数値根拠(score/ret/mom/accel/vol/dd)を引用 ②ニュース見出しを材料として結びつける ③結論：Top pickとその理由/リスクを日本語で。\n"
                f"Data: {json.dumps(ai_payload, ensure_ascii=False)}"
            )
            try:
                res = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
                st.markdown("### 🧠 STRATEGIC INTELLIGENCE BRIEF")
                st.markdown(f"<div class='glass-card'>{res.text}</div>", unsafe_allow_html=True)
            except: st.error("AI Intel Offline")

            st.markdown("### 📰 LATEST CATALYSTS")
            for x in news_bundle:
                st.write(f"**{x['name']} ({x['ticker']})**")
                if x["links"]:
                    for n in x["links"]: st.markdown(f"<div class='news-tag'><a href='{n['link']}' target='_blank'>{n['title']}</a></div>", unsafe_allow_html=True)
                else: st.write("No catalyst news found.")