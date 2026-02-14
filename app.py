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

# --- 1. DESIGN: AI-CORE CYBER TERMINAL ---
st.set_page_config(page_title="AlphaLens v20.1", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&family=Orbitron:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Fira Code', monospace; background-color: #05070a; color: #00f2ff; }
    
    .header-kpi { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
    .kpi-box { flex: 1; min-width: 140px; background: rgba(18, 22, 33, 0.95); padding: 15px; border-radius: 8px; border: 1px solid #bc13fe; text-align: center; box-shadow: 0 0 15px rgba(188, 19, 254, 0.2); }
    .kpi-label { font-size: 0.6em; color: #bc13fe; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-val { font-size: 1.05em; font-weight: bold; color: #00f2ff; font-family: 'Orbitron'; }
    
    .stButton>button { 
        width: 100%; border-radius: 4px; height: 3.1em; 
        background: #0d1117; color: #00f2ff; border: 1px solid #00f2ff; 
        font-weight: bold; transition: 0.25s;
    }
    .stButton>button:hover { background: #00f2ff; color: #05070a; box-shadow: 0 0 15px #00f2ff; }
    
    .action-btn>div>button { 
        background: linear-gradient(90deg, #bc13fe, #7000ff) !important; 
        color: white !important; height: 4.2em !important; border: none !important;
    }

    .glass-card { background: rgba(18, 22, 33, 0.95); padding: 20px; border-radius: 12px; border-left: 6px solid #bc13fe; margin-bottom: 20px; line-height: 1.8; color: #e6edf3; }
    .intel-header { color: #bc13fe; font-family: 'Orbitron'; font-size: 1.1em; margin-bottom: 10px; border-bottom: 1px solid #bc13fe; padding-bottom: 5px; }
    .news-tag { display: block; padding: 10px; background: #1a1f2b; border-radius: 6px; font-size: 0.85em; margin: 6px 0; border: 1px solid #30363d; text-decoration: none !important; }
    .metric-sub { font-size: 0.72em; color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE (PRECISION & TURBO) ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.error("API KEY MISSING"); st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()
def _get_sess():
    if getattr(_thread_local, "session", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "AlphaLens/20.1"})
        _thread_local.session = s
    return _thread_local.session

def safe_float(x):
    try:
        v = float(x)
        return v if v == v else None
    except: return None

def calc_ret(c: pd.Series, days: int):
    if c is None or len(c) < days + 1: return None
    prev = safe_float(c.iloc[-(days+1)])
    now = safe_float(c.iloc[-1])
    if prev is None or now is None or prev == 0: return None
    return (now / prev - 1) * 100.0

def rsi14(close: pd.Series):
    if close is None or len(close) < 20: return None
    d = close.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    denom = dn.iloc[-1]
    if not denom or denom != denom: return None
    rs = up.iloc[-1] / denom
    return 100 - (100 / (1 + rs))

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
                    r = sess.get(url, timeout=(3, 6))
                    if r.status_code == 200:
                        df = pd.read_csv(StringIO(r.text))
                        if not df.empty and "Date" in df.columns and "Close" in df.columns:
                            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                            df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
                            if not df.empty: return t, df.set_index("Date")[["Close"]]
                except: continue
        time.sleep((0.2 * (2 ** attempt)) + random.random() * 0.1)
    return t, None

@st.cache_data(ttl=1800)
def fetch_px_batch(tickers_tuple, suffix):
    # tickers_tuple は必ず sorted 済みで渡す
    out = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers_tuple}
        for f in as_completed(futs):
            t, df = f.result()
            if df is not None: out[t] = df
    return out

@st.cache_data(ttl=3600)
def fetch_fund_one(ticker, suffix):
    try:
        tk = yf.Ticker(f"{ticker}.T" if suffix == "JP" else ticker)
        fi = getattr(tk, "fast_info", None)
        mcap = safe_float(fi.get("market_cap") if hasattr(fi, "get") else getattr(fi, "market_cap", None))
        info = tk.info
        return {"per": safe_float(info.get("trailingPE")), "pbr": safe_float(info.get("priceToBook")), "mcap": mcap or safe_float(info.get("marketCap")), "info": info}
    except: return {"per": None, "pbr": None, "mcap": None, "info": {}}

@st.cache_data(ttl=1200)
def fetch_news(name, ticker, suffix, k=8):
    lang, gl, ceid = ("en-US","US","US:en") if suffix=="US" else ("ja-JP","JP","JP:ja")
    try:
        q = quote_plus(f"{name} {ticker}")
        url = f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={gl}&ceid={ceid}"
        xml = requests.get(url, timeout=5).text
        items = ET.fromstring(xml).findall('.//item')[:k]
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in items if i.find('title') is not None]
    except: return []

# --- 3. MASTER CATALOG (418 ASSETS) ---
SECTOR_CATALOG = {
    "US MARKET": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Alphabet","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM","DELL":"Dell","HPE":"HPE","HPQ":"HP","ACN":"Accenture","CSCO":"Cisco","NOW":"ServiceNow","INTC":"Intel"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","KLAC":"KLA","TER":"Teradyne","ON":"ON Semi","TXN":"TI","ADI":"Analog","NXPI":"NXP","MRVL":"Marvell","CDNS":"Cadence","SNPS":"Synopsys","VRT":"Vertiv","SMCI":"Supermicro"},
        "Software / SaaS / Cyber": {"SNOW":"Snowflake","PLTR":"Palantir","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","FTNT":"Fortinet","ZS":"Zscaler","OKTA":"Okta","TEAM":"Atlassian","ADSK":"Autodesk","SHOP":"Shopify","NET":"Cloudflare","CHKP":"CheckPoint","MDB":"MongoDB","U":"Unity","TWLO":"Twilio","SPLK":"Splunk"},
        "Financials": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","AXP":"Amex","BLK":"BlackRock","C":"Citigroup","USB":"USBancorp","PNC":"PNC","BK":"BNY Mellon","CME":"CME","SPGI":"S&P Global","MCO":"Moody's","PYPL":"PayPal","WFC":"Wells Fargo","SCHW":"Schwab"},
        "Healthcare / Bio": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo","BMY":"Bristol","AMGN":"Amgen","MDT":"Medtronic","BSX":"BostonSci","REGN":"Regeneron","ZTS":"Zoetis","SYK":"Stryker","DXCM":"Dexcom","GILD":"Gilead","ISRG":"Intuitive","ABT":"Abbott"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar","DE":"Deere","ETN":"Eaton","MMM":"3M","EMR":"Emerson","ITW":"ITW","UPS":"UPS","FDX":"FedEx","WM":"WasteMgmt","NSC":"Norfolk","LUV":"Southwest"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","KMI":"KinderMorgan","MPC":"Marathon","OXY":"Occidental","PSX":"Phillips66","HAL":"Halliburton","VLO":"Valero","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo","EXC":"Exelon","D":"Dominion"},
        "Consumer Disc": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","MAR":"Marriott","MCD":"McDonald's","TJX":"TJX","TGT":"Target","F":"Ford","GM":"GM","EBAY":"eBay","LULU":"Lululemon","CLX":"Clorox"},
        "Consumer Staples": {"PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","PM":"PhilipMorris","MO":"Altria","CL":"Colgate","KMB":"Kimberly","GIS":"GeneralMills","KHC":"KraftHeinz","MDLZ":"Mondelez","EL":"EsteeLauder"},
        "Comm / Media": {"DIS":"Disney","CHTR":"Charter","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CMCSA":"Comcast","PARA":"Paramount","FOXA":"Fox","WBD":"WarnerBros"},
        "Materials / REITs": {"LIN":"Linde","APD":"AirProducts","SHW":"Sherwin","ECL":"Ecolab","DOW":"Dow","DD":"DuPont","FCX":"Freeport","NEM":"Newmont","PLD":"Prologis","AMT":"AmericanTower","EQIX":"Equinix","PSA":"PublicStorage","O":"RealtyIncome","VICI":"VICI","CCI":"CrownCastle"}
    },
    "JP MARKET": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","7735":"スクリン","6526":"ソシオネ","6963":"ローム","7751":"キヤノン","6981":"村田製","6762":"TDK","6861":"キーエンス","6954":"ファナック","6503":"三菱電機","6501":"日立","6504":"富士電機","6869":"シスメックス","6758":"ソニーG"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ","6098":"リクルート","4385":"メルカリ","3923":"ラクス","9613":"NTTデータ","2121":"MIXI","3632":"グリー","9735":"セコム"},
        "重工業/防衛/建機": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC","6305":"日立建機","6113":"アマダ","6473":"ジェイテクト","6326":"クボタ","7003":"三井E&S","7014":"名村造船"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","7205":"日野自","7211":"三菱自","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO","8750":"第一生命","8309":"三井トラ","8473":"SBI","8601":"大和証"},
        "総合商社/エネルギー": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX","5020":"ENEOS","1518":"三井松島"},
        "必需品/医薬/化学": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","4911":"資生堂","4901":"富士フイルム","4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4503":"アステラス","4523":"エーザイ","4063":"信越化","3407":"旭化成"},
        "不動産/鉄道/電力": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組","1928":"積水ハウス","1925":"大和ハウス","1878":"大東建","9022":"JR東海","9020":"JR東日本","9101":"日本郵船","9104":"商船三井","9501":"東京電力","9503":"関西電力"}
    }
}

# --- 4. UI STATE & EXECUTION ---

if "m_choice" not in st.session_state: st.session_state.m_choice = "US MARKET"
if "s_choice" not in st.session_state: st.session_state.s_choice = None

st.title("ALPHALENS v20.1 // OMNISCIENCE")

# Header KPI (Strict Data)
total_a = sum(len(s) for s in SECTOR_CATALOG[st.session_state.m_choice].values())
st.markdown(f"""
<div class="header-kpi">
    <div class="kpi-box"><div class="kpi-label">Market</div><div class="kpi-val">{st.session_state.m_choice.split()[0]}</div></div>
    <div class="kpi-box"><div class="kpi-label">Audit: Total Assets</div><div class="kpi-val">{total_a}</div></div>
    <div class="kpi-box"><div class="kpi-label">Horizon</div><div class="kpi-val">1M Rolling</div></div>
    <div class="kpi-box"><div class="kpi-label">Status</div><div class="kpi-val">Ready</div></div>
</div>
""", unsafe_allow_html=True)

# Selection Window
c_m1, c_m2, c_w = st.columns([1,1,2])
if c_m1.button("🇺🇸 US CLUSTER"): st.session_state.m_choice = "US MARKET"; st.session_state.s_choice = None; st.rerun()
if c_m2.button("🇯🇵 JP CLUSTER"): st.session_state.m_choice = "JP MARKET"; st.session_state.s_choice = None; st.rerun()
window_label = c_w.radio("WINDOW", ["1W (5d)", "1M (21d)", "3M (63d)"], index=1, horizontal=True)
win_days = 5 if "1W" in window_label else (21 if "1M" in window_label else 63)

market = st.session_state.m_choice
suffix = "US" if "US" in market else "JP"

# --- 5. PULSE: SECTOR MOMENTUM (DETERMINISTIC CACHING) ---
st.subheader("🔥 SECTOR POWER HEATMAP (ALL-ASSET AVERAGE)")
all_needed = []
for gs in SECTOR_CATALOG[market].values(): all_needed.extend(list(gs.keys()))
# 重要：sorted() を使ってキャッシュキーを固定
pulse_tickers = tuple(sorted(set(all_needed)))

with st.spinner("Synchronizing Quantum Stream..."):
    px_map = fetch_px_batch(pulse_tickers, suffix)

pulse_rows = []
for g_name, tickers in SECTOR_CATALOG[market].items():
    rets = [calc_ret(px_map[t]["Close"], win_days) for t in tickers.keys() if t in px_map]
    rets = [r for r in rets if r is not None]
    pulse_rows.append({"Sector": g_name, "Avg_Ret": (sum(rets)/len(rets)) if rets else None, "N": len(rets)})

pulse_df = pd.DataFrame(pulse_rows).sort_values("Avg_Ret", ascending=False, na_position="last")
fig = px.bar(pulse_df, x="Avg_Ret", y="Sector", orientation='h', color="Avg_Ret", color_continuous_scale="RdYlGn", hover_data=["N"])
st.plotly_chart(fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

# Quick Access Chips
st.write("### 📂 ACTIVATE SECTOR DATA-STREAM")
chips = pulse_df["Sector"].dropna().tolist()[:9]
s_cols = st.columns(3)
for i, s_name in enumerate(chips):
    if s_cols[i % 3].button(f"💠 {s_name}"):
        st.session_state.s_choice = s_name
        st.rerun()

st.divider()

# --- 6. SECTOR DEPTH UNIT (TIERED LOADING) ---
if st.session_state.s_choice:
    sel_sec = st.session_state.s_choice
    st.subheader(f"📍 {sel_sec} INTELLIGENCE UNIT")
    
    # A) AI Sector Brief (Cached, Direct Volume)
    with st.spinner("AI drafting strategy..."):
        prompt = f"{sel_sec}の現状を需給・地政学・金融政策・テクニカルまで含めて日本語詳細分析せよ。挨拶不要。いきなり本題から書け。"
        try:
            sector_intel = client.models.generate_content(model="gemini-flash-latest", contents=prompt).text
            st.markdown(f"<div class='glass-card'><div class='intel-header'>SECTOR BRIEF</div>{sector_intel}</div>", unsafe_allow_html=True)
        except: st.info("AI Stream Interrupted.")

    # B) Stock List (Immediate)
    target_map = SECTOR_CATALOG[market][sel_sec]
    results = []
    for t, n in target_map.items():
        if t in px_map:
            c = px_map[t]["Close"].astype(float)
            results.append({"ticker": t, "name": n, "price": float(c.iloc[-1]), "ret_1m": calc_ret(c, 21), "df": px_map[t]})
    
    st.write("### 💠 ASSET LEADERBOARD (PRICE/RET)")
    base_disp = []
    for r in sorted(results, key=lambda x: (x["ret_1m"] is not None, x["ret_1m"]), reverse=True):
        base_disp.append({"Name": r["name"], "Ticker": r["ticker"], "Price": f"{r['price']:.2f}", "Ret(1M)": f"{r['ret_1m']:+.1f}%" if r['ret_1m'] is not None else "N/A"})
    st.dataframe(pd.DataFrame(base_disp), use_container_width=True, hide_index=True)

    # C) On-Demand Multiples
    st.write("### 💠 FINANCIAL MULTIPLES (TIER-1 LOAD)")
    top_list = sorted(results, key=lambda x: (x["ret_1m"] is not None, x["ret_1m"]), reverse=True)[:12]
    with st.spinner("Loading Multiples..."):
        m_rows = []
        for r in top_list:
            f = fetch_fund_one(r["ticker"], suffix)
            m = f.get("mcap")
            m_txt = "N/A" if not m else (f"{m/1e9:.1f}B" if suffix=="US" else f"{m/1e12:.2f}T")
            m_rows.append({"Name": r["name"], "MCap": m_txt, "PER": f"{f.get('per'):.1f}" if f.get('per') else "N/A", "PBR": f"{f.get('pbr'):.1f}" if f.get('pbr') else "N/A"})
        st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

    # D) Individual Omniscience
    st.divider()
    st.write("### 🔍 INDIVIDUAL ASSET OMNISCIENCE")
    asset_names = ["---"] + [r["name"] for r in results]
    pick = st.selectbox("Pick an asset to drill down", asset_names)
    
    if pick != "---":
        asset = next(x for x in results if x["name"] == pick)
        with st.spinner(f"Omniscient Scan: {pick}..."):
            c = asset["df"]["Close"].astype(float)
            ma20 = c.rolling(20).mean().iloc[-1] if len(c) >= 20 else None
            dist_ma20 = (asset["price"]/ma20 - 1)*100 if ma20 else None
            rsi = rsi14(c)
            f = fetch_fund_one(asset["ticker"], suffix)
            news = fetch_news(asset["name"], asset["ticker"], suffix)
            
            st.markdown(f"""
            <div class='glass-card' style='border-left: 6px solid #00f2ff;'>
              <div class='intel-header'>{pick} ({asset['ticker']}) // OMNI REPORT</div>
              <p><b>■ 企業概要:</b><br>{(f['info'].get('longBusinessSummary','N/A'))[:650]}...</p>
              <div style='display:flex; gap:10px; flex-wrap:wrap; background:rgba(0,242,255,0.05); padding:12px; border-radius:10px;'>
                <div><span class='metric-sub'>MA20乖離</span><br><b>{f"{dist_ma20:+.2f}%" if dist_ma20 is not None else "N/A"}</b></div>
                <div><span class='metric-sub'>RSI(14)</span><br><b>{f"{rsi:.1f}" if rsi is not None else "N/A"}</b></div>
                <div><span class='metric-sub'>目標株価</span><br><b>{f['info'].get('targetMeanPrice','N/A')}</b></div>
                <div><span class='metric-sub'>推奨</span><br><b style='color:#bc13fe;'>{str(f['info'].get('recommendationKey','N/A')).upper()}</b></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                ai_in = {"fundamentals": {"per": f['per'], "pbr": f['pbr']}, "technicals": {"rsi14": rsi, "dist_ma20": dist_ma20}, "news": [n['title'] for n in news[:5]]}
                rep = client.models.generate_content(model="gemini-flash-latest", contents=f"投資論点のみ日本語で詳細分析せよ。Data: {json.dumps(ai_in, ensure_ascii=False)}").text
                st.markdown(f"<div class='glass-card'>{rep}</div>", unsafe_allow_html=True)
            except: st.info("AI logic offline.")
            for n in news: st.markdown(f"<a href='{n['link']}' target='_blank' class='news-tag'>🔗 {n['title']}</a>", unsafe_allow_html=True)

    # E) Ranking Engine
    st.divider()
    st.markdown("<div class='action-btn'>", unsafe_allow_html=True)
    if st.button(f"🔍 EXECUTE STRATEGIC RANKING: {sel_sec}"):
        with st.spinner("Calculating Alpha Score..."):
            scored = []
            for r in results:
                c = r["df"]["Close"].astype(float)
                mom = (calc_ret(c, 252) - calc_ret(c, 21)) if len(c) >= 253 else 0
                vol = (c.pct_change().rolling(60).std().iloc[-1]*252**0.5*100) if len(c) >= 61 else 30
                score = round((mom or 0)*0.5 + (r['ret_1m'] or 0)*0.3 + (30-vol)*0.2, 2)
                scored.append({"name": r["name"], "ticker": r["ticker"], "price": r["price"], "score": score, "mom_12_1": mom, "vol_60d": vol})
            ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
            st.dataframe(pd.DataFrame(ranked), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)