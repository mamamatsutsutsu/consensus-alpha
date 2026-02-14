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

# --- 1. EXECUTIVE DARK DESIGN ---
st.set_page_config(page_title="AlphaLens v12.1", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Fira+Code:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #05070a; color: #e6edf3; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3.6em; background: #1f6feb; color: white; border: none; font-weight: bold; font-size: 1em; }
    .glass-card { background: rgba(22, 27, 34, 0.9); padding: 18px; border-radius: 8px; border-left: 5px solid #1f6feb; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    .metric-sub { font-family: 'Fira Code'; font-size: 0.82em; color: #8b949e; }
    .news-tag { display: block; padding: 6px 10px; background: #0d1117; border-radius: 6px; font-size: 0.85em; margin: 6px 0; border: 1px solid #30363d; }
    a { color: #58a6ff !important; text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ENGINE ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.error("GEMINI_API_KEY MISSING"); st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()
def _get_sess():
    if getattr(_thread_local, "session", None) is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "AlphaLens/12.1"})
        _thread_local.session = s
    return _thread_local.session

def safe_float(x):
    try:
        v = float(x)
        return v if v == v else None
    except: return None

def calc_ret(c: pd.Series, days: int):
    need = days + 1
    if c is None or len(c) < need: return None
    prev = safe_float(c.iloc[-need])
    now = safe_float(c.iloc[-1])
    if prev is None or now is None or prev == 0: return None
    return (now / prev - 1) * 100.0

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
def fetch_px_batch(tickers, suffix):
    out = {}
    tickers = tuple(tickers)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers}
        for f in as_completed(futs):
            t, df = f.result()
            if df is not None: out[t] = df
    return out

@st.cache_data(ttl=86400)
def fetch_fundamental_batch(tickers, suffix):
    tickers = tuple(tickers)
    def _one(t):
        try:
            tk = yf.Ticker(f"{t}.T" if suffix == "JP" else t)
            fi = getattr(tk, "fast_info", None)
            mcap = safe_float(fi.get("market_cap") if hasattr(fi, "get") else getattr(fi, "market_cap", None))
            info = tk.info
            return t, {"per": safe_float(info.get("trailingPE")), "pbr": safe_float(info.get("priceToBook")), "mcap": mcap or safe_float(info.get("marketCap"))}
        except: return t, {"per": None, "pbr": None, "mcap": None}
    out = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_one, t) for t in tickers]
        for f in as_completed(futs):
            k, v = f.result(); out[k] = v
    return out

@st.cache_data(ttl=3600)
def fetch_news_titles(name, ticker, suffix, k=2):
    hl, gl, ceid = ("en-US","US","US:en") if suffix=="US" else ("ja-JP","JP","JP:ja")
    try:
        q = quote_plus(f"{name} {ticker}")
        url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
        xml = requests.get(url, timeout=(3, 6)).text
        items = ET.fromstring(xml).findall('.//item')[:k]
        return [{"title": i.find('title').text, "link": i.find('link').text} for i in items if i.find('title') is not None]
    except: return []

@st.cache_data(ttl=1800)
def ai_sector_summary(market, sector):
    prompt = f"Hedge fund analystとして、現在の市場環境における{market}の{sector}セクター概況を日本語で3行以内で要約せよ。注目点を箇条書きで。"
    try: return client.models.generate_content(model="gemini-flash-latest", contents=prompt).text
    except: return None

# --- 3. MASTER CATALOG ---
SECTOR_CATALOG = {
    "US MARKET": {
        "Mega Tech": {"AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Google","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix"},
        "Semiconductors": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","INTC":"Intel"},
        "Software / SaaS": {"CRM":"Salesforce","ORCL":"Oracle","SNOW":"Snowflake","PLTR":"Palantir","NOW":"ServiceNow","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike"},
        "Financials": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","BLK":"BlackRock","AXP":"Amex"},
        "Healthcare": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo"},
        "Energy / Industrials": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","LMT":"Lockheed","RTX":"Raytheon","HON":"Honeywell","GE":"GE"},
        "Consumer / REITs": {"WMT":"Walmart","COST":"Costco","HD":"HomeDepot","NKE":"Nike","SBUX":"Starbucks","DIS":"Disney","PLD":"Prologis","AMT":"AmericanTower"}
    },
    "JP MARKET": {
        "半導体/ハイテク": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","7735":"スクリン","6526":"ソシオネ"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","6098":"リクルート"},
        "自動車/重工業": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8630":"SOMPO"},
        "商社/エネルギー": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX"},
        "必需品/医薬": {"2802":"味の素","2914":"JT","2502":"アサヒ","2501":"サッポロ","4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4523":"エーザイ"},
        "不動産/鉄道/電力": {"8801":"三井不動","8802":"三菱地所","9022":"JR東海","9020":"JR東日本","9101":"日本郵船","9104":"商船三井","9501":"東京電力","9503":"関西電力"}
    }
}

# --- 4. MAIN FLOW ---
st.title("ALPHALENS v12.1 // ABSOLUTE")

col_m, col_w = st.columns(2)
market = col_m.radio("MARKET", ["US MARKET", "JP MARKET"], horizontal=True)
window = col_w.radio("WINDOW", ["1W", "1M", "3M"], index=1, horizontal=True)
win_days = {"1W": 5, "1M": 21, "3M": 63}[window]
suffix = "US" if "US" in market else "JP"

# 1. MARKET PULSE
with st.spinner("Syncing Market Pulse..."):
    all_pulse_tickers = []
    for gs in SECTOR_CATALOG[market].values(): all_pulse_tickers.extend(list(gs.keys())[:4])
    pulse_batch = fetch_px_batch(tuple(sorted(set(all_pulse_tickers))), suffix)
    
    rows = []
    for g_name, tickers in SECTOR_CATALOG[market].items():
        rets = [calc_ret(pulse_batch[t]["Close"], win_days) for t in list(tickers.keys())[:4] if t in pulse_batch]
        rets = [r for r in rets if r is not None]
        rows.append({"Sector": g_name, "Return": (sum(rets)/len(rets)) if rets else None, "N": len(rets)})

pulse_df = pd.DataFrame(rows).sort_values("Return", ascending=False, na_position="last")
st.plotly_chart(px.bar(pulse_df, x="Return", y="Sector", orientation="h", color="Return", color_continuous_scale="RdYlGn", hover_data=["N"])
                .update_layout(height=300, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0)), use_container_width=True)

st.divider()
selected_sector = st.selectbox("📁 SELECT SECTOR", ["---"] + list(SECTOR_CATALOG[market].keys()))

if selected_sector != "---":
    # AI Summary
    with st.spinner(f"AI summarizing {selected_sector}..."):
        st.markdown(f"<div class='glass-card'><b>Sector Intel:</b><br>{ai_sector_summary(market, selected_sector) or 'N/A'}</div>", unsafe_allow_html=True)

    # Stock List
    target_map = SECTOR_CATALOG[market][selected_sector]
    with st.spinner("Fetching Sector Assets..."):
        px_batch = fetch_px_batch(target_map.keys(), suffix)
        fund_batch = fetch_fundamental_batch(target_map.keys(), suffix)
        
        results = []
        for t, n in target_map.items():
            if t in px_batch:
                df = px_batch[t]; f = fund_batch.get(t, {})
                results.append({"ticker": t, "name": n, "price": safe_float(df["Close"].iloc[-1]), "ret": calc_ret(df["Close"], win_days), "mcap": f.get("mcap"), "per": f.get("per"), "pbr": f.get("pbr"), "df": df})
        
        disp = []
        for r in sorted(results, key=lambda x: (x["ret"] is not None, x["ret"]), reverse=True):
            m = r["mcap"]
            m_txt = f"{m/1e9:.1f}B" if m and suffix=="US" else (f"{m/1e12:.2f}T" if m else "N/A")
            disp.append({"Name": r["name"], "Ticker": r["ticker"], "Price": f"{r['price']:.2f}" if r['price'] else "N/A", f"Ret({window})": f"{r['ret']:+.1f}%" if r['ret'] else "N/A", "MCap": m_txt, "PER": f"{r['per']:.1f}" if r['per'] else "N/A", "PBR": f"{r['pbr']:.1f}" if r['pbr'] else "N/A"})
        st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)

    # Deep Analysis
    if st.button(f"🔍 EXECUTE RANKING ANALYSIS: {selected_sector}"):
        with st.spinner("Quant Ranking..."):
            scored = []
            for r in results:
                c = r["df"]["Close"].astype(float)
                mom = (calc_ret(c, 252) - calc_ret(c, 21)) if len(c) >= 253 else 0
                accel = (calc_ret(c, 21) - (calc_ret(c, 63)/3)) if (len(c) >= 64) else 0
                vol = (c.pct_change().rolling(60).std().iloc[-1]*(252**0.5)*100) if len(c) >= 61 else 30
                dd = (c.iloc[-64:]/c.iloc[-64:].cummax()-1).min()*100 if len(c) >= 64 else 0
                score = round((mom or 0)*0.4 + (r['ret'] or 0)*0.2 + (30-vol)*0.2 - abs(dd or 0)*0.2 + (accel or 0)*0.05, 2)
                scored.append({**r, "score": score, "mom": mom, "accel": accel, "vol": vol, "dd": dd})
            
            ranked = sorted(scored, key=lambda x: x["score"], reverse=True)
            st.markdown("### 📈 QUANT RANKING")
            st.dataframe(pd.DataFrame([{k:v for k,v in x.items() if k!='df'} for x in ranked]), use_container_width=True, hide_index=True)

            news = []
            for x in ranked[:3]:
                n_items = fetch_news_titles(x["name"], x["ticker"], suffix)
                news.append({"ticker": x["ticker"], "name": x["name"], "links": n_items, "titles": [i["title"] for i in n_items]})
            
            try:
                report = client.models.generate_content(model="gemini-flash-latest", contents=f"Hedge fund analystとして分析せよ。Data: {json.dumps({'sector':selected_sector, 'top':ranked[:10], 'news':news}, ensure_ascii=False)}").text
                st.markdown(f"### 🧠 STRATEGIC REPORT\n<div class='glass-card' style='border-left-color:#ffd700'>{report}</div>", unsafe_allow_html=True)
            except: st.error("AI Offline")