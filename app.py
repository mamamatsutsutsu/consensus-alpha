import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from google import genai
import json, re, threading, random
from io import StringIO
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xml.etree.ElementTree as ET

# --- 1. SPECTRE DESIGN (TRUE TURBO) ---
st.set_page_config(page_title="AlphaLens v7.8", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Fira Code', monospace; background-color: #05070a; color: #a9b1d6; }
    .stButton>button { width: 100%; border-radius: 2px; height: 3.5em; background: linear-gradient(90deg, #1a1b26, #24283b); color: #7aa2f7; border: 1px solid #7aa2f7; font-weight: bold; letter-spacing: 2px; }
    .glass-card { background: rgba(26, 27, 38, 0.7); padding: 15px; border-radius: 4px; border-left: 4px solid #7aa2f7; margin-bottom: 15px; }
    .alert-card { background: rgba(122, 162, 247, 0.1); border: 1px solid #7aa2f7; padding: 10px; border-radius: 4px; text-align: center; }
    .metric-val { color: #bb9af7; font-size: 1.1em; font-weight: bold; }
    .metric-label { color: #565f89; font-size: 0.75em; text-transform: uppercase; }
    .audit-box { font-size: 0.75em; color: #ff9e64; padding: 5px; border: 1px solid #ff9e64; border-radius: 4px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("ALPHALENS // TRUE.TURBO.v7.8")

# --- 2. CORE UTILITIES (THREAD-SAFE) ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()

def _get_session():
    if getattr(_thread_local, "session", None) is None:
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"User-Agent": "AlphaLens/7.8"})
    return _thread_local.session

def to_f(val):
    try:
        if val is None: return None
        v = float(val); return v if v == v else None
    except: return None

# --- 3. MASTER CATALOG (DEDUPED) ---
SECTOR_CATALOG = {
    "US MARKET (CORE 200)": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Google","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","BRK-B":"Berkshire","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM","ACN":"Accenture"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","INTC":"Intel","KLAC":"KLA","TER":"Teradyne","ON":"ON Semi","TXN":"TI","ADI":"Analog","NXPI":"NXP","MRVL":"Marvell","CDNS":"Cadence","SNPS":"Synopsys"},
        "Software / SaaS": {"SNOW":"Snowflake","PLTR":"Palantir","NOW":"ServiceNow","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","FTNT":"Fortinet","ZS":"Zscaler","OKTA":"Okta","TEAM":"Atlassian","ADSK":"Autodesk","SHOP":"Shopify","NET":"Cloudflare"},
        "Financials": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","AXP":"Amex","BLK":"BlackRock","C":"Citigroup","USB":"USBancorp","PNC":"PNC","BK":"BNY Mellon","CME":"CME","SPGI":"S&P Global","MCO":"Moody's","PYPL":"PayPal","WFC":"Wells Fargo"},
        "Healthcare / Bio": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo","BMY":"Bristol","AMGN":"Amgen","MDT":"Medtronic","BSX":"BostonSci","REGN":"Regeneron","ZTS":"Zoetis","SYK":"Stryker","DXCM":"Dexcom","ISRG":"Intuitive"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar","DE":"Deere","ETN":"Eaton","MMM":"3M","EMR":"Emerson","ITW":"ITW","UPS":"UPS","FDX":"FedEx","WM":"WasteMgmt"},
        "REITs / Real Estate": {"PLD":"Prologis","AMT":"AmericanTower","EQIX":"Equinix","PSA":"PublicStorage","O":"RealtyIncome","DLR":"DigitalRealty","VICI":"VICI","CCI":"CrownCastle","CBRE":"CBRE"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","KMI":"KinderMorgan","MPC":"Marathon","OXY":"Occidental","PSX":"Phillips66","HAL":"Halliburton","VLO":"Valero","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo"},
        "Consumer Staples": {"PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","PM":"PhilipMorris","MO":"Altria","CL":"Colgate","KMB":"Kimberly","GIS":"GeneralMills","KHC":"KraftHeinz"},
        "Consumer Disc": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","MAR":"Marriott","MCD":"McDonald's","TJX":"TJX","TGT":"Target"},
        "Communication / Media": {"DIS":"Disney","CHTR":"Charter","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CMCSA":"Comcast","PARA":"Paramount","FOXA":"Fox","WBD":"WarnerBros"},
        "Materials / Chem": {"LIN":"Linde","APD":"AirProducts","SHW":"Sherwin","ECL":"Ecolab","DOW":"Dow","DD":"DuPont","FCX":"Freeport","NEM":"Newmont","NUE":"Nucor"}
    },
    "JP MARKET (CORE 200)": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","4063":"信越化","7735":"スクリン","6526":"ソシオネ","6963":"ローム","7751":"キヤノン","6981":"村田製","6762":"TDK","6861":"キーエンス","6954":"ファナック","6503":"三菱電機"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ","4385":"メルカリ","3923":"ラクス","9613":"NTTデータ","2121":"MIXI","9735":"セコム"},
        "資本財/重工/防衛": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC","6504":"富士電機","6305":"日立建機","6113":"アマダ","6473":"ジェイテクト","6326":"クボタ"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","7205":"日野自","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO","8750":"第一生命","8309":"三井トラ"},
        "総合商社": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX"},
        "食品/必需品": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","2587":"サントリーBF","4911":"資生堂"},
        "ヘルスケア": {"4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4503":"アステラス","4523":"エーザイ","4901":"富士フイルム","7741":"HOYA","4543":"テルモ","4578":"大塚HD","4507":"塩野義"},
        "建設/不動産": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組","1928":"積水ハウス","1925":"大和ハウス"},
        "物流/鉄道/海運": {"9022":"JR東海","9020":"JR東日本","9021":"JR西日本","9101":"日本郵船","9104":"商船三井","9107":"川崎汽船","9064":"ヤマトHD","9005":"東急","9007":"小田急","9143":"SGHD"},
        "小売/サービス": {"9983":"ファーストリテ","3382":"セブン&アイ","7532":"パンパシ","8267":"イオン","9843":"ニトリHD","4661":"OLC","6098":"リクルート","9627":"アインHD"},
        "電力/ガス": {"9501":"東京電力","9502":"中部電力","9503":"関西電力","9531":"東京ガス","9532":"大阪ガス","9508":"九州電力","9506":"東北電力","9513":"電源開発"},
        "化学/素材": {"3407":"旭化成","4005":"住友化","4188":"三菱ケミ","6988":"日東電工","5401":"日本製鉄","5411":"JFE","5713":"住友鉱山","5802":"住友電工","3861":"王子HD"}
    }
}

# --- 4. THE TURBO ENGINE ---

def _stooq_candidates(t):
    base = t.replace(".", "-").lower()
    if t.upper() == "BRK-B": return ["brk-b", "brk.b"]
    return [base]

def fetch_px_single(t, suffix, max_retry=2):
    sess = _get_session()
    exts = [suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]
    for attempt in range(max_retry + 1):
        for sym in _stooq_candidates(t):
            for s in exts:
                try:
                    url = f"https://stooq.com/q/d/l/?s={sym}.{s}&i=d"
                    r = sess.get(url, timeout=5)
                    if r.status_code == 200 and r.text:
                        df = pd.read_csv(StringIO(r.text))
                        if "Close" in df.columns and not df.empty:
                            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                            return t, df.dropna(subset=["Date"]).set_index("Date").sort_index()
                except: continue
        time.sleep((0.3 * (2**attempt)) + random.random()*0.1)
    return t, None

@st.cache_data(ttl=3600)
def fetch_px_batch(tickers, suffix):
    # キャッシュを安定させるためにソートされたタプルを使用
    tickers = tuple(sorted(set(tickers)))
    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers}
        for fut in as_completed(futs):
            t, df = fut.result()
            if df is not None: results[t] = df
    return results

# --- 5. AUDIT & PULSE ---

def audit_catalog():
    audit = {}
    for u, gs in SECTOR_CATALOG.items():
        all_t = []
        for g, t_map in gs.items(): all_t.extend(list(t_map.keys()))
        audit[u] = {"total": len(all_t), "dupes": [x for x in set(all_t) if all_t.count(x) > 1]}
    return audit

@st.cache_data(ttl=1800)
def get_market_pulse(universe_key, horizon):
    suffix = "US" if "US" in universe_key else "JP"
    days = {"1W": 5, "1M": 21, "3M": 63}[horizon]
    all_needed = []
    for gs in SECTOR_CATALOG[universe_key].values(): all_needed.extend(list(gs.keys())[:10])
    
    # プログレスバーの実装
    prog_bar = st.progress(0, text="Fetching Pulse Samples...")
    batch_dfs = {}
    all_needed = list(set(all_needed))
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(fetch_px_single, t, suffix) for t in all_needed]
        for i, fut in enumerate(as_completed(futs)):
            t, df = fut.result()
            if df is not None: batch_dfs[t] = df
            prog_bar.progress((i+1)/len(all_needed), text=f"Scanning {t}...")
    prog_bar.empty()

    pulse_data = []
    for g_name, tickers in SECTOR_CATALOG[universe_key].items():
        rets = []
        for t in list(tickers.keys())[:10]:
            df = batch_dfs.get(t)
            if df is not None and len(df) > days:
                rets.append((df["Close"].iloc[-1] / df["Close"].iloc[-(days+1)] - 1) * 100)
        if rets:
            s = pd.Series(rets)
            pulse_data.append({"Group": g_name, "Median": float(s.median()), "WinRate": float((s>0).mean()*100), "N": len(rets)})
    return pd.DataFrame(pulse_data).sort_values("Median", ascending=False)

# --- 6. MAIN UI ---
audit_res = audit_catalog()
with st.sidebar:
    st.markdown("### 🔍 SYSTEM_AUDIT")
    for k, v in audit_res.items():
        st.markdown(f"<div class='audit-box'><b>{k}</b><br>Count: {v['total']}<br>Dupes: {len(v['dupes'])}</div>", unsafe_allow_html=True)
    st.markdown("---")
    universe_name = st.selectbox("UNIVERSE", list(SECTOR_CATALOG.keys()))
    pulse_span = st.radio("PULSE_WINDOW", ["1W", "1M", "3M"], index=1, horizontal=True)

st.markdown(f"### MARKET_PULSE / SECTOR_ROTATION ({pulse_span})")
pulse_df = get_market_pulse(universe_name, pulse_span)
fig_pulse = px.bar(pulse_df, x="Median", y="Group", orientation='h', color="Median", 
                   color_continuous_scale="RdYlGn", hover_data=["WinRate", "N"], labels={"Median": "Median Return (%)"})
st.plotly_chart(fig_pulse.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)

# 深掘りセクション
st.sidebar.markdown("---")
group_name = st.sidebar.selectbox("GROUP", list(SECTOR_CATALOG[universe_name].keys()))
span = st.sidebar.selectbox("HORIZON", ["3M", "1Y", "3Y"], index=1)
suffix = "US" if "US" in universe_name else "JP"

if st.sidebar.button("EXECUTE DEPTH SCAN"):
    target_tickers = list(SECTOR_CATALOG[universe_name][group_name].keys())
    with st.spinner("QUANT SCANNING..."):
        batch_dfs = fetch_px_batch(target_tickers, suffix)
        raw_results = []
        days = {"3M": 63, "1Y": 252, "3Y": 756}[span]
        all_norm = []
        for t, n in SECTOR_CATALOG[universe_name][group_name].items():
            df = batch_dfs.get(t)
            if df is not None:
                c = df["Close"].astype(float); ref = min(len(c), days + 1)
                r_now = to_f(c.iloc[-1]); r_start = to_f(c.iloc[-ref])
                ret = ((r_now / r_start - 1) * 100) if (r_now is not None and r_start) else None
                mom = to_f(((c.iloc[-1]/c.iloc[-252]-1)*100)-((c.iloc[-1]/c.iloc[-21]-1)*100)) if len(c)>=252 else None
                r21 = to_f((c.iloc[-1]/c.iloc[-22]-1)*100) if len(c)>=22 else None
                r63 = to_f((c.iloc[-1]/c.iloc[-64]-1)*100) if len(c)>=64 else None
                accel = (r21 - (r63/3)) if (r21 is not None and r63 is not None) else 0.0
                vol = to_f(c.pct_change().rolling(60).std().iloc[-1]*(252**0.5)*100) if len(c)>60 else None
                dd = to_f(((c.iloc[-ref:]/c.iloc[-ref:].cummax()-1)*100).min())
                norm = (c.iloc[-ref:]/c.iloc[-ref:].iloc[0])*100
                all_norm.append(norm.rename(t))
                raw_results.append({"ticker":t,"name":n,"price":r_now,"ret":ret,"mom":mom,"accel":accel,"vol":vol,"dd":dd,"hist":c.iloc[-ref:],"norm":norm})

        if all_norm:
            sector_ret = pd.concat(all_norm, axis=1).mean(axis=1).iloc[-1] - 100
            for r in raw_results:
                r["rs"] = r["ret"] - sector_ret if r["ret"] is not None else None
                s_mom = r["mom"] if r["mom"] is not None else 0.0
                s_vol = r["vol"] if r["vol"] is not None else 30.0
                s_rs = r["rs"] if r["rs"] is not None else 0.0
                r["score"] = round(s_mom*0.45 + (30-s_vol)*0.15 + s_rs*0.2 - abs(r["dd"] or 0)*0.2, 2)
            
            # (以下、AIレポート等の表示ロジックは継続)
            st.markdown("### ALPHA ALERTS")
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='alert-card'><div class='metric-label'>ACCEL TOP</div><div class='metric-val'>{max(raw_results, key=lambda x: x['accel'])['ticker']}</div></div>", unsafe_allow_html=True)
            valid_rs = [r for r in raw_results if r["rs"] is not None]
            c2.markdown(f"<div class='alert-card'><div class='metric-label'>RS TOP (pp)</div><div class='metric-val'>{max(valid_rs, key=lambda x: x['rs'])['ticker'] if valid_rs else 'N/A'}</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='alert-card'><div class='metric-label'>SECTOR RET</div><div class='metric-val'>{sector_ret:+.1f}%</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='alert-card'><div class='metric-label'>SUCCESS N</div><div class='metric-val'>{len(raw_results)}</div></div>", unsafe_allow_html=True)

            scatter_df = pd.DataFrame([{"Ticker":r["ticker"],"Vol":r["vol"],"Ret":r["ret"],"Score":r["score"],"Name":r["name"]} for r in raw_results if r["vol"] is not None])
            st.plotly_chart(px.scatter(scatter_df, x="Vol", y="Ret", text="Ticker", size=scatter_df["Score"].clip(lower=1), color="Score", color_continuous_scale="Viridis", hover_name="Name").update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)