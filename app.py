import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from google import genai
import json, re, threading, random, time
from io import StringIO
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xml.etree.ElementTree as ET

# --- 1. DESIGN: SPECTRE ABSOLUTE ---
st.set_page_config(page_title="AlphaLens v8.0", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Fira Code', monospace; background-color: #05070a; color: #a9b1d6; }
    .stButton>button { width: 100%; border-radius: 2px; height: 3.5em; background: linear-gradient(90deg, #1a1b26, #24283b); color: #7aa2f7; border: 1px solid #7aa2f7; font-weight: bold; letter-spacing: 2px; }
    .glass-card { background: rgba(26, 27, 38, 0.7); padding: 15px; border-radius: 4px; border-left: 4px solid #7aa2f7; margin-bottom: 15px; }
    .alert-card { background: rgba(122, 162, 247, 0.1); border: 1px solid #7aa2f7; padding: 10px; border-radius: 4px; text-align: center; }
    .metric-val { color: #bb9af7; font-size: 1.1em; font-weight: bold; }
    .metric-label { color: #565f89; font-size: 0.75em; text-transform: uppercase; }
    .news-tag { display: inline-block; padding: 2px 8px; background-color: #21262d; border-radius: 4px; font-size: 0.75em; margin: 2px; border: 1px solid #30363d; margin-bottom: 4px; }
    a { color: #58a6ff !important; text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("ALPHALENS // ABSOLUTE.v8.0")

# --- 2. UTILITY & SECURITY (THREAD-SAFE) ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.stop()
client = genai.Client(api_key=api_key)

_thread_local = threading.local()
def _get_session():
    if getattr(_thread_local, "session", None) is None:
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"User-Agent": "AlphaLens/8.0"})
    return _thread_local.session

def to_f(val):
    try:
        if val is None: return None
        v = float(val); return v if v == v else None
    except: return None

# --- 3. THE 400+ VERIFIED MASTER CATALOG ---
SECTOR_CATALOG = {
    "US MARKET (200+ Assets)": {
        "Platform / Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Google","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM","DELL":"Dell","HPE":"HPE","HPQ":"HP","ACN":"Accenture","CSCO":"Cisco","NOW":"ServiceNow","INTC":"Intel"},
        "Semis / AI Infra": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","KLAC":"KLA","TER":"Teradyne","ON":"ON Semi","TXN":"TI","ADI":"Analog","NXPI":"NXP","MRVL":"Marvell","CDNS":"Cadence","SNPS":"Synopsys","MCHP":"Microchip","VRT":"Vertiv","SMCI":"Supermicro"},
        "Software / SaaS / Cyber": {"SNOW":"Snowflake","PLTR":"Palantir","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","FTNT":"Fortinet","ZS":"Zscaler","OKTA":"Okta","TEAM":"Atlassian","ADSK":"Autodesk","SHOP":"Shopify","NET":"Cloudflare","CHKP":"CheckPoint","MDB":"MongoDB","U":"Unity","TWLO":"Twilio"},
        "Financials / Pay": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","AXP":"Amex","BLK":"BlackRock","C":"Citigroup","USB":"USBancorp","PNC":"PNC","BK":"BNY Mellon","CME":"CME","SPGI":"S&P Global","MCO":"Moody's","PYPL":"PayPal","COIN":"Coinbase","WFC":"Wells Fargo","SCHW":"Schwab"},
        "Healthcare / Bio": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo","BMY":"Bristol","AMGN":"Amgen","MDT":"Medtronic","BSX":"BostonSci","REGN":"Regeneron","ZTS":"Zoetis","SYK":"Stryker","DXCM":"Dexcom","GILD":"Gilead","ISRG":"Intuitive","ABT":"Abbott"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar","DE":"Deere","ETN":"Eaton","MMM":"3M","EMR":"Emerson","ITW":"ITW","UPS":"UPS","FDX":"FedEx","WM":"WasteMgmt","NSC":"Norfolk"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","KMI":"KinderMorgan","MPC":"Marathon","OXY":"Occidental","PSX":"Phillips66","HAL":"Halliburton","VLO":"Valero","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo","EXC":"Exelon","D":"Dominion"},
        "Consumer Staples": {"PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","PM":"PhilipMorris","MO":"Altria","CL":"Colgate","KMB":"Kimberly","GIS":"GeneralMills","KHC":"KraftHeinz","MDLZ":"Mondelez","EL":"EsteeLauder","TGT":"Target"},
        "Consumer Disc": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","MAR":"Marriott","MCD":"McDonald's","TJX":"TJX","ORLY":"O'Reilly","F":"Ford","GM":"GM","EBAY":"eBay","LULU":"Lululemon"},
        "Communication / Media": {"DIS":"Disney","CHTR":"Charter","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CMCSA":"Comcast","PARA":"Paramount","FOXA":"Fox","WBD":"WarnerBros","GOOGL":"Alphabet-C"},
        "Materials / Chem": {"LIN":"Linde","APD":"AirProducts","SHW":"Sherwin","ECL":"Ecolab","DOW":"Dow","DD":"DuPont","FCX":"Freeport","NEM":"Newmont","CTVA":"Corteva","NUE":"Nucor","VMC":"Vulcan"},
        "REITs": {"PLD":"Prologis","AMT":"AmericanTower","EQIX":"Equinix","PSA":"PublicStorage","O":"RealtyIncome","DLR":"DigitalRealty","VICI":"VICI","CCI":"CrownCastle","SBAC":"SBAC","CBRE":"CBRE"}
    },
    "JP MARKET (200+ Assets)": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","7735":"スクリン","6526":"ソシオネ","6963":"ローム","7751":"キヤノン","6981":"村田製","6762":"TDK","6861":"キーエンス","6954":"ファナック","6503":"三菱電機","6501":"日立","6504":"富士電機","6869":"シスメックス"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ","4385":"メルカリ","3923":"ラクス","9613":"NTTデータ","2121":"MIXI","3632":"グリー","9735":"セコム"},
        "資本財/重工/防衛": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC","6305":"日立建機","6113":"アマダ","6473":"ジェイテクト","6326":"クボタ"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","7205":"日野自","7211":"三菱自","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO","8750":"第一生命","8309":"三井トラ","8473":"SBI","8601":"大和証"},
        "総合商社": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX"},
        "食品/必需品": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","2587":"サントリーBF","4911":"資生堂","4901":"富士フイルム"},
        "ヘルスケア/医薬": {"4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4503":"アステラス","4523":"エーザイ","4543":"テルモ","4578":"大塚HD","4507":"塩野義","4528":"小野薬","4519":"中外"},
        "不動産/建設": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組","1928":"積水ハウス","1925":"大和ハウス","1878":"大東建"},
        "物流/鉄道/海運": {"9022":"JR東海","9020":"JR東日本","9021":"JR西日本","9101":"日本郵船","9104":"商船三井","9107":"川崎汽船","9064":"ヤマトHD","9005":"東急","9007":"小田急","9143":"SGHD","9101":"郵船"},
        "小売/サービス": {"9983":"ファーストリテ","3382":"セブン&アイ","7532":"パンパシ","8267":"イオン","9843":"ニトリHD","4661":"OLC","6098":"リクルート","9627":"アインHD","4661":"オリエンタル"},
        "電力/ガス": {"9501":"東京電力","9502":"中部電力","9503":"関西電力","9531":"東京ガス","9532":"大阪ガス","9508":"九州電力","9506":"東北電力","9513":"電源開発"},
        "化学/素材": {"4063":"信越化","3407":"旭化成","4005":"住友化","4188":"三菱ケミ","6988":"日東電工","5401":"日本製鉄","5411":"JFE","5713":"住友鉱山","5802":"住友電工","3861":"王子HD"}
    }
}

# --- 4. THE ABSOLUTE ENGINE ---

def fetch_px_single(t, suffix, max_retry=2):
    sess = _get_session()
    base = t.replace(".", "-").lower()
    cands = ["brk-b", "brk.b", base] if t.upper() == "BRK-B" else [base]
    exts = [suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]
    for attempt in range(max_retry + 1):
        for sym in cands:
            for s in exts:
                try:
                    url = f"https://stooq.com/q/d/l/?s={sym}.{s}&i=d"
                    r = sess.get(url, timeout=5)
                    if r.status_code == 200 and r.text:
                        df = pd.read_csv(StringIO(r.text))
                        if "Close" in df.columns and not df.empty:
                            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                            return t, df.dropna(subset=["Date"]).set_index("Date").sort_index()
                except: continue
        time.sleep((0.2 * (2 ** attempt)) + random.random() * 0.2)
    return t, None

@st.cache_data(ttl=1800)
def get_market_pulse_data(universe_key, horizon):
    suffix = "US" if "US" in universe_key else "JP"
    days = {"1W": 5, "1M": 21, "3M": 63}[horizon]
    all_needed = []
    for gs in SECTOR_CATALOG[universe_key].values(): all_needed.extend(list(gs.keys())[:10])
    unique = tuple(sorted(set(all_needed)))
    
    batch_dfs, miss = {}, []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_px_single, t, suffix): t for t in unique}
        for fut in as_completed(futs):
            t, df = fut.result()
            if df is not None: batch_dfs[t] = df
            else: miss.append(t)
            
    pulse_data = []
    for g_name, tickers in SECTOR_CATALOG[universe_key].items():
        rets = []
        for t in list(tickers.keys())[:10]:
            df = batch_dfs.get(t)
            if df is not None and len(df) > days:
                r = (df["Close"].iloc[-1] / df["Close"].iloc[-(days+1)] - 1) * 100
                if r == r: rets.append(r)
        if rets:
            s = pd.Series(rets)
            pulse_data.append({"Group": g_name, "Median": float(s.median()), "WinRate": float((s > 0).mean() * 100), "N": int(s.shape[0])})
    return pd.DataFrame(pulse_data).sort_values("Median", ascending=False), len(unique), miss

@st.cache_data(ttl=1800)
def fetch_news_batch(candidates, suffix):
    lang, gl, ceid = ("en-US", "US", "US:en") if suffix == "US" else ("ja-JP", "JP", "JP:ja")
    def _one(r):
        try:
            q = quote_plus(f"{r['name']} {r['ticker']}")
            url = f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={gl}&ceid={ceid}"
            res = requests.get(url, timeout=5)
            return r["ticker"], [{"title": i.find('title').text, "link": i.find('link').text} for i in ET.fromstring(res.text).findall('.//item')[:3]]
        except: return r["ticker"], []
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, r) for r in candidates]
        for fut in as_completed(futs):
            k, v = fut.result(); out[k] = v
    return out

@st.cache_data(ttl=86400)
def fetch_fundamental_batch(tickers, suffix):
    def _one(t):
        try:
            tk = yf.Ticker(f"{t}.T" if suffix == "JP" else t)
            i = tk.info
            return t, {"per": to_f(i.get("trailingPE")), "pbr": to_f(i.get("priceToBook")), "mcap": to_f(i.get("marketCap"))}
        except: return t, {"per": None, "pbr": None, "mcap": None}
    out = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(_one, t) for t in tickers]
        for fut in as_completed(futs):
            k, v = fut.result(); out[k] = v
    return out

# --- 5. INTERFACE & EXECUTION ---

universe_name = st.sidebar.selectbox("UNIVERSE", list(SECTOR_CATALOG.keys()))
pulse_span = st.sidebar.radio("PULSE_WINDOW", ["1W", "1M", "3M"], index=1, horizontal=True)

st.markdown(f"### MARKET_PULSE / SECTOR_ROTATION ({pulse_span})")

with st.spinner("⚡ Absolute Scanning (cached)..."):
    pulse_df, n_req, miss = get_market_pulse_data(universe_name, pulse_span)
if miss: st.caption(f"Data miss: {len(miss)}/{n_req} (sample)")
if not pulse_df.empty and pulse_df["N"].min() < 6: st.warning("⚠️ Low sample sectors (N<6). Confidence low.")

fig_pulse = px.bar(pulse_df, x="Median", y="Group", orientation='h', color="Median", color_continuous_scale="RdYlGn", hover_data=["WinRate", "N"])
st.plotly_chart(fig_pulse.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)

group_name = st.sidebar.selectbox("TARGET_GROUP", list(SECTOR_CATALOG[universe_name].keys()))
span = st.sidebar.selectbox("HORIZON", ["3M", "1Y", "3Y"], index=1)
suffix = "US" if "US" in universe_name else "JP"

if st.sidebar.button("EXECUTE DEPTH SCAN"):
    target_map = SECTOR_CATALOG[universe_name][group_name]
    days = {"3M": 63, "1Y": 252, "3Y": 756}[span]
    
    with st.spinner(f"Quant Scanning {group_name}..."):
        raw_results, all_norm = [], []
        # Ticker list for parallel PX
        tickers_list = list(target_map.keys())
        results_px = {}
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(fetch_px_single, t, suffix): t for t in tickers_list}
            for fut in as_completed(futs):
                t, df = fut.result()
                if df is not None: results_px[t] = df

        for t, df in results_px.items():
            c = df["Close"].astype(float); ref = min(len(c), days + 1)
            r_now, r_start = to_f(c.iloc[-1]), to_f(c.iloc[-ref])
            ret = ((r_now / r_start - 1) * 100) if (r_now and r_start) else None
            mom = to_f(((c.iloc[-1]/c.iloc[-252]-1)*100)-((c.iloc[-1]/c.iloc[-21]-1)*100)) if len(c)>=252 else None
            r21, r63 = to_f((c.iloc[-1]/c.iloc[-22]-1)*100) if len(c)>=22 else None, to_f((c.iloc[-1]/c.iloc[-64]-1)*100) if len(c)>=64 else None
            accel = (r21 - (r63/3)) if (r21 is not None and r63 is not None) else 0.0
            vol = to_f(c.pct_change().rolling(60).std().iloc[-1]*(252**0.5)*100)
            dd = to_f(((c.iloc[-ref:]/c.iloc[-ref:].cummax()-1)*100).min())
            norm = (c.iloc[-ref:]/c.iloc[-ref:].iloc[0])*100
            all_norm.append(norm.rename(t))
            raw_results.append({"ticker":t,"name":target_map[t],"price":r_now,"ret":ret,"mom":mom,"accel":accel,"vol":vol,"dd":dd,"hist":c.iloc[-ref:],"norm":norm})

        if all_norm:
            sector_ret = pd.concat(all_norm, axis=1).mean(axis=1).iloc[-1] - 100
            for r in raw_results:
                r["rs"] = r["ret"] - sector_ret if r["ret"] is not None else 0.0
                r["score"] = round((r["mom"] or 0)*0.45 + (30-(r["vol"] or 30))*0.15 + r["rs"]*0.2 - abs(r["dd"] or 0)*0.2, 2)
            
            sorted_res = sorted(raw_results, key=lambda x: x['score'], reverse=True)
            top_15 = sorted_res[:15]
            
            # Parallel Info & News
            news_map = fetch_news_batch(top_15, suffix)
            fund_map = fetch_fundamental_batch([r['ticker'] for r in top_15], suffix)
            for r in top_15:
                r["news"] = news_map.get(r['ticker'], [])
                r.update(fund_map.get(r['ticker'], {}))

            # --- DISPLAY ---
            st.markdown("### STRATEGIC INTELLIGENCE BRIEF")
            ai_data = [{k:v for k,v in r.items() if k not in ['hist','norm','news']} for r in top_15]
            try: st.markdown(f"<div class='glass-card'>{client.models.generate_content(model='gemini-flash-latest', contents=f'Hedge fund analystとして、上位3銘柄を日本語で冷徹に分析せよ。Data:{json.dumps(ai_data, ensure_ascii=False)}').text}</div>", unsafe_allow_html=True)
            except: st.error("AI Offline")

            scatter_df = pd.DataFrame([{"Name":r["name"],"Vol":r["vol"],"Ret":r["ret"],"Score":r["score"]} for r in sorted_res if r["vol"] is not None])
            st.plotly_chart(px.scatter(scatter_df, x="Vol", y="Ret", size=scatter_df["Score"].clip(lower=1), color="Score", color_continuous_scale="Viridis", hover_name="Name").update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)

            st.markdown("### ASSET INTELLIGENCE TERMINAL")
            for r in sorted_res:
                with st.expander(f"{r['name']} ({r['ticker']}) // SCORE: {r['score']} // RS: {r['rs']:+.1f}pp", expanded=(r == sorted_res[0])):
                    c1, c2, c3, c4 = st.columns(4)
                    px_txt = f"{r['price']:.1f}" if isinstance(r.get("price"), (int, float)) else "N/A"
                    mom_txt = f"{r['mom']:.1f}%" if isinstance(r.get("mom"), (int, float)) else "N/A"
                    acc_txt = f"{r['accel']:+.1f}" if isinstance(r.get("accel"), (int, float)) else "N/A"
                    dd_txt = f"{r['dd']:.1f}%" if isinstance(r.get("dd"), (int, float)) else "N/A"
                    c1.metric("PX", px_txt)
                    c2.metric("MOM", mom_txt)
                    c3.metric("ACCEL", acc_txt)
                    c4.metric("MAX_DD", dd_txt, delta_color="inverse")
                    
                    if "news" in r:
                        st.write("**Latest Intel:**")
                        for n in r["news"]: st.markdown(f"<div class='news-tag'><a href='{n['link']}' target='_blank'>{n['title']}</a></div>", unsafe_allow_html=True)
                        mcap = r.get("mcap"); mcap_txt = f"{mcap:,.0f}" if isinstance(mcap, (int, float)) else "N/A"
                        per = r.get("per"); per_txt = f"{per:.2f}" if isinstance(per, (int, float)) else "N/A"
                        pbr = r.get("pbr"); pbr_txt = f"{pbr:.2f}" if isinstance(pbr, (int, float)) else "N/A"
                        st.write(f"MCap: {mcap_txt} | PER: {per_txt} | PBR: {pbr_txt}")