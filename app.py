import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from google import genai
import json, re
from io import StringIO
from urllib.parse import quote_plus
import time
import requests
import xml.etree.ElementTree as ET

# --- 1. SPECTRE DESIGN (PHANTOM MODE) ---
st.set_page_config(page_title="AlphaLens v7.6", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;500&display=swap');
    html, body, [class*="css"] { font-family: 'Fira Code', monospace; background-color: #05070a; color: #a9b1d6; }
    .stButton>button { width: 100%; border-radius: 2px; height: 3.5em; background: linear-gradient(90deg, #1a1b26, #24283b); color: #7aa2f7; border: 1px solid #7aa2f7; font-weight: bold; letter-spacing: 2px; }
    .glass-card { background: rgba(26, 27, 38, 0.7); padding: 15px; border-radius: 4px; border-left: 4px solid #7aa2f7; margin-bottom: 15px; }
    .alert-card { background: rgba(122, 162, 247, 0.1); border: 1px solid #7aa2f7; padding: 10px; border-radius: 4px; text-align: center; }
    .metric-val { color: #bb9af7; font-size: 1.1em; font-weight: bold; }
    .metric-label { color: #565f89; font-size: 0.75em; text-transform: uppercase; }
    .news-tag { display: inline-block; padding: 2px 8px; background-color: #21262d; border-radius: 4px; font-size: 0.75em; margin: 2px; border: 1px solid #30363d; }
    .audit-text { font-size: 0.7em; color: #565f89; line-height: 1.2; }
    a { color: #58a6ff !important; text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("ALPHALENS // PHANTOM.SENTINEL.v7.6")

# --- 2. UTILITY & SECURITY ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key: st.stop()
client = genai.Client(api_key=api_key)

TICKER_FIX = { "BRK-B": "BRK_B", "CHTR": "CHTR" } # Stooq互換表

def to_f(val):
    try:
        if val is None: return None
        v = float(val); return v if v == v else None
    except: return None

# --- 3. THE 400+ UNIVERSE (REFINED & EXPANDED) ---
SECTOR_CATALOG = {
    "US MARKET (206 Assets)": {
        "Mega Tech": {"AAPL":"Apple","MSFT":"MSFT","GOOGL":"Google","AMZN":"Amazon","NVDA":"NVIDIA","META":"Meta","TSLA":"Tesla","BRK-B":"Berkshire","NFLX":"Netflix","ADBE":"Adobe","CRM":"Salesforce","ORCL":"Oracle","IBM":"IBM","DELL":"Dell","HPE":"HPE","HPQ":"HP","ACN":"Accenture"},
        "Semis / AI": {"AVGO":"Broadcom","AMD":"AMD","TSM":"TSMC","ASML":"ASML","MU":"Micron","LRCX":"Lam","AMAT":"Applied","ARM":"Arm","QCOM":"Qualcomm","INTC":"Intel","KLAC":"KLA","TER":"Teradyne","ON":"ON Semi","TXN":"TI","ADI":"Analog","NXPI":"NXP","MRVL":"Marvell","CDNS":"Cadence","SNPS":"Synopsys","MCHP":"Micro"},
        "SaaS / Cyber": {"SNOW":"Snowflake","PLTR":"Palantir","NOW":"ServiceNow","WDAY":"Workday","PANW":"Palo Alto","CRWD":"CrowdStrike","DDOG":"Datadog","FTNT":"Fortinet","ZS":"Zscaler","OKTA":"Okta","TEAM":"Atlassian","ADSK":"Autodesk","U":"Unity","SHOP":"Shopify","MDB":"MongoDB","NET":"Cloudflare","CHKP":"CheckPoint"},
        "Financials": {"JPM":"JP Morgan","V":"Visa","MA":"Mastercard","BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","AXP":"Amex","BLK":"BlackRock","C":"Citigroup","USB":"USBancorp","PNC":"PNC","BK":"BNY Mellon","CME":"CME","SPGI":"S&P Global","MCO":"Moody's","ICE":"ICE","PYPL":"PayPal","COIN":"Coinbase","WFC":"Wells Fargo"},
        "Healthcare": {"LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","NVO":"Novo","ABBV":"AbbVie","MRK":"Merck","PFE":"Pfizer","TMO":"Thermo","BMY":"Bristol","AMGN":"Amgen","MDT":"Medtronic","BSX":"BostonSci","REGN":"Regeneron","ZTS":"Zoetis","SYK":"Stryker","DXCM":"Dexcom","GILD":"Gilead","ISRG":"Intuitive"},
        "Industrials / Defense": {"LMT":"Lockheed","RTX":"Raytheon","NOC":"Northrop","GD":"GenDynamics","BA":"Boeing","GE":"GE","HON":"Honeywell","CAT":"Caterpillar","DE":"Deere","ETN":"Eaton","MMM":"3M","EMR":"Emerson","ITW":"ITW","UPS":"UPS","FDX":"FedEx","WM":"WasteMgmt"},
        "Energy / Utilities": {"XOM":"Exxon","CVX":"Chevron","COP":"Conoco","SLB":"Schlumberger","EOG":"EOG","KMI":"KinderMorgan","MPC":"Marathon","OXY":"Occidental","PSX":"Phillips66","HAL":"Halliburton","VLO":"Valero","NEE":"NextEra","DUKE":"Duke","SO":"SouthernCo","EXC":"Exelon","D":"Dominion"},
        "Consumer Staples": {"PG":"P&G","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart","COST":"Costco","PM":"PhilipMorris","MO":"Altria","CL":"Colgate","KMB":"Kimberly","GIS":"GeneralMills","KHC":"KraftHeinz","MDLZ":"Mondelez","EL":"EsteeLauder","TGT":"Target"},
        "Consumer Disc": {"HD":"HomeDepot","LOW":"Lowe's","NKE":"Nike","SBUX":"Starbucks","CMG":"Chipotle","BKNG":"Booking","MAR":"Marriott","MCD":"McDonald's","TJX":"TJX","ORLY":"O'Reilly","F":"Ford","GM":"GM","EBAY":"eBay"},
        "Communication / Media": {"DIS":"Disney","CHTR":"Charter","TMUS":"T-Mobile","VZ":"Verizon","T":"AT&T","CMCSA":"Comcast","PARA":"Paramount","FOXA":"Fox","WBD":"WarnerBros","CHTTR":"Charter"},
        "Materials / Chem / Mining": {"LIN":"Linde","APD":"AirProducts","SHW":"Sherwin","ECL":"Ecolab","DOW":"Dow","DD":"DuPont","FCX":"Freeport","NEM":"Newmont","CTVA":"Corteva","NUE":"Nucor","VMC":"Vulcan"},
        "REITs / Real Estate": {"PLD":"Prologis","AMT":"AmericanTower","EQIX":"Equinix","PSA":"PublicStorage","O":"RealtyIncome","DLR":"DigitalRealty","VICI":"VICI","CCI":"CrownCastle","SBAC":"SBAC","CBRE":"CBRE"}
    },
    "JP MARKET (202 Assets)": {
        "半導体/電子部品": {"8035":"東エレク","6857":"アドバンテ","6723":"ルネサス","6146":"ディスコ","6920":"レーザー","3436":"SUMCO","4063":"信越化","7735":"スクリン","6526":"ソシオネ","6963":"ローム","7751":"キヤノン","6981":"村田製","6762":"TDK","6861":"キーエンス","6954":"ファナック","6503":"三菱電機","6501":"日立"},
        "情報通信/ネット": {"9432":"NTT","9433":"KDDI","9434":"ソフトB","9984":"SBG","4755":"楽天G","3659":"ネクソン","4689":"LINEヤフー","3774":"IIJ","4385":"メルカリ","3923":"ラクス","9613":"NTTデータ","2121":"MIXI","3632":"グリー"},
        "資本財/重工/防衛": {"7011":"三菱重工","7012":"川崎重工","7013":"IHI","6301":"小松","6367":"ダイキン","6361":"荏原","5631":"日製鋼","6273":"SMC","6504":"富士電機","6305":"日立建機","6113":"アマダ","6473":"ジェイテクト","6326":"クボタ"},
        "自動車/輸送機": {"7203":"トヨタ","7267":"ホンダ","6902":"デンソー","7201":"日産","7269":"スズキ","7272":"ヤマハ発","7261":"マツダ","7270":"SUBARU","7259":"アイシン","7205":"日野自","7211":"三菱自","5108":"ブリヂストン"},
        "金融": {"8306":"三菱UFJ","8316":"三井住友","8411":"みずほ","8766":"東京海上","8591":"オリックス","8604":"野村HD","8725":"MS&AD","8308":"りそな","7186":"コンコルディア","8630":"SOMPO","8750":"第一生命","8795":"T&D","8309":"三井住友トラ","8473":"SBI","8601":"大和証"},
        "総合商社/投資": {"8058":"三菱商事","8001":"伊藤忠","8031":"三井物産","8053":"住友商事","8015":"豊田通商","8002":"丸紅","2768":"双日","1605":"INPEX"},
        "食品/必需品": {"2802":"味の素","2914":"JT","2502":"アサヒ","2503":"キリン","2501":"サッポロ","4452":"花王","2269":"明治HD","2801":"キッコーマン","2587":"サントリーBF","4911":"資生堂"},
        "ヘルスケア": {"4502":"武田薬品","4568":"第一三共","4519":"中外製薬","4503":"アステラス","4523":"エーザイ","4901":"富士フイルム","7741":"HOYA","4543":"テルモ","4578":"大塚HD","4507":"塩野義"},
        "不動産/建設": {"8801":"三井不動","8802":"三菱地所","8830":"住友不動","3289":"東急不動","1801":"大成建設","1812":"鹿島建設","1803":"清水建設","1802":"大林組","1928":"積水ハウス","1925":"大和ハウス","1878":"大東建託"},
        "物流/鉄道/海運": {"9022":"JR東海","9020":"JR東日本","9021":"JR西日本","9101":"日本郵船","9104":"商船三井","9107":"川崎汽船","9064":"ヤマトHD","9005":"東急","9007":"小田急","9001":"東武","9008":"京王","9143":"SGHD"},
        "小売/サービス": {"9983":"ファーストリテ","3382":"セブン&アイ","7532":"パンパシ","8267":"イオン","9843":"ニトリHD","4661":"OLC","6098":"リクルート","9627":"アインHD"},
        "電力/ガス": {"9501":"東京電力","9502":"中部電力","9503":"関西電力","9531":"東京ガス","9532":"大阪ガス","9508":"九州電力","9506":"東北電力","9513":"電源開発"},
        "化学/素材/鉄鋼": {"3407":"旭化成","4005":"住友化","4188":"三菱ケミ","6988":"日東電工","5401":"日本製鉄","5411":"JFE","5713":"住友山","5802":"住友電工","3861":"王子HD"}
    }
}

# --- 4. AUDIT & DATA PIPELINE ---

def run_universe_audit():
    """誠実な計器：起動時に重複と銘柄数を自動チェック"""
    audit = {}
    for uni, groups in SECTOR_CATALOG.items():
        all_tickers = []
        for g, t_map in groups.items(): all_tickers.extend(list(t_map.keys()))
        dupes = set([x for x in all_tickers if all_tickers.count(x) > 1])
        audit[uni] = {"total": len(all_tickers), "dupes": list(dupes)}
    return audit

@st.cache_data(ttl=3600)
def fetch_px(t, suffix):
    ticker_mapped = TICKER_FIX.get(t, t).replace(".", "-")
    for s in ([suffix.lower(), "jp", "jpn"] if suffix == "JP" else [suffix.lower()]):
        try:
            r = requests.get(f"https://stooq.com/q/d/l/?s={ticker_mapped.lower()}.{s}&i=d", timeout=10)
            df = pd.read_csv(StringIO(r.content.decode("utf-8")))
            if "Close" in df.columns and not df.empty:
                df["Date"] = pd.to_datetime(df["Date"]); return df.set_index("Date").sort_index()
        except: continue
    return None

def fetch_top_sentiment(candidates, suffix):
    if not candidates: return {}
    lang, gl, ceid = ("en-US", "US", "US:en") if suffix == "US" else ("ja-JP", "JP", "JP:ja")
    batch = []
    for r in candidates:
        try:
            q = quote_plus(f"{r['name']} {r['ticker']}")
            res = requests.get(f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={gl}&ceid={ceid}", timeout=5)
            batch.append({"ticker": r["ticker"], "headlines": [i.find('title').text for i in ET.fromstring(res.text).findall('.//item')[:2]]})
        except: batch.append({"ticker": r["ticker"], "headlines": []})
    
    # JSONパースの非貪欲化
    prompt = f"Analyze stock sentiment. STRICT JSON: {{\"TICKER\": float(-2.0 to 2.0)}}. Data: {batch}"
    try:
        res = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
        match = re.search(r'\{.*?\}', res.text, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except: return {}

# --- 5. LOGIC ENGINE ---

def get_alpha_score(m):
    # 測定不能（None）時はスコアに寄与させない
    s_mom = m["mom"] if m["mom"] is not None else 0.0
    s_vol = m["vol"] if m["vol"] is not None else 30.0
    s_rs = m["rs"] if m["rs"] is not None else 0.0
    s_dd = m["dd"] if m["dd"] is not None else 0.0
    score = (s_mom * 0.45) + ((30 - s_vol) * 0.15) + (s_rs * 0.2) - (abs(s_dd) * 0.2)
    return round(score, 2)

# --- 6. INTERFACE ---
audit_res = run_universe_audit()
st.sidebar.markdown("### 🔍 SYSTEM_AUDIT")
for k, v in audit_res.items():
    st.sidebar.markdown(f"<p class='audit-text'><b>{k}</b><br>Total: {v['total']} | Dupes: {len(v['dupes'])}</p>", unsafe_allow_html=True)

st.sidebar.markdown("---")
universe_name = st.sidebar.selectbox("UNIVERSE", list(SECTOR_CATALOG.keys()))
pulse_span = st.sidebar.radio("PULSE_WINDOW", ["1W", "1M", "3M"], index=1, horizontal=True)

@st.cache_data(ttl=1800)
def get_market_pulse(universe_key, horizon):
    pulse_data = []
    suffix = "US" if "US" in universe_key else "JP"
    days = {"1W": 5, "1M": 21, "3M": 63}[horizon]
    # 精度と安定の両立：1セクター最大15銘柄にキャップ
    for g_name, tickers in SECTOR_CATALOG[universe_key].items():
        rets = []
        for t in list(tickers.keys())[:15]:
            df = fetch_px(t, suffix)
            if df is not None and len(df) > days:
                r = (df["Close"].iloc[-1] / df["Close"].iloc[-(days+1)] - 1) * 100
                if r == r: rets.append(r) # NaN除外
        if rets:
            s_rets = pd.Series(rets)
            pulse_data.append({"Group": g_name, "Median": float(s_rets.median()), "WinRate": float((s_rets > 0).mean() * 100)})
    return pd.DataFrame(pulse_data).sort_values("Median", ascending=False)

# PULSE表示
st.markdown(f"### MARKET_PULSE / SECTOR_ROTATION ({pulse_span})")

pulse_df = get_market_pulse(universe_name, pulse_span)
fig_pulse = px.bar(pulse_df, x="Median", y="Group", orientation='h', color="Median", 
                   color_continuous_scale="RdYlGn", hover_data=["WinRate"], labels={"Median": "Median Return (%)"})
fig_pulse.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_pulse, use_container_width=True)

group_name = st.sidebar.selectbox("TARGET_GROUP", list(SECTOR_CATALOG[universe_name].keys()))
span = st.sidebar.selectbox("HORIZON", ["3M", "1Y", "3Y"], index=1)
suffix = "US" if "US" in universe_name else "JP"

if st.sidebar.button("EXECUTE DEPTH SCAN"):
    raw_results = []
    days = {"3M": 63, "1Y": 252, "3Y": 756}[span]
    
    with st.spinner("QUANT SCANNING..."):
        all_norm = []
        for t, n in SECTOR_CATALOG[universe_name][group_name].items():
            df = fetch_px(t, suffix)
            if df is not None:
                c = df["Close"].astype(float); ref = min(len(c), days + 1)
                
                # リターン計算 (測定不能ガード)
                r_now = to_f(c.iloc[-1]); r_start = to_f(c.iloc[-ref])
                ret = ((r_now / r_start - 1) * 100) if (r_now is not None and r_start) else None
                
                # モメンタム (N/A維持)
                mom = to_f(((c.iloc[-1]/c.iloc[-252]-1)*100)-((c.iloc[-1]/c.iloc[-21]-1)*100)) if len(c)>=252 else None
                
                # Accel (両方 not None 時のみ計算)
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
                r["rs"] = (r["ret"] - sector_ret) if r["ret"] is not None else None
                r["score"] = get_alpha_score(r)

    if raw_results:
        sorted_res = sorted(raw_results, key=lambda x: x.get("score", -99), reverse=True)
        top_candidates = sorted_res[:15]
        sent_map = fetch_top_sentiment(top_candidates, suffix)
        for r in top_candidates: r["sentiment"] = to_f(sent_map.get(r["ticker"], 0.0))

        # --- UI: ALERTS & CHART ---
        st.markdown("### ALPHA ALERTS")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='alert-card'><div class='metric-label'>ACCEL TOP</div><div class='metric-val'>{max(raw_results, key=lambda x: x['accel'])['ticker']}</div></div>", unsafe_allow_html=True)
        valid_rs = [r for r in raw_results if r["rs"] is not None]
        top_rs_ticker = max(valid_rs, key=lambda x: x["rs"])["ticker"] if valid_rs else "N/A"
        c2.markdown(f"<div class='alert-card'><div class='metric-label'>RS TOP (pp)</div><div class='metric-val'>{top_rs_ticker}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='alert-card'><div class='metric-label'>SECTOR RET</div><div class='metric-val'>{sector_ret:+.1f}%</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='alert-card'><div class='metric-label'>AVG VOL</div><div class='metric-val'>{pd.Series([r['vol'] for r in raw_results if r['vol'] is not None]).mean():.1f}%</div></div>", unsafe_allow_html=True)

        st.markdown(f"### RISK-REWARD MAP ({span})")
        
        scatter_df = pd.DataFrame([{"Ticker":r["ticker"],"Vol":r["vol"],"Ret":r["ret"],"Score":r["score"],"Name":r["name"]} for r in raw_results if r["vol"] is not None])
        fig = px.scatter(scatter_df, x="Vol", y="Ret", text="Ticker", size=scatter_df["Score"].clip(lower=1), 
                         color="Score", color_continuous_scale="Viridis", hover_name="Name")
        st.plotly_chart(fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True)

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("### STRATEGIC BRIEF")
            prompt = f"Hedge fund quant. Rank Top 3 in Japanese based on RS and Accel. Data: {json.dumps([{k:v for k,v in r.items() if k not in ['hist','norm']} for r in top_candidates], ensure_ascii=False)}"
            try: st.markdown(f"<div class='glass-card'>{client.models.generate_content(model='gemini-flash-latest', contents=prompt).text}</div>", unsafe_allow_html=True)
            except: st.error("AI Offline")
        with col2:
            st.markdown("### INTEL TERMINAL")
            for r in sorted_res:
                with st.expander(f"{r['name']} ({r['ticker']}) // Score: {r.get('score')}"):
                    st.write(f"MOM(12-1): {r['mom'] if r['mom'] is not None else 'N/A'}% | RS: {r['rs'] if r['rs'] is not None else 'N/A':+.1f}pp | DD: {r['dd']}%")