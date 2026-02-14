import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from google import genai
import json
from io import StringIO
from datetime import datetime, timezone
import time
import requests
import xml.etree.ElementTree as ET

# --- 1. 設定 & 200銘柄カタログ ---
st.set_page_config(page_title="ConsensusAlpha v5.2.1", layout="wide")
st.title("🧠 ConsensusAlpha v5.2.1: 200銘柄・精密分析エンジン")

SECTOR_CATALOG = {
    "米国株 (US)": {
        "マグニフィセント7": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla"},
        "半導体・AI": {"AVGO": "Broadcom", "AMD": "AMD", "ASML": "ASML", "TSM": "TSMC", "INTC": "Intel", "QCOM": "Qualcomm", "MU": "Micron", "ARM": "Arm", "LRCX": "Lam Research", "AMAT": "Applied Materials"},
        "金融・決済": {"JPM": "JP Morgan", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "AXP": "American Express"},
        "ヘルスケア": {"LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "J&J", "NVO": "Novo Nordisk", "ABBV": "AbbVie", "MRK": "Merck", "PFE": "Pfizer"},
        "消費財・小売": {"WMT": "Walmart", "PG": "P&G", "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco", "NKE": "Nike", "MCD": "McDonald's"}
    },
    "日本株 (JP)": {
        "半導体・ハイテク": {"8035": "東京エレクトロン", "6857": "アドバンテスト", "6758": "ソニー", "6723": "ルネサス", "6146": "ディスコ", "6920": "レーザーテック", "6501": "日立", "6702": "富士通"},
        "金融・メガバンク": {"8306": "三菱UFJ", "8316": "三井住友", "8411": "みずほ", "8766": "東京海上", "8591": "オリックス", "8308": "りそな", "8604": "野村HD"},
        "自動車・輸送": {"7203": "トヨタ", "7267": "ホンダ", "6902": "デンソー", "7201": "日産", "7261": "マツダ", "7270": "SUBARU", "7011": "三菱重工", "7013": "IHI"},
        "総合商社": {"8058": "三菱商事", "8001": "伊藤忠", "8031": "三井物産", "8053": "住友商事", "8015": "豊田通商", "2768": "双日", "8002": "丸紅"},
        "通信・小売・医薬": {"9432": "NTT", "9433": "KDDI", "9984": "ソフトバンクG", "9983": "ファーストリテイリング", "7114": "セブン＆アイ", "4502": "武田薬品", "2802": "味の素"}
    }
}

# --- 2. セキュリティチェック ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("🔑 Secretsに GEMINI_API_KEY を設定してください。")
    st.stop()
client = genai.Client(api_key=api_key)

# --- 3. キャッシュ機能付き取得エンジン ---

@st.cache_data(ttl=3600)
def fetch_price_stooq(ticker, market_suffix):
    suffixes = [market_suffix.lower(), "jp", "jpn"] if market_suffix == "JP" else [market_suffix.lower()]
    for s in suffixes:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.{s}&i=d"
        try:
            r = requests.get(url, timeout=10)
            df = pd.read_csv(StringIO(r.content.decode("utf-8")))
            if "Close" in df.columns and not df.empty:
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date").sort_index(), s
        except: continue
    return None, None

@st.cache_data(ttl=86400)
def fetch_fundamentals_yf(ticker, market_suffix):
    yf_ticker = f"{ticker}.T" if market_suffix == "JP" else ticker
    try:
        tk = yf.Ticker(yf_ticker)
        info = tk.info
        return {"market_cap": info.get("marketCap"), "per": info.get("trailingPE"), "pbr": info.get("priceToBook")}
    except: return {"market_cap": None, "per": None, "pbr": None}

@st.cache_data(ttl=900)
def fetch_news(ticker, name, market_suffix):
    lang, gl, ceid = ("en-US", "US", "US:en") if market_suffix == "US" else ("ja-JP", "JP", "JP:ja")
    query = f"{name} {ticker}" if name != ticker else f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={query}&hl={lang}&gl={gl}&ceid={ceid}"
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.text)
        return list(set([item.find('title').text for item in root.findall('.//item')[:5]]))
    except: return []

# --- 4. スコアリング・エンジン ---

def calculate_logic_score(m):
    score = 0
    # 1. モメンタム
    if m.get("mom_12_1") is not None: score += m["mom_12_1"] * 0.4
    else: score -= 5
    # 2. 低ボラ加点
    if m.get("vol_60d") is not None: score += (30 - m["vol_60d"]) * 0.2
    # 3. 割安性 (上限15点)
    if m.get("per") is not None and m["per"] > 0: score += min((15 / m["per"]) * 10, 15)
    else: score -= 2
    # 4. リスク減点
    score += abs(m["max_dd_period"]) * -0.3
    return round(score, 2)

# --- 5. メイン UI ---

st.sidebar.header("📁 セクター戦略")
market_choice = st.sidebar.selectbox("市場", list(SECTOR_CATALOG.keys()))
sector_choice = st.sidebar.selectbox("セクター", list(SECTOR_CATALOG[market_choice].keys()))
period_choice = st.sidebar.selectbox("分析期間", ["3ヶ月", "1年", "3年"], index=1)

DD_THRESHOLD_MAP = {"3ヶ月": -20.0, "1年": -35.0, "3年": -50.0}
RISK_DD_REJECT = DD_THRESHOLD_MAP[period_choice]

TICKER_MAP = SECTOR_CATALOG[market_choice][sector_choice]
tickers_input = st.sidebar.text_area("銘柄リスト (カンマ区切り)", value=",".join(TICKER_MAP.keys()))
SELECTED_TICKERS = [t.strip() for t in tickers_input.split(",") if t.strip()]

if st.sidebar.button(f"🚀 {sector_choice}一括精査開始"):
    if not SELECTED_TICKERS:
        st.error("銘柄を入力してください。")
        st.stop()

    results = []
    progress_bar = st.progress(0)
    
    for i, t in enumerate(SELECTED_TICKERS):
        name = TICKER_MAP.get(t, t)
        market_suffix = "US" if "US" in market_choice else "JP"
        
        df, s_used = fetch_price_stooq(t, market_suffix)
        if df is not None:
            c = df["Close"].astype(float)
            days = {"3ヶ月": 63, "1年": 252, "3年": 756}[period_choice]
            ref_idx = min(len(c), days + 1)
            
            mom_12_1 = None
            if len(c) >= 252:
                r252 = (c.iloc[-1]/c.iloc[-252]-1)*100
                r21 = (c.iloc[-1]/c.iloc[-21]-1)*100
                mom_12_1 = r252 - r21
            
            vol_60 = c.pct_change().rolling(60).std().iloc[-1] * (252**0.5) * 100 if len(c) > 60 else None
            sub = c.iloc[-ref_idx:]
            max_dd = ((sub / sub.cummax() - 1) * 100).min()
            
            if max_dd >= RISK_DD_REJECT:
                f = fetch_fundamentals_yf(t, market_suffix)
                m = {
                    "ticker": t, "name": name, "stooq": f"{t}.{s_used}",
                    "price": round(c.iloc[-1], 2), "vol_60d": vol_60,
                    "mom_12_1": round(mom_12_1, 2) if mom_12_1 is not None else None,
                    "max_dd_period": round(max_dd, 2), **f, "history": sub
                }
                m["score"] = calculate_logic_score(m)
                m["news"] = fetch_news(t, name, market_suffix)
                results.append(m)
        
        progress_bar.progress((i + 1) / len(SELECTED_TICKERS))

    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # チャート
        st.header(f"📈 相対比較 ({period_choice}：開始時=100)")
        fig = go.Figure()
        for r in results[:10]:
            norm = (r["history"] / r["history"].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm, name=r["name"]))
        st.plotly_chart(fig, use_container_width=True)

        # AI
        ai_payload = [{k: v for k, v in r.items() if k != 'history'} for r in results[:10]]
        prompt = f"あなたは投資議長です。以下のデータを元に分析レポートを作成して。ニュース無しは「不能」とすること。データ:{json.dumps(ai_payload, ensure_ascii=False)}"
        report = client.models.generate_content(model='gemini-flash-latest', contents=prompt)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🏆 委員会レポート")
            st.markdown(report.text)
        with col2:
            st.header("📋 スコア明細")
            st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k not in ['history', 'news']} for r in results]))