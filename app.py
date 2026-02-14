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

# --- 1. 設定 & 200銘柄セクターカタログ ---
st.set_page_config(page_title="ConsensusAlpha v5.3.1", layout="wide")
st.title("🧠 ConsensusAlpha v5.3.1: セクター委員会・高精度版")

SECTOR_CATALOG = {
    "米国株 (US)": {
        "マグニフィセント7": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla"},
        "半導体精鋭": {"AVGO": "Broadcom", "AMD": "AMD", "ASML": "ASML", "TSM": "TSMC", "ARM": "Arm", "MU": "Micron", "LRCX": "Lam Research"},
        "金融・決済": {"JPM": "JP Morgan", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America", "GS": "Goldman Sachs", "AXP": "American Express"}
    },
    "日本株 (JP)": {
        "半導体・ハイテク": {"8035": "東京エレクトロン", "6857": "アドバンテスト", "6758": "ソニー", "6723": "ルネサス", "6146": "ディスコ", "6920": "レーザーテック", "6501": "日立"},
        "金融・メガバンク": {"8306": "三菱UFJ", "8316": "三井住友", "8411": "みずほ", "8766": "東京海上", "8591": "オリックス", "8604": "野村HD"},
        "自動車・重工業": {"7203": "トヨタ", "7267": "ホンダ", "6902": "デンソー", "7011": "三菱重工", "7012": "川崎重工", "7013": "IHI"}
    }
}

# --- 2. セキュリティ設定 ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("🔑 Secretsに GEMINI_API_KEY を設定してください。")
    st.stop()
client = genai.Client(api_key=api_key)

# --- 3. 高精度データ取得エンジン ---

@st.cache_data(ttl=3600)
def fetch_data(ticker, market_suffix):
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
def fetch_info(ticker, market_suffix):
    yf_ticker = f"{ticker}.T" if market_suffix == "JP" else ticker
    try:
        tk = yf.Ticker(yf_ticker)
        info = tk.info
        return {"per": info.get("trailingPE"), "pbr": info.get("priceToBook")}
    except: return {"per": None, "pbr": None}

@st.cache_data(ttl=900)
def fetch_news(ticker, name, market_suffix):
    """URLエンコードによる日本語対応ニュース検索"""
    lang, gl, ceid = ("en-US", "US", "US:en") if market_suffix == "US" else ("ja-JP", "JP", "JP:ja")
    # アドバイス反映：会社名とティッカーをエンコード
    query = quote_plus(f"{name} {ticker}")
    url = f"https://news.google.com/rss/search?q={query}&hl={lang}&gl={gl}&ceid={ceid}"
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.text)
        return [item.find('title').text for item in root.findall('.//item')[:3]]
    except: return []

# --- 4. 厳格なスコアリング ---

def calculate_logic_score(m):
    score = 0
    # 1. 12-1ヶ月モメンタム (40%)
    if m.get("mom_12_1") is not None: score += m["mom_12_1"] * 0.4
    else: score -= 5 # 欠損ペナルティ
    
    # 2. 低ボラ加点 (30%基準)
    if m.get("vol_60d") is not None: score += (30 - m["vol_60d"]) * 0.2
    
    # 3. 割安性 (PER上限15点、欠損時ペナルティ)
    if m.get("per") is not None and m["per"] > 0:
        score += min((15 / m["per"]) * 10, 15)
    else:
        score -= 2 # 財務不明ペナルティ
        
    # 4. 期間ドローダウン
    score += abs(m["max_dd_period"]) * -0.3
    return round(score, 2)

# --- 5. メイン UI ---

st.sidebar.header("📁 セクター戦略設定")
market_choice = st.sidebar.selectbox("市場", list(SECTOR_CATALOG.keys()))
sector_choice = st.sidebar.selectbox("セクター", list(SECTOR_CATALOG[market_choice].keys()))
period_choice = st.sidebar.selectbox("分析スパン", ["3ヶ月", "1年", "3年"], index=1)

DD_LIMIT = {"3ヶ月": -20.0, "1年": -35.0, "3年": -50.0}[period_choice]
TICKER_MAP = SECTOR_CATALOG[market_choice][sector_choice]

if st.sidebar.button(f"🚀 {sector_choice}・精密評議を開始"):
    results = []
    market_suffix = "US" if "US" in market_choice else "JP"
    
    with st.status("セクターデータを統合中...", expanded=True) as status:
        for t, name in TICKER_MAP.items():
            st.write(f"📡 {name} ({t}) の全指標を精査中...")
            df, s_used = fetch_data(t, market_suffix)
            if df is not None:
                c = df["Close"].astype(float)
                days = {"3ヶ月": 63, "1年": 252, "3年": 756}[period_choice]
                ref = min(len(c), days + 1)
                
                # 指標計算 (mom_12_1 を明示)
                mom_12_1 = None
                if len(c) >= 252:
                    r252 = (c.iloc[-1]/c.iloc[-252]-1)*100
                    r21 = (c.iloc[-1]/c.iloc[-21]-1)*100
                    mom_12_1 = r252 - r21
                
                vol = c.pct_change().rolling(60).std().iloc[-1] * (252**0.5) * 100 if len(c)>60 else None
                sub = c.iloc[-ref:]
                dd = ((sub / sub.cummax() - 1) * 100).min()
                
                if dd >= DD_LIMIT:
                    info = fetch_info(t, market_suffix)
                    m = {
                        "ticker": t, "name": name, "st_symbol": f"{t}.{s_used}",
                        "price": round(c.iloc[-1], 2), "mom_12_1": round(mom_12_1, 2) if mom_12_1 is not None else None,
                        "vol_60d": vol, "max_dd_period": round(dd, 2), **info, "history": sub
                    }
                    m["score"] = calculate_logic_score(m)
                    m["news"] = fetch_news(t, name, market_suffix)
                    results.append(m)
            time.sleep(0.1)
        status.update(label="データ収集完了", state="complete")

    if results:
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # グラフ表示
        st.header(f"📈 {sector_choice}：相対パフォーマンス比較 ({period_choice})")
        
        fig = go.Figure()
        for r in results:
            norm = (r["history"] / r["history"].iloc[0]) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm, name=r["name"]))
        fig.update_layout(height=450, yaxis_title="指数 (開始日=100)")
        st.plotly_chart(fig, use_container_width=True)

        # AIレポート (指示の強化)
        ai_payload = [{k: v for k, v in r.items() if k != 'history'} for r in results]
        prompt = f"""
        あなたは投資委員会の議長です。{sector_choice}セクターの以下のデータを元に、総合順位を決定しレポートを作成してください。
        
        【評議のルール】
        1. Pythonスコアを基本とするが、ニュースやPER/PBRを元に順位を最大2名まで入れ替えてもよい（その場合は理由を明記）。
        2. 各銘柄の評価では、必ず「mom_12_1」「PER」「max_dd_period」の数値を引用して説明すること。
        3. ニュースがない銘柄は「ニュースによる判断不能」と明記し、憶測で書かないこと。
        
        【セクターデータ】
        {json.dumps(ai_payload, ensure_ascii=False)}
        """
        
        try:
            # モデル名を安定版 gemini-flash-latest に
            response = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.header("🏆 委員会による総合評議レポート")
                st.markdown(response.text)
            with col2:
                st.header("📋 セクター・エビデンス一覧")
                st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k not in ['history', 'news']} for r in results]))
        except Exception as e:
            st.error("AIレポート生成中にエラーが発生しました。右側のデータ一覧を確認してください。")