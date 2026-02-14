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

# --- 1. 設定 & セクター辞書 ---
st.set_page_config(page_title="ConsensusAlpha v5.2", layout="wide")
st.title("🧠 ConsensusAlpha v5.2: 200銘柄・精密分析エンジン")

# 銘柄リスト（日米合計約200銘柄規模への拡張例）
SECTOR_CATALOG = {
    "米国株 (US)": {
        "マグニフィセント7": {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla"},
        "半導体・AI": {"AVGO": "Broadcom", "AMD": "AMD", "ASML": "ASML", "TSM": "TSMC", "INTC": "Intel", "QCOM": "Qualcomm", "MU": "Micron", "ARM": "Arm", "LRCX": "Lam Research", "AMAT": "Applied Materials"},
        "金融・決済": {"JPM": "JP Morgan", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America", "GS": "Goldman Sachs", "MS": "Morgan Stanley", "AXP": "American Express"},
        "ヘルスケア": {"LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "J&J", "NVO": "Novo Nordisk", "ABBV": "AbbVie", "MRK": "Merck", "PFE": "Pfizer", "TMO": "Thermo Fisher"},
        "消費財・小売": {"WMT": "Walmart", "PG": "P&G", "KO": "Coca-Cola", "PEP": "PepsiCo", "COST": "Costco", "NKE": "Nike", "MCD": "McDonald's", "DIS": "Disney"}
    },
    "日本株 (JP)": {
        "半導体・ハイテク": {"8035": "東京エレクトロン", "6857": "アドバンテスト", "6758": "ソニー", "6723": "ルネサス", "6146": "ディスコ", "6920": "レーザーテック", "6501": "日立", "6702": "富士通", "6645": "オムロン"},
        "金融・メガバンク": {"8306": "三菱UFJ", "8316": "三井住友", "8411": "みずほ", "8766": "東京海上", "8591": "オリックス", "8308": "りそな", "8604": "野村HD", "8725": "MS&AD"},
        "自動車・輸送": {"7203": "トヨタ", "7267": "ホンダ", "6902": "デンソー", "7201": "日産", "7261": "マツダ", "7270": "SUBARU", "7011": "三菱重工", "7012": "川崎重工", "7013": "IHI"},
        "総合商社": {"8058": "三菱商事", "8001": "伊藤忠", "8031": "三井物産", "8053": "住友商事", "8015": "豊田通商", "2768": "双日", "8002": "丸紅"},
        "通信・小売・その他": {"9432": "NTT", "9433": "KDDI", "9984": "ソフトバンクG", "9983": "ファーストリテイリング", "7114": "セブン＆アイ", "4502": "武田薬品", "2802": "味の素", "1925": "大和ハウス"}
    }
}

# --- 2. セキュリティチェック ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("🔑 Secretsに GEMINI_API_KEY が設定されていません。")
    st.stop()
client = genai.Client(api_key=api_key)

# --- 3. ユーティリティ & キャッシュ ---

@st.cache_data(ttl=3600)
def fetch_price_stooq(ticker, market_suffix):
    """サフィックスフォールバックを伴う価格取得"""
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

@st.cache_data(ttl=900) # ニュースも15分間キャッシュ
def fetch_news(ticker, name, market_suffix):
    lang, gl, ceid = ("en-US", "US", "US:en") if market_suffix == "US" else ("ja-JP", "JP", "JP:ja")
    # 名称がない（追加銘柄）場合はティッカーのみ
    query = f"{name} {ticker}" if name != ticker else f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={query}&hl={lang}&gl={gl}&ceid={ceid}"
    try:
        r = requests.get(url, timeout=5)
        root = ET.fromstring(r.text)
        return list(set([item.find('title').text for item in root.findall('.//item')[:5]]))
    except: return []

# --- 4. スコアリング・エンジン ---

def calculate_logic_score(m):
    """Pythonによる決定論的スコアリング (アドバイス反映版)"""
    score = 0
    # 1. モメンタム (40%)
    if m.get("mom_12_1") is not None:
        score += m["mom_12_1"] * 0.4
    else:
        score -= 5 # データ欠損ペナルティ
    
    # 2. 低ボラ加点 (修正：キー名を vol_60d に統一)
    if m.get("vol_60d") is not None:
        score += (30 - m["vol_60d"]) * 0.2
    
    # 3. 割安性 (修正：PER加点に上限設定)
    if m.get("per") is not None and m["per"] > 0:
        value_points = (15 / m["per"]) * 10
        score += min(value_points, 15) # 最大15点に制限して異常値を防ぐ
    else:
        score -= 2
    
    # 4. リスク減点 (DDをそのまま加算)
    score += abs(m["max_dd_period"]) * -0.3
    return round(score, 2)

# --- 5. メイン UI ---

st.sidebar.header("📁 セクター戦略")
market_choice = st.sidebar.selectbox("市場", list(SECTOR_CATALOG.