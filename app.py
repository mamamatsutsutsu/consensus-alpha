import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google import genai
import requests
import json
from io import StringIO
from datetime import datetime, timezone
import time
import xml.etree.ElementTree as ET

# --- 1. 設定 & UI構成 ---
st.set_page_config(page_title="ConsensusAlpha Global v2", layout="wide")
st.title("🧠 ConsensusAlpha: グローバル投資委員会 v2.0")

# --- 2. セキュリティ & API設定 ---
# Secretsから取得、なければサイドバー（ローカル開発用）
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key (Local only)", type="password")

if not api_key:
    st.warning("APIキーが設定されていません。StreamlitのSecretsに設定するか、サイドバーに入力してください。")
    st.stop()

client = genai.Client(api_key=api_key)

# サイドバー設定
st.sidebar.header("🔧 分析設定")
market = st.sidebar.selectbox("分析対象の市場", ["米国株 (US)", "日本株 (JP)"])

if market == "米国株 (US)":
    DEFAULT_TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]
    market_suffix = "US"
    news_params = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
else:
    DEFAULT_TICKERS = ["7203", "6758", "9984", "8035", "6857"]
    market_suffix = "JP"
    news_params = {"hl": "ja-JP", "gl": "JP", "ceid": "JP:ja"}

tickers_input = st.sidebar.text_input("銘柄コード（カンマ区切り）", value=",".join(DEFAULT_TICKERS))
TICKERS = [t.strip() for t in tickers_input.split(",")]
RISK_DD_REJECT = -40.0

# --- 3. 堅牢なデータ取得エンジン ---

def fetch_news_headlines(ticker, params):
    """市場に合わせた言語・地域設定でニュースを取得"""
    query = f"{ticker} stock" if market == "米国株 (US)" else f"{ticker} 株価"
    url = f"https://news.google.com/rss/search?q={query}&hl={params['hl']}&gl={params['gl']}&ceid={params['ceid']}"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        headlines = [item.find('title').text for item in root.findall('.//item')[:5]]
        return headlines if headlines else ["関連ニュースは見つかりませんでした"]
    except Exception:
        return ["ニュースの取得に失敗しました"]

def fetch_stock_data_with_fallback(ticker, suffix):
    """サフィックスのフォールバックを試行する堅牢なデータ取得"""
    # 日本株の場合は .JP と .JPN の両方を試す（Stooqの気まぐれ対策）
    suffixes = [suffix] if suffix == "US" else ["JP", "JPN"]
    
    for s in suffixes:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.{s.lower()}&i=d"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.content.decode("utf-8")))
            if "Close" in df.columns and len(df) > 0:
                df["Date"] = pd.to_datetime(df["Date"])
                return df.set_index("Date").sort_index()
        except:
            continue
    raise ValueError(f"銘柄 {ticker} のデータを取得できませんでした。")

def calc_metrics(df):
    c = df["Close"].astype(float)
    rows = len(c)
    r252 = ((c.iloc[-1] / c.iloc[-252]) - 1) * 100 if rows >= 252 else None
    r21  = ((c.iloc[-1] / c.iloc[-21]) - 1) * 100 if rows >= 21 else None
    
    # 改良された 12-1ヶ月モメンタム
    mom_12_1 = (r252 - r21) if (r252 is not None and r21 is not None) else None
    vol_60d = c.pct_change().rolling(60).std().iloc[-1] * (252 ** 0.5) * 100 if rows >= 62 else None
    ma_200 = c.rolling(200).mean().iloc[-1]
    ma_200_gap = ((c.iloc[-1] / ma_200) - 1) * 100 if (rows >= 200 and ma_200 != 0) else None
    window = min(252, rows)
    sub = c.iloc[-window:]
    max_dd = ((sub / sub.cummax() - 1) * 100).min()
    
    return {
        "price": round(c.iloc[-1], 2),
        "mom_12_1": round(mom_12_1, 2) if mom_12_1 is not None else None,
        "vol_60d": round(vol_60d, 2) if vol_60d is not None else None,
        "max_dd_252d": round(max_dd, 2) if max_dd is not None else None,
        "ma_200_gap": round(ma_200_gap, 2) if ma_200_gap is not None else None,
        "ret_1m": round(r21, 2) if r21 is not None else None,
    }

# --- 4. 委員会合議プロンプト（構造化） ---

def run_structured_committee(analyzed_data):
    data_json = json.dumps(analyzed_data, ensure_ascii=False)
    prompt = f"""
    あなたは投資委員会の議長です。以下のデータを「唯一の事実」として、Top3銘柄を選定してください。

    【制約事項】
    - 提供された「news」以外の外部ニュースを推測で書かないでください。
    - ニュース見出しから業界事情を深読みしすぎず、見出しの事実に限定してください。
    - 各銘柄に [Pos/Neu/Neg] のセンチメントラベルを付与してください。
    - 不明な点は「データ不足により不明」と明記してください。

    【データ】
    {data_json}

    【出力形式】
    1. 各銘柄の個別分析（数値引用、ニュースのセンチメントとその根拠）
    2. 最終Top3ランキング
    3. 全体を通したリスク管理上の注意点
    """
    response = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
    return response.text

# --- 5. メイン画面の挙動 ---

if st.sidebar.button("🚀 グローバル精査を開始"):
    final_list = []
    all_dfs = {}
    
    with st.status("世界市場のデータを統合中...", expanded=True) as status:
        for ticker in TICKERS:
            try:
                st.write(f"⏳ {ticker}.{market_suffix} の多角分析中...")
                df = fetch_stock_data_with_fallback(ticker, market_suffix)
                m = calc_metrics(df)
                m["ticker"] = ticker
                m["news"] = fetch_news_headlines(ticker, news_params)
                
                if m["max_dd_252d"] is not None and m["max_dd_252d"] < RISK_DD_REJECT:
                    st.write(f"🚫 {ticker}: リスク（DD {m['max_dd_252d']}%）が許容範囲外のため除外")
                    continue
                
                final_list.append(m)
                all_dfs[ticker] = df
                time.sleep(0.5)
            except Exception as e:
                st.error(f"❌ {ticker} の分析をスキップ: {e}")
        status.update(label="精査完了", state="complete", expanded=False)

    if len(final_list) >= 1:
        report = run_structured_committee(final_list)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("🏆 投資委員会・最終評議")
            st.markdown(report)
        
        with col2:
            st.header("📈 データ・エビデンス")
            for m in final_list:
                with st.expander(f"📊 {m['ticker']} ({market_suffix})", expanded=True):
                    # 簡易チャート
                    fig = go.Figure()
                    df_plot = all_dfs[m['ticker']]
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name='Price'))
                    ma200 = df_plot['Close'].rolling(200).mean()
                    fig.add_trace(go.Scatter(x=df_plot.index, y=ma200, name='200MA', line=dict(dash='dash')))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("**最新ヘッドライン:**")
                    for h in m["news"]:
                        st.write(f"🔹 {h}")
                    
                    # ログ保存用データの表示
                    st.json(m)
        
        # 実行ログの保存
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"log_{run_id}.json", "w", encoding="utf-8") as f:
            json.dump({"report": report, "data": final_list}, f, ensure_ascii=False, indent=2)
    else:
        st.error("分析対象銘柄が足りません。")