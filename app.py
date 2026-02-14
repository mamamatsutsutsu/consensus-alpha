import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from google import genai
import requests
import json
from io import StringIO
from datetime import datetime, timezone
import time

# --- 1. 設定 & UI構成 ---
st.set_page_config(page_title="ConsensusAlpha", layout="wide")
st.title("🧠 ConsensusAlpha: 投資委員会エージェント")

# サイドバー設定
st.sidebar.header("設定")
api_key = st.sidebar.text_input("Gemini API Key", type="password", value="") # 毎回入れるか、コードに直書き
TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META"]
RISK_DD_REJECT = -40.0

# --- 2. 処理エンジン (前回までのロジックを統合) ---

def fetch_data(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=15)
    df = pd.read_csv(StringIO(r.content.decode("utf-8")))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()

def calc_metrics(df):
    c = df["Close"].astype(float)
    rows = len(c)
    r252 = ((c.iloc[-1] / c.iloc[-252]) - 1) * 100 if rows >= 252 else None
    r21  = ((c.iloc[-1] / c.iloc[-21]) - 1) * 100 if rows >= 21 else None
    mom_12_1 = (r252 - r21) if (r252 is not None and r21 is not None) else None
    vol_60d = c.pct_change().rolling(60).std().iloc[-1] * (252 ** 0.5) * 100 if rows >= 62 else None
    ma_200 = c.rolling(200).mean().iloc[-1]
    ma_200_gap = ((c.iloc[-1] / ma_200) - 1) * 100 if rows >= 200 else None
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

# --- 3. グラフ作成機能 (Plotly) ---

def create_chart(df, ticker):
    fig = go.Figure()
    # 株価チャート
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='株価', line=dict(color='#1f77b4')))
    # 200日移動平均
    ma200 = df['Close'].rolling(200).mean()
    fig.add_trace(go.Scatter(x=df.index, y=ma200, name='200日移動平均', line=dict(color='orange', dash='dash')))
    
    fig.update_layout(
        title=f"{ticker} 株価推移 (200MA付き)",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- 4. メイン画面の挙動 ---

if not api_key:
    st.warning("サイドバーに Gemini API キーを入力してください。")
else:
    client = genai.Client(api_key=api_key)
    if st.sidebar.button("🚀 分析を開始"):
        final_list = []
        all_dfs = {}
        
        with st.status("データを取得・分析中...", expanded=True) as status:
            for ticker in TICKERS:
                try:
                    st.write(f"🔍 {ticker} を精査中...")
                    df = fetch_data(ticker)
                    m = calc_metrics(df)
                    m["ticker"] = ticker
                    
                    if m["max_dd_252d"] < RISK_DD_REJECT:
                        st.write(f"⚠️ {ticker} はリスク過多のため除外")
                        continue
                    
                    final_list.append(m)
                    all_dfs[ticker] = df
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"{ticker} でエラー: {e}")
            status.update(label="分析完了！", state="complete", expanded=False)

        if len(final_list) >= 3:
            # AIレポートの作成
            prompt = f"以下の投資データを元に、Top3銘柄の選定レポートを作成してください。各エージェント（Quality, Value, Momentum, Heat, Risk）の視点を含めてください。データ：{json.dumps(final_list)}"
            response = client.models.generate_content(model='gemini-flash-latest', contents=prompt)
            
            # 画面表示 (左右に分割)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.header("🏆 投資委員会レポート")
                st.markdown(response.text)
            
            with col2:
                st.header("📈 テクニカルチャート")
                # Top3に選ばれた銘柄だけでなく、分析した銘柄を表示
                for m in final_list[:3]: # とりあえず上位3つを表示
                    st.subheader(f"{m['ticker']} (12-1Mom: {m['mom_12_1']}%)")
                    st.plotly_chart(create_chart(all_dfs[m['ticker']], m['ticker']), use_container_width=True)
                    st.write(f"**ボラティリティ:** {m['vol_60d']}% | **最大下落率:** {m['max_dd_252d']}%")
                    st.divider()
        else:
            st.error("分析可能な銘柄が足りませんでした。")