import os
import time
import re
import math
import urllib.parse
import urllib.request
import traceback
import xml.etree.ElementTree as ET
import email.utils
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# -----------------------------------------------------
# IMPORT EXTERNAL DICTIONARY
# -----------------------------------------------------
try:
    import universe
except ImportError:
    st.error("CRITICAL: 'universe.py' not found.")
    st.stop()

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AlphaLens Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦅"
)

# Session State
if "system_logs" not in st.session_state: st.session_state.system_logs = []
if "user_access_granted" not in st.session_state: st.session_state.user_access_granted = False
if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
if "last_market_key" not in st.session_state: st.session_state.last_market_key = None

def log_system_event(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")

def error_boundary(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_system_event(f"{func.__name__}: {str(e)}", "ERROR")
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
            return None
    return wrapper

# ==========================================
# 1. UI STYLING (High-End Professional)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Noto+Sans+JP:wght@400;700&display=swap');

:root {
  --bg: #050505; --panel: #0a0a0a; --card: #121212; --border: #333333;
  --accent: #00f2fe; --accent-2: #ff0055; --text: #e0e0e0;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; }
* { font-family: 'Noto Sans JP', sans-serif !important; letter-spacing: 0.03em !important; }
h1, h2, h3, .brand, .kpi-val { font-family: 'Orbitron', sans-serif !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

h1, h2, h3, .brand {
  background: linear-gradient(90deg, #fff, #00f2fe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900 !important;
  margin-bottom: 0px !important; padding-bottom: 5px;
}
.deck { background: var(--panel); border-bottom: 1px solid var(--accent); padding: 15px; margin-bottom: 20px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 20px; margin-bottom: 15px; }
div[data-testid="stDataFrame"] { background-color: #151515 !important; border: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] * { color: #ffffff !important; font-size: 12px !important; }
[data-testid="stHeader"] { background-color: #222 !important; border-bottom: 2px solid var(--accent) !important; }
div[data-baseweb="select"] > div { background-color: #111 !important; border-color: #444 !important; color: #fff !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #111 !important; border: 1px solid #444 !important; }
li[data-baseweb="option"]:hover { background-color: #222 !important; color: #00f2fe !important; }
button { background-color: #111 !important; color: var(--accent) !important; border: 1px solid #333 !important; border-radius: 4px !important; font-weight: 700 !important; }
button:hover { border-color: var(--accent) !important; box-shadow: 0 0 10px var(--accent) !important; color: #fff !important; }

.agent-row { display: flex; align-items: baseline; margin-bottom: 8px; padding: 10px; border-radius: 4px; background: #0f0f0f; border-left: 4px solid #555; font-size: 13px; line-height: 1.6; }
.agent-label { font-weight: 900; margin-right: 12px; white-space: nowrap; font-family: 'Orbitron'; letter-spacing: 1px; min-width: 110px; }
.agent-fundamental { border-left-color: #00f2fe; } .agent-fundamental .agent-label { color: #00f2fe; }
.agent-sentiment { border-left-color: #ff0055; } .agent-sentiment .agent-label { color: #ff0055; }
.agent-valuation { border-left-color: #00ff88; } .agent-valuation .agent-label { color: #00ff88; }
.agent-skeptic { border-left-color: #ffcc00; } .agent-skeptic .agent-label { color: #ffcc00; }
.agent-risk { border-left-color: #888; } .agent-risk .agent-label { color: #888; }
.agent-verdict { border: 1px solid #fff; background: #1a1a1a; padding: 20px; margin-top: 15px; font-weight: 500; font-size: 14px; line-height: 1.8; }
.market-box { background: #080808; border: 1px solid #333; padding: 20px; margin-bottom: 20px; font-size: 13px; line-height: 1.8; color: #ddd; }
.report-box { background: #111; border-top: 3px solid var(--accent); padding: 25px; margin-top: 10px; line-height: 1.9; color: #eee; font-size: 13px; white-space: pre-wrap; }
.highlight { color: #00f2fe; font-weight: bold; } .highlight-neg { color: #ff0055; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. AUTH & AI
# ==========================================
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
APP_PASS = st.secrets.get("APP_PASSWORD")

try:
    import google.generativeai as genai
    HAS_LIB = True
    if API_KEY: genai.configure(api_key=API_KEY)
except: HAS_LIB = False

def check_access():
    if not APP_PASS: return True
    if st.session_state.user_access_granted: return True
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<br><br><h3 style='text-align:center'>SECURITY GATE</h3>", unsafe_allow_html=True)
        with st.form("access_form"):
            p = st.text_input("PASSCODE", type="password")
            if st.form_submit_button("UNLOCK", use_container_width=True):
                if p == APP_PASS:
                    st.session_state.user_access_granted = True
                    st.rerun()
                else: st.error("DENIED")
    return False

if not check_access(): st.stop()

# ==========================================
# 3. SETTINGS & UTILS
# ==========================================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"
MARKETS = universe.MARKETS
NAME_DB = universe.NAME_DB

def get_name(t: str) -> str: return NAME_DB.get(t, t)

def sfloat(x):
    try: return float(x)
    except: return np.nan

# ==========================================
# 4. CORE ENGINES
# ==========================================
@st.cache_data(ttl=1800)
def fetch_market_data(tickers: Tuple[str, ...], period: str) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if t]))
    frames = []
    chunk = 40 
    for i in range(0, len(tickers), chunk):
        c = tickers[i:i+chunk]
        try:
            r = yf.download(" ".join(c), period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if not r.empty: frames.append(r)
        except: continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

def extract_close_prices(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0): close = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(1): close = df.xs("Close", axis=1, level=1)
            else: return pd.DataFrame()
        else: return pd.DataFrame()
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return close[[c for c in expected if c in close.columns]]
    except: return pd.DataFrame()

def calc_technical_metrics(s: pd.Series, b: pd.Series, win: int) -> Dict:
    s_clean, b_clean = s.dropna(), b.dropna()
    if len(s_clean) < win + 1 or len(b_clean) < win + 1: return None
    s_win, b_win = s.ffill().tail(win+1), b.ffill().tail(win+1)
    if s_win.isna().iloc[0] or b_win.isna().iloc[0]: return None

    p_ret = (s_win.iloc[-1]/s_win.iloc[0]-1)*100
    b_ret = (b_win.iloc[-1]/b_win.iloc[0]-1)*100
    rs = p_ret - b_ret
    
    half = max(1, win//2)
    p_half = (s_win.iloc[-1]/s_win.iloc[-half-1]-1)*100
    accel = p_half - (p_ret/2)
    dd = abs(((s_win/s_win.cummax()-1)*100).min())
    
    year_high = s_clean.tail(252).max() if len(s_clean) >= 252 else s_clean.max()
    high_dist = (s_win.iloc[-1] / year_high - 1) * 100 if year_high > 0 else 0
    
    rets = {}
    s_ffill = s.ffill()
    for l, d in [("1W",5), ("1M",21), ("3M",63), ("12M",252)]:
        if len(s_ffill) > d: rets[l] = (s_ffill.iloc[-1] / s_ffill.iloc[-1-d] - 1) * 100
        else: rets[l] = np.nan
    
    return {"RS": rs, "Accel": accel, "MaxDD": dd, "Ret": p_ret, "HighDist": high_dist, **rets}

def calculate_regime(bench_series: pd.Series) -> Tuple[str, float]:
    if len(bench_series) < 200: return "Unknown", 0.5
    curr = bench_series.iloc[-1]
    ma200 = bench_series.rolling(200).mean().iloc[-1]
    vol20 = bench_series.pct_change().tail(20).std() * np.sqrt(252)
    trend = "Bull" if curr > ma200 else "Bear"
    vol_state = "High" if vol20 > 0.15 else "Low"
    return f"{trend} / {vol_state} Vol", 0.6 if trend == "Bull" else 0.3

def audit_data_availability(expected: List[str], df: pd.DataFrame, win: int):
    present = [t for t in expected if t in df.columns]
    if not present: return {"ok": False, "list": []}
    last = df[present].apply(lambda x: x.last_valid_index())
    mode = last.mode().iloc[0] if not last.mode().empty else None
    computable = [t for t in present if last[t] == mode and len(df[t].dropna()) >= win + 1]
    return {"ok": True, "list": computable, "mode": mode, "count": len(computable), "total": len(expected)}

def calculate_zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# --- ADVANCED DATA ---
@st.cache_data(ttl=3600)
def fetch_fundamentals_batch(tickers: List[str]) -> pd.DataFrame:
    data = []
    def get_info(t):
        try:
            i = yf.Ticker(t).info
            return {
                "Ticker": t, "MCap": i.get("marketCap", 0), "PER": i.get("trailingPE", np.nan),
                "FwdPE": i.get("forwardPE", np.nan), "PBR": i.get("priceToBook", np.nan), "ROE": i.get("returnOnEquity", np.nan)
            }
        except: return {"Ticker": t, "MCap": 0}
    with ThreadPoolExecutor(max_workers=10) as executor:
        data = list(executor.map(get_info, tickers))
    return pd.DataFrame(data).set_index("Ticker")

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker: str) -> Dict[str, Any]:
    try:
        i = yf.Ticker(ticker).info
        return {
            "MCap": i.get("marketCap", np.nan), "PER": i.get("trailingPE", np.nan), "FwdPE": i.get("forwardPE", np.nan),
            "PBR": i.get("priceToBook", np.nan), "PEG": i.get("pegRatio", np.nan), "ROE": i.get("returnOnEquity", np.nan),
            "Target": i.get("targetMeanPrice", "N/A"), "Rec": i.get("recommendationKey", "N/A")
        }
    except: return {}

# --- AI & NEWS ---
def clean_ai_text(text: str) -> str:
    text = text.replace("**", "").replace('"', "")
    # ' は残す (e.g. McDonald's)
    return re.sub(r"\n{2,}", "\n", text).strip()

def enforce_market_format(text: str) -> str:
    if "【主な変動要因】" not in text: text += "\n【主な変動要因】\n(+ )\n(- )"
    text = re.sub(r"\n\s*\n(【主な変動要因】)", r"\n\1", text)
    text = re.sub(r"(。)(【主な変動要因】)", r"\1\n\2", text)
    if "【今後3ヶ月" not in text: text += "\n【今後3ヶ月のコンセンサス見通し】\n( )"
    text = re.sub(r"\n\s*\n(【今後3ヶ月[^】]*】)", r"\n\1", text)
    return text.strip()

def market_to_html(text: str) -> str:
    # (+)/(-) line highlighting
    text = re.sub(r"(^\(\+\).*$)", r"<span class='highlight'>\1</span>", text, flags=re.MULTILINE)
    text = re.sub(r"(^\(\-\).*$)", r"<span class='highlight-neg'>\1</span>", text, flags=re.MULTILINE)
    return text.replace("\n", "<br>")

@st.cache_data(ttl=1800)
def get_news_consolidated(ticker: str, name: str, limit_each: int = 10) -> Tuple[List[dict], str]:
    news_items, context_lines = [], []
    try:
        raw = yf.Ticker(ticker).news or []
        for n in raw[:limit_each]:
            t, l, p = n.get("title",""), n.get("link",""), n.get("providerPublishTime",0)
            news_items.append({"title":t, "link":l, "pub":p, "src":"Yahoo"})
            if t: context_lines.append(f"- [Yahoo] {t}")
    except: pass
    try:
        q = urllib.parse.quote(f"{name} 株")
        url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
        with urllib.request.urlopen(url, timeout=3) as r:
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:limit_each]:
                t, l, d = i.findtext("title"), i.findtext("link"), i.findtext("pubDate")
                try: pub = int(email.utils.parsedate_to_datetime(d).timestamp())
                except: pub = 0
                news_items.append({"title":t, "link":l, "pub":pub, "src":"Google"})
                if t: context_lines.append(f"- [Google] {t}")
    except: pass
    news_items.sort(key=lambda x: x["pub"], reverse=True)
    return news_items, "\n".join(context_lines[:15])

@st.cache_data(ttl=3600)
def generate_ai_content(prompt_key: str, context: Dict) -> str:
    if not HAS_LIB or not API_KEY: return "AI OFFLINE"
    
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    p = ""
    
    if prompt_key == "market":
        p = f"""
        期間:{context['s_date']}〜{context['e_date']}
        市場平均:{context['ret']:.2f}%
        最強:{context['top']} 最弱:{context['bot']}
        ニュース:{context['headlines']}
        
        この期間の市場概況をプロ向けに450-600字で解説せよ。
        「ベンチマーク」禁止。「市場平均」を使用。
        段落間の空行は禁止。改行は許可するが連続改行は禁止。
        1) 市場概況（因果と数値を結びつける）
        2) 【主な変動要因】（箇条書き。行頭に必ず(+)/(-)を付記）
        3) 【今後3ヶ月のコンセンサス見通し】（金利・景気・需給の観点から簡潔に）
        """
    elif prompt_key == "sector_debate":
        p = f"""
        あなたは5名の専門エージェント(Fundamental, Sentiment, Valuation, Skeptic, Risk)。
        対象セクター:{context['sec']}
        候補データ（必ず引用し比較せよ）:
        {context['candidates']}
        セクター関連ニュース:
        {context.get('news','')}
        
        タスク: セクター内で「最も推奨できる銘柄（ロング）」を1つ、次点を2つ選ぶ。
        可能なら「回避すべき銘柄（ネガティブ）」を1つ挙げる。
        推奨理由は必ず「定量(RS/Ret)→ファンダ(ROE/成長)→バリュエーション(PER/PBR)→ニュース」の順で。
        
        出力フォーマット（タグ必須）:
        [FUNDAMENTAL] ...
        [SENTIMENT] ...
        [VALUATION] ...
        [SKEPTIC] ...
        [RISK] ...
        [JUDGE] （従来の3〜5倍の分量。トップ1・次点2・ネガ1を明確に結論。条件付きの反証・注意点も書く）
        """
    elif prompt_key == "stock_report":
        p = f"""
        銘柄:{context['name']} ({context['ticker']})
        基礎データ:{context['fund_str']}
        比較:{context['m_comp']}
        ニュース:{context['news']}
        
        プロのアナリストレポートを作成せよ。文体は「だ・である」。記号禁止。
        分量は充実させること（約1000文字）。
        1. 定量サマリー（株価/時価総額/リターン/PER/PBR）
        2. バリュエーション（市場平均・セクター平均と比較して割安か？）
        3. 需給/センチメント
        4. ニュース/非構造情報（事象→業績への含意）
        5. 3ヶ月見通し（ベース/強気/弱気シナリオ）
        6. 監視ポイント
        """

    for m in models:
        try:
            model = genai.GenerativeModel(m)
            text = model.generate_content(p).text
            return clean_ai_text(text)
        except Exception as e:
            if "429" in str(e): time.sleep(1); continue
    return "AI Unavailable"

def parse_agent_debate(text: str) -> str:
    mapping = {
        "[FUNDAMENTAL]": ("agent-fundamental", "FUNDAMENTAL"),
        "[SENTIMENT]": ("agent-sentiment", "SENTIMENT"),
        "[VALUATION]": ("agent-valuation", "VALUATION"),
        "[SKEPTIC]": ("agent-skeptic", "SKEPTIC"),
        "[RISK]": ("agent-risk", "RISK"),
        "[JUDGE]": ("agent-verdict", "JUDGE")
    }
    clean = clean_ai_text(text.replace("```html", "").replace("```", ""))
    parts = re.split(r'(\[[A-Z]+\])', clean)
    html = ""
    curr_cls, label, buffer = "agent-box", "", ""
    
    def flush():
        nonlocal html, buffer, curr_cls, label
        if curr_cls and label and buffer.strip():
            b = re.sub(r"\s*\n+\s*", " ", buffer).strip()
            html += f"<div class='agent-row {curr_cls}'><div class='agent-label'>{label}</div><div>{b}</div></div>"
        buffer = ""

    for part in parts:
        part = part.strip()
        if not part: continue
        if part in mapping:
            flush()
            curr_cls, label = mapping[part]
        else:
            buffer += (" " + part)
            
    flush()
    if not html: html = f"<div class='agent-box'>{clean}</div>"
    return html

# ==========================================
# 5. MAIN UI LOGIC
# ==========================================
@error_boundary
def main():
    st.markdown("<h1 class='brand'>ALPHALENS</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### SYSTEM LOGS")
        if st.button("CLEAR LOGS"): st.session_state.system_logs = []; st.rerun()

    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 0.6])
    with c1: market_key = st.selectbox("MARKET", list(MARKETS.keys()))
    with c2: lookback_key = st.selectbox("WINDOW", list(LOOKBACKS.keys()), index=1)
    with c3: st.caption(f"FETCH: {FETCH_PERIOD}"); st.progress(100)
    with c4: 
        st.write("")
        sync = st.button("SYNC", type="primary", use_container_width=True)

    if st.session_state.last_market_key != market_key:
        st.session_state.selected_sector = None
        st.session_state.last_market_key = market_key

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if sync or "core_df" not in st.session_state:
        with st.spinner("SYNCING MARKET DATA..."):
            raw = fetch_market_data(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close_prices(raw, core_tickers)
    
    core_df = st.session_state.get("core_df", pd.DataFrame())
    if core_df.empty or len(core_df) < win + 1: st.warning("WAITING FOR DATA..."); return

    audit = audit_data_availability(core_tickers, core_df, win)
    if bench not in audit["list"]: st.error("BENCHMARK MISSING"); return

    # 1. Market Overview
    b_stats = calc_technical_metrics(core_df[bench], core_df[bench], win)
    if not b_stats: st.error("BENCH ERROR"); return

    regime, weight_mom = calculate_regime(core_df[bench].dropna())
    
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit["list"]:
            res = calc_technical_metrics(core_df[s_t], core_df[bench], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    if not sec_rows: st.warning("SECTOR DATA INSUFFICIENT"); return
    sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)
    
    s_date = core_df.index[-win-1].strftime('%Y/%m/%d')
    e_date = core_df.index[-1].strftime('%Y/%m/%d')
    _, market_context = get_news_consolidated(bench, m_cfg["name"])
    
    market_text = generate_ai_content("market", {
        "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
        "top": sdf.iloc[-1]["Sector"], "bot": sdf.iloc[0]["Sector"],
        "market_name": m_cfg["name"], "headlines": market_context
    })
    market_text = enforce_market_format(market_text)
    market_html = market_to_html(market_text)
    
    st.markdown(f"""
    <div class='market-box'>
    <b>MARKET PULSE ({s_date} - {e_date})</b> | Spread: {sdf.iloc[-1]['RS'] - sdf.iloc[0]['RS']:.1f}pts | Regime: {regime}<br>
    {market_html}
    </div>
    """, unsafe_allow_html=True)

    # 2. Sector Rotation
    st.subheader(f"SECTOR ROTATION ({s_date} - {e_date})")
    
    if not sdf.empty and "Ret" in sdf.columns:
        top_row = sdf.iloc[-1]
        bot_row = sdf.iloc[0]
        rot_sum = (
            f"市場平均: <span class='highlight'>{b_stats['Ret']:.2f}%</span> | "
            f"最強: <span class='highlight'>{top_row['Sector']}</span> ({top_row.get('Ret',0):.2f}%) | "
            f"最弱: <span class='highlight-neg'>{bot_row['Sector']}</span> ({bot_row.get('Ret',0):.2f}%)"
        )
        st.markdown(f"<div style='margin-bottom:10px; font-size:13px'>{rot_sum}</div>", unsafe_allow_html=True)

    sdf["Label"] = sdf["Sector"] + " (" + sdf["Ret"].apply(lambda x: f"{x:+.1f}%") + ")"
    colors = ["#333"] * len(sdf)
    if st.session_state.selected_sector in sdf["Sector"].values:
        idx = sdf[sdf["Sector"] == st.session_state.selected_sector].index[0]
        colors[sdf.index.get_loc(idx)] = "#00f2fe"

    fig = px.bar(sdf, x="RS", y="Label", orientation='h', title=f"Relative Strength ({lookback_key})")
    fig.update_traces(customdata=np.stack([sdf["Ret"]], axis=-1), hovertemplate="%{y}<br>Ret: %{customdata[0]:+.1f}%<br>RS: %{x:.2f}<extra></extra>", marker_color=colors)
    fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', font_family="Orbitron", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True, 'displayModeBar': False}, key="sector_chart")
    
    st.write("SELECT SECTOR:")
    cols = st.columns(2)
    for i, row in enumerate(sdf.itertuples()):
        s = row.Sector
        label = f"✅ {s}" if s == st.session_state.selected_sector else s
        if cols[i%2].button(label, key=f"btn_{s}", use_container_width=True):
            st.session_state.selected_sector = s
            st.rerun()
            
    target_sector = st.session_state.selected_sector or sdf.iloc[-1]["Sector"]
    if target_sector: st.caption(f"Current: **{target_sector}** → [Jump to Analysis](#sector_anchor)")

    # 3. Sector Forensic
    st.markdown(f"<div id='sector_anchor'></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader(f"SECTOR FORENSIC: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    if not stock_list: st.warning("No stocks."); return

    full_list = [bench] + stock_list
    cache_key = f"{market_key}_{target_sector}_{lookback_key}"
    
    if cache_key != st.session_state.get("sec_cache_key") or sync:
        with st.spinner(f"ANALYZING {len(stock_list)} STOCKS..."):
            raw_s = fetch_market_data(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close_prices(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_data_availability(full_list, sec_df, win)
    
    results = []
    for t in [x for x in s_audit["list"] if x != bench]:
        stats = calc_technical_metrics(sec_df[t], sec_df[bench], win)
        if stats:
            stats["Ticker"] = t
            stats["Name"] = get_name(t)
            results.append(stats)
            
    if not results: st.warning("NO DATA."); return
    df = pd.DataFrame(results)
    
    # Pre-fetch fundamentals for top candidates (Apex sort first)
    df["Apex"] = weight_mom * calculate_zscore(df["RS"]) + (0.8 - weight_mom) * calculate_zscore(df["Accel"]) + 0.2 * calculate_zscore(df["Ret"])
    df = df.sort_values("Apex", ascending=False)
    
    # Fetch fundamentals for top 30
    tickers_for_fund = df["Ticker"].head(30).tolist()
    with st.spinner("Fetching Fundamentals..."):
        fund_df = fetch_fundamentals_batch(tickers_for_fund)
        df = df.join(fund_df, on="Ticker")
    
    # 4. 5-AGENT SECTOR COUNCIL
    st.markdown("##### 🦅 5-AGENT SECTOR COUNCIL (Top Picks)")
    
    top3 = df.head(3).copy()
    neg = df.sort_values(["RS","MaxDD"], ascending=[True, False]).head(1)
    
    cand_lines = []
    for _, r in top3.iterrows():
        cand_lines.append(f"{r['Name']}({r['Ticker']}): Ret {r['Ret']:.1f}%, RS {r['RS']:.2f}, Acc {r['Accel']:.2f}, HighDist {r['HighDist']:.1f}%, MCap {sfloat(r.get('MCap',0))/1e9:.1f}B, PER {sfloat(r.get('PER',np.nan)):.1f}, PBR {sfloat(r.get('PBR',np.nan)):.1f}")
    
    if not neg.empty:
        nr = neg.iloc[0]
        cand_lines.append(f"\n[AVOID] {nr['Name']}: Ret {nr['Ret']:.1f}%, RS {nr['RS']:.2f}")

    _, sec_news = get_news_consolidated(m_cfg["sectors"][target_sector], target_sector, limit_each=5)
    
    sec_ai_raw = generate_ai_content("sector_debate", {
        "sec": target_sector, "count": len(df), "candidates": "\n".join(cand_lines),
        "top5": ", ".join(top3["Name"].tolist()), "avg_rs": f"{df['RS'].mean():.2f}", "news": sec_news
    })
    st.markdown(parse_agent_debate(sec_ai_raw), unsafe_allow_html=True)
    
    with st.expander("EVIDENCE: TOP CANDIDATES (Definitions)", expanded=False):
        st.markdown("- **Apex**: 投資魅力度スコア (RS + Accel + Ret)\n- **RS**: 市場平均に対する相対強度\n- **HighDist**: 52週高値からの乖離率")
        st.dataframe(top3[["Name","Apex","RS","Ret","HighDist","PER","PBR"]], hide_index=True, use_container_width=True)

    # 5. Leaderboard
    st.markdown("##### LEADERBOARD")
    
    def fmt_mcap(x):
        if pd.isna(x) or x == 0: return "-"
        if x >= 1e12: return f"{x/1e12:.1f}T"
        if x >= 1e9: return f"{x/1e9:.1f}B"
        return f"{x/1e6:.0f}M"
    
    df["MCapDisp"] = df["MCap"].apply(fmt_mcap)
    df_sorted = df.sort_values("MCap", ascending=False)
    
    event = st.dataframe(
        df_sorted[["Name", "MCapDisp", "PER", "PBR", "Apex", "RS", "1M", "12M"]],
        column_config={
            "MCapDisp": st.column_config.TextColumn("Market Cap"),
            "Apex": st.column_config.NumberColumn(format="%.2f"),
            "RS": st.column_config.ProgressColumn(format="%.1f%%", min_value=-20, max_value=20),
            "PER": st.column_config.NumberColumn(format="%.1f"),
            "PBR": st.column_config.NumberColumn(format="%.1f"),
            "1M": st.column_config.NumberColumn(format="%.1f%%"),
            "12M": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table"
    )
    
    # 6. Deep Dive
    top = df_sorted.iloc[0]
    try:
        if hasattr(event, "selection") and event.selection:
            sel_rows = event.selection.get("rows", [])
            if sel_rows: top = df_sorted.iloc[sel_rows[0]]
    except: pass

    st.divider()
    st.markdown(f"### 🦅 DEEP DIVE: {top['Name']}")
    
    news_items, news_context = get_news_consolidated(top["Ticker"], top["Name"], limit_each=10)
    fund_data = get_fundamental_data(top["Ticker"])
    bench_fd = get_fundamental_data(bench)
    
    # Comparison Context
    bench_per = sfloat(bench_fd.get("PER"))
    sector_per = pd.to_numeric(df["PER"], errors="coerce").median()
    stock_per = sfloat(fund_data.get("PER"))
    m_comp = f"市場平均PER: {bench_per:.1f}倍 / セクター中央値PER: {sector_per:.1f}倍 / 当該銘柄PER: {stock_per:.1f}倍"
    
    fund_str = f"PER:{fund_data.get('PER')}, PBR:{fund_data.get('PBR')}, PEG:{fund_data.get('PEG')}, Target:{fund_data.get('Target')}"

    report_txt = generate_ai_content("stock_report", {
        "name": top["Name"], "ticker": top["Ticker"],
        "fund_str": fund_str, "m_comp": m_comp, "news": news_context
    })
    
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        st.markdown(f"<div class='report-box'><b>ANALYST REPORT</b><br>{report_txt}</div>", unsafe_allow_html=True)
    with nc2:
        st.caption("INTEGRATED NEWS FEED")
        for n in news_items[:20]:
            dt = datetime.fromtimestamp(n["pub"]).strftime("%Y/%m/%d") if n["pub"] else "-"
            st.markdown(f"- {dt} [{n['src']}] [{n['title']}]({n['link']})")

if __name__ == "__main__":
    main()