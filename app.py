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
    initial_sidebar_state="collapsed",
    page_icon="🦅"
)

# Session State
if "system_logs" not in st.session_state: st.session_state.system_logs = []
if "user_access_granted" not in st.session_state: st.session_state.user_access_granted = False
if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
if "last_market_key" not in st.session_state: st.session_state.last_market_key = None

def log_system_event(msg: str, level: str = "INFO", tag: str = "SYS"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] [{tag}] {msg}"
    print(line)
    st.session_state.system_logs.append(line)
    st.session_state.system_logs = st.session_state.system_logs[-300:]

def error_boundary(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_system_event(f"{func.__name__}: {str(e)}", "ERROR", "SYS")
            st.error(f"⚠️ SYSTEM ERROR: {str(e)}")
            return None
    return wrapper

# ==========================================
# 1. UI STYLING (CYBER TERMINAL - FONT FIXED)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&family=Orbitron:wght@400;600;900&family=Noto+Sans+JP:wght@400;700&display=swap');

:root {
  --bg: #000000;
  --panel: #0a0a0a;
  --card: #111111;
  --border: #333333;
  --accent: #00f2fe;     /* Cyan */
  --accent-2: #ff0055;   /* Pink */
  --accent-3: #00ff88;   /* Green */
  --text: #e0e0e0;
}

/* FONT POLICY: SYSTEM LOGS FIRST */
html, body, .stApp { 
  background-color: var(--bg) !important; 
  color: var(--text) !important; 
  font-family: 'Roboto Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Noto Sans JP", monospace !important;
}

/* Force Monospace everywhere */
* { 
  font-family: inherit !important; 
  letter-spacing: 0.02em !important; 
}

/* BRAND & HEADERS: Orbitron */
.brand, h1, h2, h3, .kpi-val, .agent-label, div[data-testid="stMetricValue"], .orbitron { 
  font-family: 'Orbitron', sans-serif !important; 
  letter-spacing: 0.05em !important; 
  text-transform: uppercase;
}

/* CAPTIONS: Orbitron for Pro look */
.caption-text, div[data-testid="stCaptionContainer"] *, div[data-testid="stMarkdownContainer"] small {
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.06em !important;
}

/* HIDE DEFAULTS */
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
div[data-testid="stSidebar"] { display: none; }

/* BRANDING */
h1, h2, h3, .brand {
  background: linear-gradient(90deg, #fff, #00f2fe);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 900 !important;
  padding-bottom: 5px;
}

/* CONTAINERS */
.deck { background: var(--panel); border-bottom: 1px solid var(--accent); padding: 15px; margin-bottom: 20px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 0px; padding: 20px; margin-bottom: 15px; }

/* TABLE */
div[data-testid="stDataFrame"] { background-color: #050505 !important; border: 1px solid #444 !important; }
div[data-testid="stDataFrame"] * { color: #f0f0f0 !important; font-size: 11px !important; font-family: 'Roboto Mono', monospace !important; }
[data-testid="stHeader"] { background-color: #151515 !important; border-bottom: 2px solid var(--accent) !important; }

/* INPUTS & BUTTONS */
div[data-baseweb="select"] > div { background-color: #111 !important; border-color: #555 !important; color: #fff !important; }
div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #111 !important; border: 1px solid #555 !important; }
button {
  background-color: #111 !important; color: var(--accent) !important;
  border: 1px solid #444 !important; border-radius: 0px !important;
  font-family: 'Orbitron', sans-serif !important; font-weight: 700 !important; 
}
button:hover { border-color: var(--accent) !important; box-shadow: 0 0 10px rgba(0, 242, 254, 0.4) !important; color: #fff !important; }

/* AGENT COUNCIL (MOBILE OPTIMIZED) */
.agent-row {
    display: flex; align-items: flex-start; gap: 10px;
    margin-bottom: 8px; padding: 10px;
    border-radius: 0px; background: #080808; border-left: 3px solid #555;
    font-size: 12px; line-height: 1.6; width: 100%; box-sizing: border-box;
}
.agent-label { 
    flex: 0 0 70px; max-width: 70px; min-width: 70px;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700; text-align: right; line-height: 1.2;
    padding-top: 2px; white-space: normal; word-break: break-word;
}
.agent-content { 
    flex: 1 1 auto; min-width: 0; 
    overflow-wrap: anywhere; word-break: break-word; 
}

.agent-fundamental { border-left-color: #00f2fe; } .agent-fundamental .agent-label { color: #00f2fe; }
.agent-sentiment { border-left-color: #ff0055; } .agent-sentiment .agent-label { color: #ff0055; }
.agent-valuation { border-left-color: #00ff88; } .agent-valuation .agent-label { color: #00ff88; }
.agent-skeptic { border-left-color: #ffcc00; } .agent-skeptic .agent-label { color: #ffcc00; }
.agent-risk { border-left-color: #888; } .agent-risk .agent-label { color: #888; }
.agent-verdict { 
    border: 1px solid #fff; background: #111; padding: 20px; 
    margin-top: 15px; font-weight: 500; font-size: 13px; line-height: 1.8;
    word-break: break-word; overflow-wrap: anywhere; box-sizing: border-box; width: 100%;
}
.agent-outlook {
    border: 1px solid #00f2fe; background: rgba(0, 242, 254, 0.05); padding: 15px;
    margin-bottom: 10px; font-weight: bold; color: #fff; border-left: 5px solid #00f2fe;
}

/* BOXES */
.log-top {
  font-family: 'Roboto Mono', monospace !important; font-size: 11px !important; color: #cfcfcf !important;
  background: #050505 !important; border: 1px solid #333 !important; padding: 10px !important;
  margin: 8px 0 14px 0 !important; max-height: 110px !important; overflow-y: auto !important;
  white-space: pre-wrap !important; line-height: 1.45 !important;
}
.log-mobile {
  font-family: 'Roboto Mono', monospace !important; font-size: 12px !important; color: #e6e6e6 !important;
  background: #050505 !important; border: 1px solid #333 !important; padding: 12px !important;
  max-height: 240px !important; overflow-y: auto !important;
  white-space: pre-wrap !important; line-height: 1.55 !important;
}
div[data-testid="stExpander"] details { border: 1px solid #333 !important; background:#070707 !important; }
div[data-testid="stExpander"] summary { font-family:'Roboto Mono', monospace !important; font-size:12px !important; color:#00f2fe !important; }
div[data-testid="stExpander"] summary:hover { background:#0b0b0b !important; }

.market-box {
    background: #080808; border: 1px solid #333; padding: 20px;
    margin-bottom: 20px; font-size: 12px; line-height: 1.8; color: #ddd;
}
.report-box {
    background: #0a0a0a; border-top: 2px solid var(--accent);
    padding: 20px; margin-top: 10px; line-height: 1.8; color: #eee; font-size: 12px;
    white-space: pre-wrap; 
}
.highlight { color: #00f2fe; font-weight: bold; } .highlight-neg { color: #ff0055; font-weight: bold; }
.def-text { font-size: 11px; color: #888; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 8px; line-height: 1.6; font-family: 'Orbitron', sans-serif !important; }
.caption-text { font-size: 11px; color: #666; margin-top: 5px; font-family: 'Orbitron', sans-serif !important; }
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
        st.markdown("<h3 style='text-align:center'>SECURITY GATE</h3>", unsafe_allow_html=True)
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

@st.cache_data(ttl=86400)
def fetch_name_fallback(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        n = info.get("shortName") or info.get("longName")
        if n and isinstance(n, str) and len(n) >= 2: return n
    except: pass
    return ticker

def get_name(t: str) -> str:
    n = NAME_DB.get(t)
    if n and n != t: return n
    return fetch_name_fallback(t)

def sfloat(x):
    try: return float(x)
    except: return np.nan

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def sentiment_label(score: int) -> str:
    if score >= 3: return "POSITIVE"
    if score <= -3: return "NEGATIVE"
    return "NEUTRAL"

def dash(x, fmt="%.1f"):
    if pd.isna(x): return "-"
    try: return fmt % float(x)
    except: return "-"

def pct(x, fmt="%.1f"):
    if pd.isna(x): return "-"
    try: return (fmt % (float(x)*100)) + "%"
    except: return "-"

def outlook_date_slots(days: List[int] = [7, 21, 35, 49, 63, 84]) -> List[str]:
    base = datetime.now().date()
    return [(base + timedelta(days=d)).strftime("%Y/%m/%d") for d in days]

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
    trend = "Bull" if curr > ma200 else "Bear"
    return trend, 0.6 if trend == "Bull" else 0.3

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

# --- FUNDAMENTALS & NEWS ---
@st.cache_data(ttl=3600)
def fetch_fundamentals_batch(tickers: List[str]) -> pd.DataFrame:
    data = []
    def get_info(t):
        try:
            i = yf.Ticker(t).info
            pe = i.get("trailingPE", np.nan)
            if pe is not None and pe < 0: pe = np.nan
            pbr = i.get("priceToBook", np.nan)
            if pbr is not None and pbr < 0: pbr = np.nan
            return {
                "Ticker": t, "MCap": i.get("marketCap", 0),
                "PER": pe, "PBR": pbr, "FwdPE": i.get("forwardPE", np.nan),
                "ROE": i.get("returnOnEquity", np.nan),
                "OpMargin": i.get("operatingMargins", np.nan),
                "RevGrow": i.get("revenueGrowth", np.nan),
                "Beta": i.get("beta", np.nan)
            }
        except: return {"Ticker": t, "MCap": 0}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        data = list(executor.map(get_info, tickers))
    return pd.DataFrame(data).set_index("Ticker")

@st.cache_data(ttl=3600)
def get_fundamental_data(ticker: str) -> Dict[str, Any]:
    try:
        i = yf.Ticker(ticker).info
        pe = i.get("trailingPE", np.nan)
        if isinstance(pe, (int, float)) and pe < 0: pe = np.nan
        pbr = i.get("priceToBook", np.nan)
        if isinstance(pbr, (int, float)) and pbr < 0: pbr = np.nan
        
        return {
            "MCap": i.get("marketCap", 0), "PER": pe, "FwdPE": i.get("forwardPE", np.nan),
            "PBR": pbr, "PEG": i.get("pegRatio", np.nan), "Target": i.get("targetMeanPrice", np.nan),
            "Rec": i.get("recommendationKey", "N/A")
        }
    except: return {"PRICE": np.nan, "MCap": np.nan, "PER": np.nan, "FwdPE": np.nan, "PBR": np.nan, "PEG": np.nan}

@st.cache_data(ttl=3600)
def fetch_earnings_dates(ticker: str) -> Dict[str,str]:
    out = {}
    try:
        cal = yf.Ticker(ticker).calendar
        # yfinance calendar structure varies
        if cal is not None:
             # Check if it's a dict or df
            if isinstance(cal, dict):
                if 'Earnings Date' in cal:
                    out["EarningsDate"] = str(cal['Earnings Date'][0])
            elif isinstance(cal, pd.DataFrame):
                 for k in ["Earnings Date", "EarningsDate"]:
                    if k in cal.index:
                        v = cal.loc[k].values
                        out["EarningsDate"] = ", ".join([str(x)[:10] for x in v if str(x) != "nan"])
    except: pass
    return out

# --- TEXT PROCESSING ---
def clean_ai_text(text: str) -> str:
    text = text.replace("```text", "").replace("```", "")
    text = text.replace("**", "").replace('"', "").replace("'", "")
    text = re.sub(r"(?m)^\s*text\s*$", "", text)
    text = re.sub(r"(?m)^\s*#{2,}\s*", "", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def force_nonempty_outlook_market(text: str, trend: str, ret: float, spread: float, market_key: str) -> str:
    m = re.search(r"【今後3ヶ月[^】]*】\n?(.*)", text, flags=re.DOTALL)
    body = m.group(1).strip() if m else ""
    if len(re.sub(r"[\s\(\)・\-−\n]", "", body)) >= 30: return text

    slots = outlook_date_slots()
    if "US" in market_key:
        events = [
            f"FOMC({slots[1]})→金利織り込み再計算でハイPERの変動が増幅",
            f"CPI/PCE({slots[0]})→インフレ鈍化ならリスクオン、再加速ならリスクオフ",
            f"雇用統計({slots[0]})→賃金の粘着性が長期金利を左右",
            f"主要決算({slots[2]})→ガイダンスで指数寄与が集中しやすい",
            f"クレジット/流動性({slots[3]})→スプレッド拡大は株の上値抑制",
            f"需給イベント({slots[4]})→オプション・リバランスで短期スパイク"
        ]
    else:
        events = [
            f"日銀会合({slots[1]})→金利と円が同時に動き、外需/内需の優劣が反転しやすい",
            f"米金利・円相場({slots[0]})→輸出・インバウンドの感応度が高い",
            f"主要決算({slots[2]})→通期見通し修正と株主還元が需給を決める",
            f"指数リバランス({slots[3]})→需給歪みで短期変動が出やすい",
            f"賃上げ・物価({slots[4]})→実質賃金で消費関連の相対が動く",
            f"海外投資家フロー({slots[5]})→資金流入の継続性が地合いを規定"
        ]

    fallback = "【今後3ヶ月のコンセンサス見通し】\n" + "\n".join([f"・{e}" for e in events]) + \
               f"\n・強気条件：インフレ鎮静化＋業績ガイダンス上振れ（基調:{trend}）\n・弱気条件：金利再上昇＋ガイダンス下方修正の連鎖"

    if "【今後3ヶ月" in text:
        text = re.sub(r"【今後3ヶ月[^】]*】.*", fallback, text, flags=re.DOTALL)
    else:
        text = text.rstrip() + "\n" + fallback
    return text

def enforce_market_format(text: str) -> str:
    if "【主な変動要因】" not in text: text += "\n【主な変動要因】\n(+ )\n(- )"
    text = re.sub(r"\n\s*\n(【主な変動要因】)", r"\n\1", text)
    text = re.sub(r"(。)(【主な変動要因】)", r"\1\n\2", text)
    if "【今後3ヶ月" not in text: text += "\n【今後3ヶ月のコンセンサス見通し】\n( )"
    text = re.sub(r"\n\s*\n(【今後3ヶ月[^】]*】)", r"\n\1", text)
    return text.strip()

def enforce_da_dearu_soft(text: str) -> str:
    text = re.sub(r"です。", "だ。", text)
    text = re.sub(r"です$", "だ", text, flags=re.MULTILINE)
    text = re.sub(r"ます。", "する。", text)
    text = re.sub(r"ます$", "する", text, flags=re.MULTILINE)
    return text

def market_to_html(text: str) -> str:
    text = re.sub(r"(^\(\+\).*$)", r"<span class='highlight'>\1</span>", text, flags=re.MULTILINE)
    text = re.sub(r"(^\(\-\).*$)", r"<span class='highlight-neg'>\1</span>", text, flags=re.MULTILINE)
    return text.replace("\n", "<br>")

@st.cache_data(ttl=1800)
def get_news_consolidated(ticker: str, name: str, market_key: str, limit_each: int = 10) -> Tuple[List[dict], str, int, Dict[str,int]]:
    news_items, context_lines = [], []
    pos_words = ["増益", "最高値", "好感", "上昇", "自社株買い", "上方修正", "急騰", "beat", "high", "jump"]
    neg_words = ["減益", "安値", "嫌気", "下落", "下方修正", "急落", "赤字", "miss", "low", "drop"]
    sentiment_score = 0
    meta = {"yahoo":0, "google":0, "pos":0, "neg":0}

    # Yahoo
    try:
        raw = yf.Ticker(ticker).news or []
        for n in raw[:limit_each]:
            t, l, p = n.get("title",""), n.get("link",""), n.get("providerPublishTime",0)
            news_items.append({"title":t, "link":l, "pub":p, "src":"Yahoo"})
            if t:
                meta["yahoo"] += 1
                # Weight by recency (48h)
                weight = 2 if (time.time() - p) < 172800 else 1
                context_lines.append(f"- [Yahoo] {t}")
                if any(w in t for w in pos_words): sentiment_score += 1*weight; meta["pos"] += 1
                if any(w in t for w in neg_words): sentiment_score -= 1*weight; meta["neg"] += 1
    except: pass

    # Google
    try:
        if "US" in market_key:
            hl, gl, ceid = "en", "US", "US:en"
            q = urllib.parse.quote(f"{name} stock")
        else:
            hl, gl, ceid = "ja", "JP", "JP:ja"
            q = urllib.parse.quote(f"{name} 株")
            
        url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
        with urllib.request.urlopen(url, timeout=3) as r:
            root = ET.fromstring(r.read())
            for i in root.findall(".//item")[:limit_each]:
                t, l, d = i.findtext("title"), i.findtext("link"), i.findtext("pubDate")
                try: pub = int(email.utils.parsedate_to_datetime(d).timestamp())
                except: pub = 0
                news_items.append({"title":t, "link":l, "pub":pub, "src":"Google"})
                if t:
                    meta["google"] += 1
                    weight = 2 if (time.time() - pub) < 172800 else 1
                    context_lines.append(f"- [Google] {t}")
                    if any(w in t for w in pos_words): sentiment_score += 1*weight; meta["pos"] += 1
                    if any(w in t for w in neg_words): sentiment_score -= 1*weight; meta["neg"] += 1
    except: pass

    news_items.sort(key=lambda x: x["pub"], reverse=True)
    return news_items, "\n".join(context_lines[:15]), sentiment_score, meta

def temporal_sanity_flags(text: str) -> List[str]:
    bad = ["年末年始", "クリスマス", "夏休み", "お盆", "来年", "昨年末"]
    return [w for w in bad if w in text]

@st.cache_data(ttl=3600)
def generate_ai_content(prompt_key: str, context: Dict) -> str:
    if not HAS_LIB or not API_KEY: return "AI OFFLINE"
    
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    p = ""
    market_n = context.get('market_name', 'Global')
    
    slots = context.get("date_slots", [])
    slot_line = " / ".join(slots) if slots else ""
    today_str = datetime.now().strftime('%Y年%m月%d日')

    if prompt_key == "market":
        p = f"""
        現在: {today_str} (この日付を基準に分析せよ)
        対象市場: {market_n} (これ以外の市場の話は禁止)
        期間:{context['s_date']}〜{context['e_date']}
        市場平均:{context['ret']:.2f}%
        最強:{context['top']} 最弱:{context['bot']}
        ニュース:{context['headlines']}
        
        この期間の{market_n}市場概況をプロ向けに450-600字で解説せよ。
        「ベンチマーク」禁止。「市場平均」を使用。
        段落間の空行禁止。改行は許可するが連続改行禁止。
        
        必ず次の順番で出力せよ（見出しは固定）：
        1) 市場概況（材料→結果を因果で、数値必須）
        2) 【主な変動要因】（3〜6行、各行は必ず(+)または(-)で開始）
        3) 【今後3ヶ月のコンセンサス見通し】
        - 予定日は必ず次の候補日から選んで書け：{slot_line}
        - 90日以内に起きやすい具体イベント/予定を最大6つ列挙
        - 各行は「イベント名(YYYY/MM/DD)→株価に効きやすい方向→理由」
        - 最後に強気/弱気の条件分岐
        - この期間から外れる季節表現（年末年始、来年など）は禁止
        """
    elif prompt_key == "sector_debate":
        p = f"""
        現在: {today_str}
        あなたは5名の専門エージェント。対象市場は{market_n}。
        対象セクター:{context['sec']}
        候補データ:
        {context['candidates']}
        ニュース（非構造）:
        {context.get('news','')}
        セクターニュース要約: {context.get('sector_news','')}

        厳守ルール:
        - 文体は「だ・である」。です・ます調は禁止。
        - 各エージェントは最低8行以上。短文禁止。具体で書く。
        - モメンタム（RS/Accel）とセンチメント（ニュース）を最重視せよ。
        - 「抽象語（不透明、堅調、注視、様子見）」は禁止。

        タスク:
        1) まず冒頭に[SECTOR_OUTLOOK]タグで、セクター全体の見通し（{today_str}から3ヶ月）を宣言抜きで記述。
        2) その後、各エージェントが「推奨銘柄（ロング）」と「回避銘柄（ショート）」を議論。
        
        [JUDGE]では、トップピック1銘柄と次点2銘柄を決定し、その論理的根拠を詳細（従来の5倍の分量）に記述せよ。
        候補データ(candidates)は必ず引用し、比較（AよりBが良い/条件）で結論を出すこと。
        ネガティブな銘柄があれば具体的に指摘せよ。
        
        出力フォーマット（タグ厳守）:
        [SECTOR_OUTLOOK] ...
        [FUNDAMENTAL] ...
        [SENTIMENT] ...
        [VALUATION] ...
        [SKEPTIC] ...
        [RISK] ...
        [JUDGE] ...
        """
    elif prompt_key == "stock_report":
        p = f"""
        現在: {today_str}
        銘柄:{context['name']} ({context['ticker']})
        基礎データ:{context['fund_str']}
        市場・セクター比較:{context['m_comp']}
        ニュース:{context['news']}
        次回決算日(取得値): {context.get("earnings_date","-")}。これが'-'でない場合、監視ポイントに必ず含めよ。
        
        プロのアナリストレポートを作成せよ。文体は「だ・である」。
        記号(「**」や「""」)は使用禁止。
        「不明」「わからない」という言葉は禁止。データがない場合はその項目を黙ってスキップせよ。
        
        必ず次の順に出力（見出し固定）：
        1) 定量サマリー（株価/時価総額/リターン/PER/PBR）
        2) バリュエーション：市場平均・セクター平均との比較（割安か？）
        3) 需給/センチメント（直近リターンから逆回転条件）
        4) ニュース/非構造情報（事象→業績→3ヶ月株価ドライバー）
        5) 3ヶ月見通し（ベース/強気/弱気シナリオ）
        6) 監視ポイント（次の決算や金利等）
        """

    # Retry Loop for Temporal Sanity
    for _ in range(2):
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                text = model.generate_content(p).text
                text = clean_ai_text(enforce_da_dearu_soft(text))
                if not temporal_sanity_flags(text):
                    return text
                log_system_event("Temporal sanity failed, retrying...", "WARN", "AI")
            except Exception as e:
                if "429" in str(e): time.sleep(1); continue
    return text

def parse_agent_debate(text: str) -> str:
    mapping = {
        "[SECTOR_OUTLOOK]": ("agent-outlook", "SECTOR OUTLOOK"),
        "[FUNDAMENTAL]": ("agent-fundamental", "FUNDAMENTAL"),
        "[SENTIMENT]": ("agent-sentiment", "SENTIMENT"),
        "[VALUATION]": ("agent-valuation", "VALUATION"),
        "[SKEPTIC]": ("agent-skeptic", "SKEPTIC"),
        "[RISK]": ("agent-risk", "RISK"),
        "[JUDGE]": ("agent-verdict", "JUDGE")
    }
    clean = clean_ai_text(text.replace("```html", "").replace("```", ""))
    parts = re.split(r'(\[[A-Z_]+\])', clean)
    html = ""
    curr_cls, label, buffer = "agent-box", "", ""
    
    for part in parts:
        if part in mapping:
            if buffer and label:
                content = f"<div class='agent-content'>{buffer}</div>"
                if "outlook" in curr_cls:
                    html += f"<div class='{curr_cls}' style='border-left:5px solid #00f2fe; margin-bottom:15px;'><b>{label}</b><br>{content}</div>"
                else:
                    html += f"<div class='agent-row {curr_cls}'><div class='agent-label'>{label}</div>{content}</div>"
            curr_cls, label = mapping[part]
            buffer = ""
        else: buffer += part
    
    # Flush last
    if buffer and label:
        content = f"<div class='agent-content'>{buffer}</div>"
        if "outlook" in curr_cls:
            html += f"<div class='{curr_cls}' style='border-left:5px solid #00f2fe; margin-bottom:15px;'><b>{label}</b><br>{content}</div>"
        else:
            html += f"<div class='agent-row {curr_cls}'><div class='agent-label'>{label}</div>{content}</div>"
    return html

# ==========================================
# 5. MAIN UI LOGIC
# ==========================================
@error_boundary
def main():
    st.markdown("<h1 class='brand'>ALPHALENS</h1>", unsafe_allow_html=True)
    
    # --- LOG CONSOLE (TOP FIXED) ---
    log_short = "\n".join(st.session_state.system_logs[-25:]) if st.session_state.system_logs else "No logs."
    cL1, cL2 = st.columns([0.8, 0.2])
    with cL1:
        st.markdown("<div class='log-top'>" + log_short + "</div>", unsafe_allow_html=True)
    with cL2:
        if st.button("CLEAR LOGS", use_container_width=True):
            st.session_state.system_logs = []
            st.rerun()

    with st.expander("LOG CONSOLE (Mobile)", expanded=False):
        log_long = "\n".join(st.session_state.system_logs[-200:]) if st.session_state.system_logs else "No logs."
        st.markdown("<div class='log-mobile'>" + log_long + "</div>", unsafe_allow_html=True)

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

    # 1. Market Pulse
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
    _, market_context, m_sent, m_meta = get_news_consolidated(bench, m_cfg["name"], market_key)
    
    # Definition Header (ORDER FIXED)
    s_score = clamp(m_sent, -10, 10)
    lbl = sentiment_label(s_score)
    spread = sdf.iloc[-1]['RS'] - sdf.iloc[0]['RS']
    
    st.markdown(f"""
    <div class='market-box'>
    <div class='def-text'>
    <b>DEFINITIONS</b> | 
    <b>Spread</b>: セクターRSの最大−最小(pt)。市場内で勝ち負けがどれだけ鮮明かを示す | 
    <b>Regime</b>: 200DMA判定 (Bull/Bear)。中期トレンドの地合い | 
    <b>NewsSent</b>: 見出しキーワード命中の合計(-10~+10)。短期の需給・期待変化 |
    <b>RS</b>: 相対リターン差(pt)
    </div>
    <b class='orbitron'>MARKET PULSE ({s_date} - {e_date})</b><br>
    <span class='caption-text'>Spread: {spread:.1f}pt | Regime: {regime} | NewsSent: <span class='highlight'>{s_score:+d}</span> ({lbl}) [Hit:{m_meta['pos']}/{m_meta['neg']}]</span><br><br>
    """ + market_to_html(force_nonempty_outlook_market(
        enforce_market_format(generate_ai_content("market", {
            "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
            "top": sdf.iloc[-1]["Sector"], "bot": sdf.iloc[0]["Sector"],
            "market_name": m_cfg["name"], "headlines": market_context,
            "date_slots": outlook_date_slots()
        })), regime, b_stats["Ret"], spread, market_key
    )) + "</div>", unsafe_allow_html=True)

    # 2. Sector Rotation
    st.subheader(f"SECTOR ROTATION ({s_date} - {e_date})")
    
    sdf["Label"] = sdf["Sector"] + " (" + sdf["Ret"].apply(lambda x: f"{x:+.1f}%") + ")"
    
    # DEFAULT SELECTION: Max Return Sector
    if not st.session_state.selected_sector:
        best_row = sdf.loc[sdf["Ret"].idxmax()]
        st.session_state.selected_sector = best_row["Sector"]
        log_system_event(f"Default sector: {st.session_state.selected_sector}", "INFO", "UI")

    colors = ["#333"] * len(sdf)
    if st.session_state.selected_sector in sdf["Sector"].values:
        idx = sdf[sdf["Sector"] == st.session_state.selected_sector].index[0]
        colors[sdf.index.get_loc(idx)] = "#00f2fe"

    fig = px.bar(sdf, x="RS", y="Label", orientation='h', title=f"Relative Strength ({lookback_key})")
    fig.update_traces(customdata=np.stack([sdf["Ret"]], axis=-1), hovertemplate="%{y}<br>Ret: %{customdata[0]:+.1f}%<br>RS: %{x:.2f}<extra></extra>", marker_color=colors)
    fig.update_layout(height=380, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font_color='#e0e0e0', font_family="Orbitron", xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True))
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True, 'displayModeBar': False})
    
    st.write("SELECT SECTOR:")
    cols = st.columns(2)
    for i, row in enumerate(sdf.itertuples()):
        s = row.Sector
        label = f"✅ {s} ({row.Ret:+.1f}%)" if s == st.session_state.selected_sector else f"{s} ({row.Ret:+.1f}%)"
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
    
    df["Apex"] = weight_mom * calculate_zscore(df["RS"]) + (0.8 - weight_mom) * calculate_zscore(df["Accel"]) + 0.2 * calculate_zscore(df["Ret"])
    df = df.sort_values("Apex", ascending=False)
    
    # 4. 5-AGENT SECTOR COUNCIL
    st.markdown("##### 🦅 5-AGENT SECTOR COUNCIL (Top Picks)")
    
    top3 = df.head(3).copy()
    neg = df.sort_values(["RS","MaxDD"], ascending=[True, False]).head(1)
    
    cand_lines = []
    for _, r in top3.iterrows():
        fd = get_fundamental_data(r["Ticker"])
        cand_lines.append(
            f"{r['Name']}({r['Ticker']}): Ret {r['Ret']:.1f}%, RS {r['RS']:.2f}, Accel {r['Accel']:.2f}, HighDist {r['HighDist']:.1f}%, "
            f"MCap {sfloat(fd.get('MCap',0))/1e9:.1f}B, PER {dash(fd.get('PER'))}, PBR {dash(fd.get('PBR'))}"
        )
    if not neg.empty:
        nr = neg.iloc[0]
        cand_lines.append(f"\n[AVOID] {nr['Name']}: Ret {nr['Ret']:.1f}%, RS {nr['RS']:.2f}")

    _, sec_news, _, _ = get_news_consolidated(m_cfg["sectors"][target_sector], target_sector, market_key, limit_each=6)
    
    sec_ai_raw = generate_ai_content("sector_debate", {
        "sec": target_sector, "count": len(df), "candidates": "\n".join(cand_lines),
        "news": sec_news, "market_name": m_cfg["name"]
    })
    st.markdown(parse_agent_debate(sec_ai_raw), unsafe_allow_html=True)
    
    # EVIDENCE TABLE (Always Visible)
    st.markdown("###### EVIDENCE (Top Candidates)")
    st.caption(
        "DEFINITIONS | Apex: zscore合成=weight_mom*z(RS)+(0.8-weight_mom)*z(Accel)+0.2*z(Ret) | "
        "RS: Ret(銘柄)−Ret(市場平均) | Accel: 直近半期間リターン−(全期間リターン/2) | "
        "HighDist: 直近価格の52週高値乖離(%) | MaxDD: 期間内最大ドローダウン(%) | "
        "PER/PBR/ROE等: yfinance.Ticker().info（負のPER/PBRは除外、欠損は'-'）"
    )
    ev_fund = fetch_fundamentals_batch(top3["Ticker"].tolist()).reset_index()
    ev_df = top3.merge(ev_fund, on="Ticker", how="left")
    for c in ["PER","PBR"]: ev_df[c] = ev_df[c].apply(lambda x: dash(x))
    for c in ["ROE","RevGrow","OpMargin"]: ev_df[c] = ev_df[c].apply(pct)
    ev_df["Beta"] = ev_df["Beta"].apply(lambda x: dash(x, "%.2f"))
    
    st.dataframe(ev_df[["Name","Ticker","Apex","RS","Accel","Ret","HighDist","MaxDD","PER","PBR","ROE","RevGrow","OpMargin","Beta"]], hide_index=True, use_container_width=True)

    # 5. Leaderboard
    st.markdown("##### LEADERBOARD")
    st.caption("SOURCE & NOTES | Price: yfinance | Fundamentals: yfinance.Ticker().info | PER/PBR: 負値は除外 | ROE/RevGrow/OpMargin/Beta: 取得できる場合のみ表示")
    
    tickers_for_fund = df.head(30)["Ticker"].tolist()
    with st.spinner("Fetching Fundamentals..."):
        rest = fetch_fundamentals_batch(tickers_for_fund).reset_index()
        df = df.merge(rest, on="Ticker", how="left", suffixes=("", "_rest"))
        for c in ["MCap", "PER", "PBR", "FwdPE", "ROE", "RevGrow", "OpMargin", "Beta"]:
            if c in df.columns and f"{c}_rest" in df.columns:
                df[c] = df[c].fillna(df[f"{c}_rest"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_rest")])

    def fmt_mcap(x):
        if pd.isna(x) or x == 0: return "-"
        if x >= 1e12: return f"{x/1e12:.1f}T"
        if x >= 1e9: return f"{x/1e9:.1f}B"
        return f"{x/1e6:.0f}M"
    
    df["MCapDisp"] = df["MCap"].apply(fmt_mcap)
    
    df_disp = df.copy()
    for c in ["PER", "PBR"]: df_disp[c] = df_disp[c].apply(lambda x: dash(x))
    for c in ["ROE", "RevGrow", "OpMargin"]: df_disp[c] = df_disp[c].apply(pct)
    df_disp["Beta"] = df_disp["Beta"].apply(lambda x: dash(x, "%.2f"))

    df_sorted = df_disp.sort_values("MCap", ascending=False)
    
    event = st.dataframe(
        df_sorted[["Name", "Ticker", "MCapDisp", "ROE", "RevGrow", "PER", "PBR", "Apex", "RS", "1M", "12M"]],
        column_config={
            "Ticker": st.column_config.TextColumn("Code"),
            "MCapDisp": st.column_config.TextColumn("Market Cap"),
            "Apex": st.column_config.NumberColumn(format="%.2f"),
            "RS": st.column_config.NumberColumn("RS (pt)", format="%.2f"),
            "PER": st.column_config.TextColumn("PER"),
            "PBR": st.column_config.TextColumn("PBR"),
            "ROE": st.column_config.TextColumn("ROE"),
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
    st.caption(f"Data Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: yfinance (PER/PBR exclude negatives)")
    
    news_items, news_context, _, _ = get_news_consolidated(top["Ticker"], top["Name"], market_key, limit_each=10)
    fund_data = get_fundamental_data(top["Ticker"])
    ed = fetch_earnings_dates(top["Ticker"]).get("EarningsDate", "-")
    bench_fd = get_fundamental_data(bench)
    
    bench_per = dash(bench_fd.get("PER"))
    sector_per = dash(pd.to_numeric(df["PER"], errors="coerce").median())
    stock_per = dash(fund_data.get("PER"))
    m_comp = f"市場平均PER: {bench_per}倍 / セクター中央値PER: {sector_per}倍 / 当該銘柄PER: {stock_per}倍"
    
    fund_str = f"PER:{stock_per}, PBR:{dash(fund_data.get('PBR'))}, PEG:{dash(fund_data.get('PEG'))}, Target:{dash(fund_data.get('Target'))}"

    report_txt = generate_ai_content("stock_report", {
        "name": top["Name"], "ticker": top["Ticker"],
        "fund_str": fund_str, "m_comp": m_comp, "news": news_context,
        "earnings_date": ed
    })
    
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        st.markdown(f"<div class='report-box'><b>ANALYST REPORT</b><br>{report_txt}</div>", unsafe_allow_html=True)
        st.caption(
            "PEER LOGIC | Nearest Market Cap: |MCap(peer)−MCap(target)|が小さい順に抽出（同一セクター内） | "
            "SOURCE: yfinance.Ticker().info（欠損は'-'）"
        )
        try:
            target_mcap = top["MCap"] if pd.notna(top["MCap"]) else 0
            df_sorted["Dist"] = (df_sorted["MCap"] - target_mcap).abs()
            df_peers = df_sorted.sort_values("Dist").iloc[1:5]
            st.dataframe(df_peers[["Name", "ROE", "RevGrow", "PER", "PBR", "RS", "12M"]], hide_index=True)
        except: pass

    with nc2:
        st.caption("INTEGRATED NEWS FEED")
        for n in news_items[:20]:
            dt = datetime.fromtimestamp(n["pub"]).strftime("%Y/%m/%d") if n["pub"] else "-"
            st.markdown(f"- {dt} [{n['src']}] [{n['title']}]({n['link']})")

if __name__ == "__main__":
    main()