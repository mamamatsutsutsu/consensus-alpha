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

# Import Universe
try:
    import universe
except ImportError:
    st.error("CRITICAL: 'universe.py' not found.")
    st.stop()

# --- UTILS ---
def log_system_event(msg: str, level: str = "INFO", tag: str = "SYS"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] [{level}] [{tag}] {msg}"
    print(line)
    if "system_logs" in st.session_state:
        st.session_state.system_logs.append(line)
        st.session_state.system_logs = st.session_state.system_logs[-300:]

MARKETS = universe.MARKETS
NAME_DB = universe.NAME_DB
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

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
    if score >= 3: return "POS"
    if score <= -3: return "NEG"
    return "NEUT"

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

def safe_link_button(label: str, url: str, use_container_width: bool = True):
    if not url:
        st.button(label, disabled=True, use_container_width=use_container_width)
        return
    try:
        fn = getattr(st, "link_button", None)
        if callable(fn):
            fn(label, url, use_container_width=use_container_width)
        else:
            st.markdown(f"- [{label}]({url})")
    except Exception:
        st.markdown(f"- [{label}]({url})")

def build_ir_links(name: str, ticker: str, website: Optional[str], market_key: str) -> Dict[str, str]:
    q_site = urllib.parse.quote(name)
    q_ir = urllib.parse.quote(f"{name} IR")
    if "US" in market_key:
        q_deck = urllib.parse.quote(f"{name} investor presentation earnings pdf")
    else:
        q_deck = urllib.parse.quote(f"{name} 決算説明資料 pdf")
            
    official = website if website and website.startswith("http") else f"https://www.google.com/search?q={q_site}+official+site"
    
    return {
        "official": official,
        "ir_search": f"https://www.google.com/search?q={q_ir}",
        "earnings_deck": f"https://www.google.com/search?q={q_deck}",
    }

# --- DATA FETCHING ---
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

def price_action_pack(price: pd.Series) -> Dict[str, float]:
    p = price.dropna()
    if len(p) < 60: return {}
    out = {}
    out["Last"] = float(p.iloc[-1])
    try:
        out["1D"] = float((p.iloc[-1] / p.iloc[-2] - 1) * 100) if len(p) >= 2 else np.nan
        out["1W"] = float((p.iloc[-1] / p.iloc[-6] - 1) * 100) if len(p) >= 6 else np.nan
        out["1M"] = float((p.iloc[-1] / p.iloc[-22] - 1) * 100) if len(p) >= 22 else np.nan
        out["3M"] = float((p.iloc[-1] / p.iloc[-64] - 1) * 100) if len(p) >= 64 else np.nan
        ma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
        out["200DMA_Dist"] = float((p.iloc[-1] / ma200 - 1) * 100) if pd.notna(ma200) and ma200 != 0 else np.nan
        dd = (p / p.cummax() - 1) * 100
        out["MaxDD_6M"] = float(dd.tail(126).min()) if len(dd) >= 126 else float(dd.min())
    except: pass
    return out

# --- FUNDAMENTALS ---
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
    """Fundamentals snapshot (best-effort).
    yfinance.info can be empty; fallback to fast_info where possible.
    Always returns display-safe keys used by the UI.
    """
    out: Dict[str, Any] = {
        "Name": ticker,
        "Sector": "-",
        "Industry": "-",
        "MCap": np.nan,
        "PER": np.nan,
        "FwdPE": np.nan,
        "PBR": np.nan,
        "PEG": np.nan,
        "ROE": np.nan,
        "RevGrow": np.nan,
        "OpMargin": np.nan,
        "Beta": np.nan,
        "Website": None,
        "Summary": "-",
        "Currency": None,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        fi = {}
        try:
            fi = getattr(t, "fast_info", {}) or {}
        except Exception:
            fi = {}

        name = info.get("shortName") or info.get("longName") or out["Name"]
        if isinstance(name, str) and name.strip():
            out["Name"] = name.strip()

        out["Sector"] = info.get("sector") or out["Sector"]
        out["Industry"] = info.get("industry") or out["Industry"]

        mcap = info.get("marketCap") or fi.get("market_cap")
        if isinstance(mcap, (int, float)) and mcap > 0:
            out["MCap"] = mcap

        pe = info.get("trailingPE", np.nan)
        if isinstance(pe, (int, float)) and pe < 0:
            pe = np.nan
        out["PER"] = pe

        fpe = info.get("forwardPE", np.nan)
        if isinstance(fpe, (int, float)) and fpe < 0:
            fpe = np.nan
        out["FwdPE"] = fpe

        pbr = info.get("priceToBook", np.nan)
        if isinstance(pbr, (int, float)) and pbr < 0:
            pbr = np.nan
        out["PBR"] = pbr

        out["PEG"] = info.get("pegRatio", np.nan)
        out["ROE"] = info.get("returnOnEquity", np.nan)
        out["RevGrow"] = info.get("revenueGrowth", np.nan)
        out["OpMargin"] = info.get("operatingMargins", np.nan)
        out["Beta"] = info.get("beta", np.nan)
        out["Website"] = info.get("website", None)
        out["Summary"] = info.get("longBusinessSummary") or info.get("businessSummary") or out["Summary"]
        out["Currency"] = info.get("currency") or fi.get("currency") or out["Currency"]
    except Exception:
        pass
    return out


def pick_fund_row(cand_fund: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    try:
        if cand_fund is None or cand_fund.empty: return {}
        m = cand_fund[cand_fund["Ticker"] == ticker]
        if m.empty: return {}
        return m.iloc[0].to_dict()
    except: return {}

@st.cache_data(ttl=3600)
def fetch_earnings_dates(ticker: str) -> Dict[str,str]:
    out = {}
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None:
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                out["EarningsDate"] = str(cal['Earnings Date'][0])
            elif isinstance(cal, pd.DataFrame):
                 for k in ["Earnings Date", "EarningsDate"]:
                    if k in cal.index:
                        v = cal.loc[k].values
                        out["EarningsDate"] = ", ".join([str(x)[:10] for x in v if str(x) != "nan"])
    except: pass
    return out


# --- Company Profile & AI Overview ---

@st.cache_data(ttl=24*3600)
def get_company_profile(ticker: str) -> Dict[str, Any]:
    """Return a stable subset of yfinance profile fields. Falls back to fast_info when possible."""
    out: Dict[str, Any] = {}
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        fast = {}
        try:
            fast = getattr(tk, "fast_info", {}) or {}
        except Exception:
            fast = {}

        def _pick(*keys):
            for k in keys:
                v = info.get(k)
                if v not in (None, "", "nan"):
                    return v
            return None

        out["Name"] = _pick("longName", "shortName") or fast.get("shortName") or ticker
        out["Sector"] = _pick("sector") or "-"
        out["Industry"] = _pick("industry") or "-"
        out["Country"] = _pick("country") or "-"
        out["Website"] = _pick("website") or None
        out["MarketCap"] = _pick("marketCap") or fast.get("market_cap") or fast.get("marketCap") or None
        out["BusinessSummary"] = (_pick("longBusinessSummary") or "").strip()
    except Exception:
        out = {"Name": ticker, "Sector": "-", "Industry": "-", "Country": "-", "Website": None, "MarketCap": None, "BusinessSummary": ""}
    return out

@st.cache_data(ttl=24*3600, show_spinner=False)
def ai_company_summary_cached(name: str, facts: Dict[str, Any], nonce: int = 0) -> str:
    """Generate a conservative company summary from FACTS only. Cached for speed."""
    if not HAS_LIB or not API_KEY:
        return ""
    try:
        m = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "あなたは金融機関向け商用アプリの編集者。\n"
            "以下のFACTS以外の新しい事実を追加してはいけない。推測で固有名詞（製品名/顧客/競合/地域）を増やさない。\n"
            "日本語で3〜5文、簡潔に会社の事業概要を要約せよ。断定を避け『〜とみられる/可能性』程度に留める。\n"
            "FACTSが乏しい場合は、Sector/Industry/Website/MarketCapの範囲で一般的な表現に留める（具体例を作らない）。\n"
            "出力は本文のみ。\n"
            f"FACTS: {facts}"
        )
        txt = m.generate_content(prompt).text or ""
        return clean_ai_text(txt)
    except Exception:
        return ""

def build_company_overview(profile: Dict[str, Any], enable_ai: bool, nonce: int = 0) -> Dict[str, Any]:
    name = str(profile.get("Name") or "-")
    sector = str(profile.get("Sector") or "-")
    industry = str(profile.get("Industry") or "-")
    country = str(profile.get("Country") or "-")
    website = profile.get("Website") or None
    mcap = profile.get("MarketCap")
    mcap_disp = dash(mcap, "%.0f")
    if isinstance(mcap, (int, float)) and mcap:
        if mcap >= 1e12:
            mcap_disp = f"{mcap/1e12:.1f}T"
        elif mcap >= 1e9:
            mcap_disp = f"{mcap/1e9:.1f}B"
        elif mcap >= 1e6:
            mcap_disp = f"{mcap/1e6:.0f}M"

    summary = str(profile.get("BusinessSummary") or "").strip()
    # If yfinance has no summary, ask AI to produce a conservative one from facts only
    if enable_ai:
        facts = {
            "name": name,
            "sector": sector,
            "industry": industry,
            "country": country,
            "website": website or "-",
            "marketCap": mcap_disp,
            "yfinance_summary": (summary[:800] if summary else "-"),
        }
        ai_sum = ai_company_summary_cached(name, facts, nonce=nonce)
        if ai_sum:
            summary = ai_sum

    if not summary:
        summary = "-"

    overview_html = f"Sector:{sector} | Industry:{industry} | MCap:{mcap_disp} | Country:{country}"
    overview_plain = f"Name:{name}\nSector:{sector}\nIndustry:{industry}\nMCap:{mcap_disp}\nCountry:{country}\nWebsite:{website or '-'}\nSummary:{summary}"
    return {
        "name": name, "sector": sector, "industry": industry, "country": country,
        "website": website, "mcap": mcap, "mcap_disp": mcap_disp,
        "summary": summary, "overview_html": overview_html, "overview_plain": overview_plain
    }

# --- AI & TEXT ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
try:
    import google.generativeai as genai
    HAS_LIB = True
    if API_KEY: genai.configure(api_key=API_KEY)
except: HAS_LIB = False

def clean_ai_text(text: str) -> str:
    text = text.replace("```text", "").replace("```", "")
    text = text.replace("**", "").replace('"', "").replace("'", "")
    text = re.sub(r"(?m)^\s*text\s*$", "", text)
    text = re.sub(r"(?m)^\s*#{2,}\s*", "", text)
    text = re.sub(r"(?im)^\s*(agent|エージェント)\s*[A-E0-9]+[:：]\s*", "", text)
    # strip polite / meta preambles
    text = re.sub(r"(?m)^\s*(はい、)?\s*承知(いたしました|しました)。?.*$\n?", "", text)
    text = re.sub(r"(?m)^\s*以下に(.*)作成(する|します)。?.*$\n?", "", text)
    text = re.sub(r"(?m)^\s*ご依頼(.*)ありがとうございます。?.*$\n?", "", text)
    bad = ["不明", "わからない", "分からない", "unknown"]
    for w in bad: text = re.sub(rf"(?m)^.*{re.escape(w)}.*$\n?", "", text)
    # remove stray backreference artifacts like \\1 or ASCII SOH
    text = text.replace('\\1', '').replace('\x01', '')
    # remove any leading backslash-number artifacts (e.g., \1, \2) that may leak from regex groups
    text = re.sub(r'(?m)^\s*\\\d+\s*', '', text)
    text = text.replace('\u0001', '')
    text = re.sub(r'(?m)^\s*\\1', '', text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def quality_gate_text(text: str, enable: bool = True) -> str:
    """Lightweight, safe post-processor for AI text before rendering.
    - Removes meta/preambles
    - Removes stray artifacts (e.g., \1)
    - Softens overconfident claims
    - Keeps structure but does NOT add any new facts
    """
    t = clean_ai_text(text)
    # soften absolutes (JP/EN)
    t = re.sub(r"(必ず|確実に|間違いなく|断言できる)", "可能性が高い", t)
    t = re.sub(r"\b(guaranteed|certainly|definitely|undoubtedly)\b", "likely", t, flags=re.I)
    # remove empty lines
    t = re.sub(r"\n{2,}", "\n", t).strip()
    return t


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
    """Normalize Market Pulse text to required sections; resilient to messy LLM outputs."""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove common assistant boilerplate/meta
    text = re.sub(r"(?im)^\s*(はい、)?\s*承知(いた)?しました[。!！]*.*\n+", "", text)
    text = re.sub(r"(?im)^\s*以下に.*(作成|生成).*(します|いたします)[。!！]*\s*$", "", text)
    # Remove '予定日:' label if present in outlook bullets
    text = re.sub(r"(?im)^\s*-\s*予定日\s*[:：]\s*", "- ", text)

    # Remove unwanted date suffix right after the outlook header

    # Replace placeholder event names (EventA/B/C...) with meaningful labels (best-effort)
    def _event_label(reason: str) -> str:
        r = reason
        if re.search(r"(CPI|インフレ|物価|PCE)", r, re.I): return "Inflation data"
        if re.search(r"(雇用|Payroll|失業|NFP)", r, re.I): return "Jobs data"
        if re.search(r"(FOMC|FRB|Fed|利上げ|利下げ|金融政策)", r, re.I): return "Central bank"
        if re.search(r"(決算|earnings)", r, re.I): return "Earnings"
        if re.search(r"(地政学|中東|台湾|ウクライナ|紛争)", r, re.I): return "Geopolitics"
        if re.search(r"(金利|長期金利|利回り|bond)", r, re.I): return "Rates"
        if re.search(r"(原油|OPEC)", r, re.I): return "Oil supply"
        return "Macro catalyst"

    def _rename_event_lines(t: str) -> str:
        # Pattern: - イベントA(2026-03-01)→...→理由
        out_lines = []
        for ln in t.splitlines():
            m = re.match(r"^\s*-\s*(イベント|Event)\s*([A-F])\s*\(([^)]+)\)\s*→\s*(.*)$", ln)
            if m:
                date = m.group(3)
                rest = m.group(4)
                label = _event_label(rest)
                ln = f"- {label} ({date})→{rest}"
            out_lines.append(ln)
        return "\n".join(out_lines)

    text = re.sub(r"(【今後3ヶ月[^】]*】)\s*\(\d{4}[-/]\d{2}[-/]\d{2}\)", r"\1", text)
    text = re.sub(r"(【今後3ヶ月[^】]*】)\s*\d{4}[-/]\d{2}[-/]\d{2}", r"\1", text)

    # Remove standalone date line immediately following the outlook header
    text = re.sub(r"(【今後3ヶ月[^】]*】)\n\s*\d{4}[-/]\d{2}[-/]\d{2}\s*\n", r"\1\n", text)

    # Ensure required headers exist
    if "【市場概況】" not in text:
        text = "【市場概況】\n" + text

    if "【主な変動要因】" not in text:
        text += "\n\n【主な変動要因】\n(+) 上昇要因:\n(-) 下落要因:"

    if "【今後3ヶ月" not in text:
        text += "\n\n【今後3ヶ月のコンセンサス見通し】\n"

    text = _rename_event_lines(text)

    return text

def enforce_index_naming(text: str, index_label: str) -> str:
    if not index_label:
        return text
    # Replace vague wording with explicit index label
    text = re.sub(r"市場平均(リターン)?", index_label, text)
    text = re.sub(r"指数(?:全体)?", index_label, text)
    # Ensure the index label appears at least once in the market overview
    if index_label not in text and "【市場概況】" in text:
        text = re.sub(r"(【市場概況】\n?)", rf"\1{index_label}を基準に記述する。\n", text, count=1)
    return text

def group_plus_minus_blocks(text: str) -> str:
    # Extract the block
    m = re.search(r"【主な変動要因】\n?(.*?)(?=\n【|\Z)", text, flags=re.DOTALL)
    if not m:
        return text
    block = m.group(1).strip()
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    # remove fake headings that often get bulletized
    heading_trash = {"上昇要因:", "下落要因:", "(+) 上昇要因:", "(-) 下落要因:", "（+）上昇要因:", "（−）下落要因:"}
    cleaned = []
    for l in lines:
        l2 = l.lstrip("-・ ").strip()
        if l2 in heading_trash:
            continue
        # remove "イベントA" etc accidentally placed here
        if l2.startswith("3)") or "今後3ヶ月" in l2:
            continue
        cleaned.append(l)

    pos, neg, oth = [], [], []
    pos_kw = ["上方修正","増益","好調","回復","低下","鈍化","利下げ","利回り低下","金利低下","緩和","買い","資金流入","強い","上昇","改善","割安","自社株買い","需要増","受注増","インフレ低下","ソフトインフレ","景気後退懸念後退"]
    neg_kw = ["下方修正","減益","悪化","失速","再加速","利上げ","引き締め","タカ派","売り","資金流出","下落","警戒","高止まり","リスク","地政学","長期金利上昇","金利上昇","利回り上昇","ボラティリティ","懸念","警告シグナル","テック売り","リプライシング"]

    for l in cleaned:
        raw = l.lstrip("-・ ").strip()
        # explicit sign markers
        if raw.startswith("(+)") or raw.startswith("＋") or raw.startswith("+"):
            pos.append(raw.lstrip("()+＋+ ").strip())
            continue
        if raw.startswith("(-)") or raw.startswith("−") or raw.startswith("-"):
            neg.append(raw.lstrip("()-−- ").strip())
            continue
        # keyword routing
        score = 0
        if any(k in raw for k in pos_kw): score += 1
        if any(k in raw for k in neg_kw): score -= 1
        if score > 0:
            pos.append(raw)
        elif score < 0:
            neg.append(raw)
        else:
            oth.append(raw)

    # Build normalized section
    def bullets(arr):
        return "\n".join([f"- {x}" for x in arr[:6]]) if arr else "- （該当材料を抽出できず）"
    out = "【主な変動要因】\n(+) 上昇要因:\n" + bullets(pos) + "\n(−) 下落要因:\n" + bullets(neg)
    if oth:
        out += "\n(補足):\n" + "\n".join([f"- {x}" for x in oth[:6]])
    # Replace original block
    return text[:m.start()] + out + text[m.end():]
def enforce_da_dearu_soft(text: str) -> str:
    text = re.sub(r"です。", "だ。", text)
    text = re.sub(r"です$", "だ", text, flags=re.MULTILINE)
    text = re.sub(r"ます。", "する。", text)
    text = re.sub(r"ます$", "する", text, flags=re.MULTILINE)
    return text

def market_to_html(text: str) -> str:
    text = re.sub(r"(^\(\+\s*\).*$)", r"<span class='hl-pos'>\1</span>", text, flags=re.MULTILINE)
    text = re.sub(r"(^\(\-\s*\).*$)", r"<span class='hl-neg'>\1</span>", text, flags=re.MULTILINE)
    return text.replace("\n", "<br>")

@st.cache_data(ttl=1800)
def get_news_consolidated(ticker: str, name: str, market_key: str, limit_each: int = 10) -> Tuple[List[dict], str, int, Dict[str,int]]:
    news_items, context_lines = [], []
    pos_words = ["増益", "最高値", "好感", "上昇", "自社株買い", "上方修正", "急騰", "beat", "high", "jump", "record"]
    neg_words = ["減益", "安値", "嫌気", "下落", "下方修正", "急落", "赤字", "miss", "low", "drop", "warn"]
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
                dt = datetime.fromtimestamp(p).strftime("%Y/%m/%d") if p else "-"
                weight = 2 if (time.time() - p) < 172800 else 1
                context_lines.append(f"- [Yahoo {dt}] {t}")
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
                    dt = datetime.fromtimestamp(pub).strftime("%Y/%m/%d") if pub else "-"
                    weight = 2 if (time.time() - pub) < 172800 else 1
                    context_lines.append(f"- [Google {dt}] {t}")
                    if any(w in t for w in pos_words): sentiment_score += 1*weight; meta["pos"] += 1
                    if any(w in t for w in neg_words): sentiment_score -= 1*weight; meta["neg"] += 1
    except: pass

    # Free public RSS feeds (fallback / enrichment). English-only is OK.
    try:
        rss_sources = [
            ("Reuters Markets", "https://feeds.reuters.com/reuters/marketsNews"),
            ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
            ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
            ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
            ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ]
        for src, url2 in rss_sources:
            try:
                with urllib.request.urlopen(url2, timeout=3) as r:
                    root = ET.fromstring(r.read())
                    for it in root.findall('.//item')[: max(3, limit_each//3) ]:
                        t2, l2, d2 = it.findtext('title'), it.findtext('link'), it.findtext('pubDate')
                        try: pub2 = int(email.utils.parsedate_to_datetime(d2).timestamp())
                        except: pub2 = 0
                        if not t2: continue
                        news_items.append({"title": t2, "link": l2, "pub": pub2, "src": src})
                        dt2 = datetime.fromtimestamp(pub2).strftime('%Y/%m/%d') if pub2 else "-"
                        weight = 2 if (pub2 and (time.time() - pub2) < 172800) else 1
                        context_lines.append(f"- [{src} {dt2}] {t2}")
                        if any(w in t2 for w in pos_words): sentiment_score += 1*weight; meta["pos"] += 1
                        if any(w in t2 for w in neg_words): sentiment_score -= 1*weight; meta["neg"] += 1
            except Exception:
                pass
    except Exception:
        pass


    news_items.sort(key=lambda x: x["pub"], reverse=True)
    return news_items, "\n".join(context_lines[:15]), sentiment_score, meta

def temporal_sanity_flags(text: str) -> List[str]:
    bad = ["年末年始", "クリスマス", "夏休み", "お盆", "来年", "昨年末"]
    return [w for w in bad if w in text]

def sector_debate_quality_ok(text: str) -> bool:
    needed = ["[SECTOR_OUTLOOK]", "[FUNDAMENTAL]", "[SENTIMENT]", "[VALUATION]", "[SKEPTIC]", "[RISK]", "[JUDGE]"]
    if any(t not in text for t in needed): return False
    min_chars = {
        "[SECTOR_OUTLOOK]": 220, "[FUNDAMENTAL]": 260, "[SENTIMENT]": 260,
        "[VALUATION]": 220, "[SKEPTIC]": 220, "[RISK]": 220, "[JUDGE]": 520,
    }
    for k, mn in min_chars.items():
        m = re.search(re.escape(k) + r"(.*?)(?=\n\[[A-Z_]+\]|\Z)", text, flags=re.DOTALL)
        if not m or len(re.sub(r"\s+", "", m.group(1))) < mn: return False
    if re.search(r"(?im)(私はエージェント|僕はエージェント|俺はエージェント|エージェント[A-E])", text): return False
    return True

@st.cache_data(ttl=3600)
def generate_ai_content(prompt_key: str, context: Dict) -> str:
    if not HAS_LIB or not API_KEY: return "AI OFFLINE"
    
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    p = ""
    market_n = context.get('market_name', 'Global')
    today_str = datetime.now().strftime('%Y年%m月%d日')
    # slot_line: candidate dates for the next 3 months (used in market prompt)
    slot_line = context.get("slot_line")
    if not slot_line:
        # fallback: today + 7d steps (within 90 days)
        base = datetime.now().date()
        slots = [base + timedelta(days=d) for d in [7,14,21,28,35,42,49,56,63,70,77,84]]
        slot_line = ", ".join([s.strftime("%Y-%m-%d") for s in slots])
    
    
    if prompt_key == "market":
        p = f"""
        現在: {today_str} (この日付を基準に分析せよ)
        対象市場: {market_n} (これ以外の市場の話は禁止)
        対象指数: {context.get('index_label','')}（この指数名を必ず本文に明記せよ。「市場平均」という語は禁止）
        期間:{context['s_date']}〜{context['e_date']}
        対象指数リターン:{context['ret']:.2f}%
        最強:{context['top']} 最弱:{context['bot']}
        ニュース:{context['headlines']}
        Nonce:{context.get('nonce',0)}
        
        この期間の{market_n}市場概況をプロ向けに450-650字で記述せよ。
        禁止: 「市場平均」「一般論」「様子見」「不透明」「注視」などの抽象語。
        段落間の空行禁止。改行は許可するが連続改行禁止。
        
        必ず次の順番で出力せよ（見出しは固定）：
        1) 【市場概況】（文章で記述。箇条書き禁止。材料→結果を因果で、数値必須。指数名={context.get('index_label','')}を本文に必ず入れる）
        2) 【主な変動要因】（文章でよい。上昇要因と下落要因をそれぞれ具体に書く。片方しか無い場合はある方だけでよいが、可能な限り両方を書く。見出し語は「上昇要因:」「下落要因:」を各1回だけ使い、その後は文章で続ける）
        3) 【今後3ヶ月のコンセンサス見通し】
        - 日付は次の候補日から選んで書け（本文に「予定日」という語は使うな）：{slot_line}
        - 90日以内に起きやすい具体イベント/予定を最大6つ列挙（日付も想定せよ）
        - 各行は「イベント名(時期)→株価に効きやすい方向→理由」
        - 最後に強気/弱気の条件分岐
        - この期間から外れる季節表現（年末年始、来年など）は禁止
        """
    elif prompt_key == "sector_debate_fast":
        p = f"""
        現在: {today_str}
        あなたは5名の専門エージェントが合議して投資推奨を出す。対象市場は{market_n}。
        対象セクター:{context["sec"]}
        セクター統計:{context.get("sector_stats","")}
        候補（定量/モメンタム中心。TopPick候補の材料）:
        {context.get("top","")}
        ニュース（直近優先。根拠として最低2本引用）:
        {context.get("news","")}
        Nonce:{context.get("nonce",0)}

        厳守:
        - 文体は「だ・である」。自己紹介、承知しました等の前置きは禁止。
        - 3ヶ月で最も上がる確度が高いトップピックは1銘柄のみ。ティッカーを必ず明記。
        - 重視順: 直近ニュース/株価モメンタム(1M/3M/RS) ＞ リスク(最大DD/高値乖離) ＞ バリュエーション。
        - 抽象語（不透明、堅調、注視、様子見）禁止。数値と因果で書く。
        - 各タグは短くてもよいが「論点の役割」を崩すな。

        出力フォーマット（タグ厳守。全体で900〜1400字目安）:
        [SECTOR_OUTLOOK] セクター全体の3ヶ月見通し（3〜5文）
        [FUNDAMENTAL] 形式厳守：最初に「Sector view: ...」(1〜2文)→次に「Stock pick: <TICKER> ...」(3〜5文)
        [SENTIMENT] 形式厳守：Sector view→Stock pick（ニュース根拠2本以上。数値と因果）
        [VALUATION] 形式厳守：Sector view→Stock pick（PER/PBR等が使える場合のみ。使えない場合は触れない）
        [SKEPTIC] 形式厳守：Sector view→Stock pick（反対意見。何が外れるとダメか）
        [RISK] 形式厳守：最初に「Sector view: ...」(1〜2文)→次に「Stock pick: <TICKER> ...」(2〜4文)→最後にトリガー箇条書き3つ（上昇シナリオを壊す要因）
        [JUDGE] タイトルは本文で「TOP PICK JUDGE」と明記。形式厳守：Sector view→Stock pick（トップピック1銘柄のみ、ティッカー必須）→なぜ他候補ではないか（2点）→次に見るべき指標1つ
"""
    elif prompt_key == "sector_debate":
        p = f"""
        現在: {today_str}
        あなたは5名の専門エージェント。対象市場は{market_n}。
        対象セクター:{context['sec']}
        期間:{context.get('s_date','-')}〜{context.get('e_date','-')}
        セクター統計（必ず参照し、過去推移に触れる）:
        {context.get('sector_stats','')}
        候補データ（必ず比較で使う）:
        {context['candidates']}
        ニュース（非構造。最低2本は本文で引用し、根拠化）:
        {context.get('news','')}
        Nonce:{context.get('nonce',0)}

        厳守ルール:
        - 文体は「だ・である」。です・ます調は禁止。
        - 抽象語（不透明、堅調、注視、様子見）は禁止。必ず「何が→どう効く→価格/需給にどう反映」を書く。
        - 定量の優先順位は「モメンタム/センチメント＞バリュエーション＞ファンダ」である。
        - 事実追加は禁止。与えた候補データ/セクター統計/ニュースの範囲で推論せよ。

        タスク:
        1) 冒頭に[SECTOR_OUTLOOK]で、以下を必ず含めてセクターのこれまでの動向と見通し（今後3ヶ月）を自然文で書け:
           - 指定期間のセクターの値動き/モメンタムの特徴（加速/減速など）
           - セクター内の個別銘柄の強弱（上位/下位の特徴を最低2つ言及）
           - 今後3ヶ月のシナリオ（上昇/下落それぞれ1つ以上の具体要因）
        2) その後、各エージェントは必ず次の順で書け（改行で区切る）:
           Sector view: 1〜2文でセクター見通し（[SECTOR_OUTLOOK]と矛盾させない）
           Stock pick: 候補から1銘柄を推奨（ティッカー/短い結論）
           Rationale: 定量（RS/Accel/Ret/HighDist/MaxDD等）＋ニュース根拠で説明
           Risks: 具体的なリスクと否定条件を2つ
        3) [JUDGE]は「TOP PICK JUDGE」として、最終トップピック1銘柄だけを決める。
           - ここでは"Sector view:"という見出しは使わない（重複回避）。
           - 最初に結論（Top pick: <ticker>）を書き、その後に根拠（定量＋ニュース）と否定条件を簡潔にまとめる。

        出力フォーマット（タグ厳守）:
        [SECTOR_OUTLOOK] ...
        [JUDGE] ...
        [FUNDAMENTAL] ...
        [SENTIMENT] ...
        [VALUATION] ...
        [SKEPTIC] ...
        [RISK] ...
        """
    elif prompt_key == "sector_report":
        p = f"""
        現在: {today_str}
        対象市場: {market_n}
        対象セクター: {context['sec']}
        期間:{context['s_date']}〜{context['e_date']}
        セクター統計: {context.get('sector_stats','')}
        上位候補(定量): {context['candidates']}
        セクター関連ニュース: {context.get('news','')}
        Nonce:{context.get('nonce',0)}
        ルール:
        - 文体は「だ・である」。自己紹介禁止。
        - 構成は必ず「セクター全体→個別銘柄（上位3）→リスク→3ヶ月の監視ポイント」。
        - 抽象語禁止。数値を必ず入れる（RS/Accel/Ret/HighDist/MaxDDなど）。
        出力見出し（固定）：
        【セクター概況】
        【上位3銘柄の見立て】
        【想定リスク】
        【今後3ヶ月の監視ポイント】
        """
    elif prompt_key == "stock_report":
        p = f"""
        現在: {today_str}
        銘柄:{context['name']} ({context['ticker']})
        企業概要:{context.get('overview','')}
        基礎データ:{context['fund_str']}
        市場・セクター比較:{context['m_comp']}
        株価動向:{context.get('price_action','')}
        ニュース:{context['news']}
        次回決算日(取得値): {context.get("earnings_date","-")}。これが'-'でない場合、監視ポイントに必ず含めよ。
        Nonce:{context.get('nonce',0)}
        
        あなたはAIエージェントとして、プロ向けの企業分析メモを作成せよ。
        文体は「だ・である」。
        記号(「**」や「""」)は使用禁止。
        「不明」「わからない」という言葉は禁止。データがない場合は言及しない。
        株価動向とニュースは必ず因果で結び、材料→期待→株価の順で説明せよ。
        分量: 900-1400字程度。冗長な言い換え禁止。各段落は新情報/新しい推論のみ。
        
        必ず次の順に出力（見出し固定）：
        1) 定量サマリー（株価動向/バリュエーション/リターン）
        2) バリュエーション評価（市場平均・セクター平均との乖離）
        3) 需給/センチメント（直近リターンから逆回転条件）
        4) ニュース/非構造情報（事象→業績→3ヶ月株価ドライバー）
        5) 3ヶ月見通し（ベース/強気/弱気シナリオ）
        6) 監視ポイント（この銘柄に固有のKPI/イベント/競合/規制/価格指標に紐づける。一般論禁止。次の決算日が取れている場合は必ず含める）
        """

    attempts = 3 if prompt_key == "sector_debate" else (1 if prompt_key == "sector_debate_fast" else 2)
    last_text = ""
    for a in range(attempts):
        extra = ""
        if prompt_key == "sector_debate" and a >= 1:
            extra = "\n\n重要: 前回出力が短すぎ/ルール違反だった。各タグの分量を1.6倍に増やし、必ず「セクター全体→個別銘柄」の順で書け。抽象語禁止。"
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                text = model.generate_content(p + extra).text
                text = clean_ai_text(enforce_da_dearu_soft(text))
                last_text = text
                if temporal_sanity_flags(text):
                    continue
                if prompt_key == "sector_debate":
                    if sector_debate_quality_ok(text):
                        return text
                    else:
                        continue
                return text
            except Exception as e:
                if "429" in str(e): time.sleep(1); continue
    return last_text or "AI OFFLINE"


def plot_relative_1y(ticker: str, sector_etf: str, bench: str, market_key: str):
    """1Y normalized price comparison: stock vs sector ETF vs benchmark."""
    try:
        tickers = [t for t in [ticker, sector_etf, bench] if t]
        df = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        df = df.dropna(how="all")
        if df.empty:
            st.info("Price comparison chart unavailable (no data).")
            return
        # keep columns we have
        cols = [c for c in tickers if c in df.columns]
        if len(cols) < 2:
            st.info("Price comparison chart unavailable (insufficient series).")
            return
        sdf = df[cols].dropna()
        if len(sdf) < 10:
            st.info("Price comparison chart unavailable (insufficient history).")
            return
        base = sdf.iloc[0]
        norm = (sdf / base) * 100.0
        norm = norm.reset_index().melt(id_vars=[norm.columns[0]], var_name="Series", value_name="Index")
        # rename date column robustly
        date_col = norm.columns[0]
        fig = px.line(norm, x=date_col, y="Index", color="Series", title="1Y Relative Performance (Normalized=100)")
        fig.update_layout(height=260, margin=dict(l=10,r=10,t=45,b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Price comparison chart unavailable.")


def _compress_bullets_in_views(content: str) -> str:
    """Flatten bullet lines into compact prose inside 'Sector view:' / 'Stock pick:' blocks."""
    try:
        lines = content.splitlines()
        norm_lines = []
        for ln in lines:
            ln2 = re.sub(r'^[\s\t]*[-•]+\s*', '', ln).strip()
            norm_lines.append(ln2)
        content2 = "\n".join(norm_lines)

        chunks = re.split(r'(?mi)^(Sector view:|Stock pick:)', content2)
        if len(chunks) <= 1:
            return " ".join([x for x in norm_lines if x]).strip()

        out = []
        i = 0
        while i < len(chunks):
            part = chunks[i]
            if part.lower().startswith("sector view:") or part.lower().startswith("stock pick:"):
                heading = part.strip()
                body = (chunks[i+1] if i+1 < len(chunks) else "").strip()
                body_lines = [b.strip() for b in body.splitlines() if b.strip()]
                body_lines = [re.sub(r'(?mi)^(Triggers?|トリガー)\s*[:：]\s*', '', b).strip() for b in body_lines]
                body_txt = " ".join([b for b in body_lines if b])
                out.append(f"{heading} {body_txt}".strip())
                i += 2
            else:
                if part.strip():
                    out.append(part.strip())
                i += 1
        return "\n".join(out).strip()
    except Exception:
        return content


def parse_agent_debate(text: str) -> str:
    """Parse tagged multi-agent debate and render in a fixed, pro layout.
    Always enforce: SECTOR_OUTLOOK -> (agents...) -> JUDGE (styled).
    """
    mapping = {
        "[SECTOR_OUTLOOK]": ("agent-outlook", "SECTOR OUTLOOK"),
        "[FUNDAMENTAL]": ("agent-fundamental", "FUNDAMENTAL"),
        "[SENTIMENT]": ("agent-sentiment", "SENTIMENT"),
        "[VALUATION]": ("agent-valuation", "VALUATION"),
        "[SKEPTIC]": ("agent-skeptic", "SKEPTIC"),
        "[RISK]": ("agent-risk", "RISK"),
        "[JUDGE]": ("agent-verdict", "TOP PICK JUDGE"),
        "[TOP_PICK_JUDGE]": ("agent-verdict", "TOP PICK JUDGE"),
        "[TOPPICK_JUDGE]": ("agent-verdict", "TOP PICK JUDGE"),
        "[TOP_PICK]": ("agent-verdict", "TOP PICK JUDGE"),
    }
    clean = clean_ai_text(text.replace("```html", "").replace("```", ""))
    parts = re.split(r'(\[[A-Z_]+\])', clean)

    buckets: Dict[str, str] = {}
    cur = None
    buf = []
    for p in parts:
        if p in mapping:
            if cur and buf:
                buckets[cur] = (buckets.get(cur, "") + "\n" + "".join(buf)).strip()
            cur = p
            buf = []
        else:
            buf.append(p)
    if cur and buf:
        buckets[cur] = (buckets.get(cur, "") + "\n" + "".join(buf)).strip()

    order = [
        "[SECTOR_OUTLOOK]",
        "[JUDGE]",
        "[TOP_PICK_JUDGE]",
        "[TOPPICK_JUDGE]",
        "[TOP_PICK]",
        "[FUNDAMENTAL]",
        "[SENTIMENT]",
        "[VALUATION]",
        "[SKEPTIC]",
        "[RISK]",
    ]

    html = ""
    for tag in order:
        if tag not in buckets or not buckets[tag].strip():
            continue
        cls, label = mapping[tag]
        content = buckets[tag].strip()
        if tag == "[JUDGE]":
            # Judge should not show a separate 'Sector view:' block (avoid duplication)
            content = re.sub(r"(?mi)^Sector view:\s*.*?(\n\s*\n|\n(?=Stock pick:)|\Z)", "", content)
            content = content.strip()
        # remove extra headings like 'トリガー:' / 'Triggers:'; keep the sentence inside Sector view / Stock pick flow
        content = re.sub(r"(?mi)^(?:トリガー|トリガー\s*\(.*?\)|Triggers?)\s*[:：]\s*", "", content)
        # compact: remove excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)
        # remove stray backreference artifacts (\\1) and SOH that sometimes leak from regex replacement
        content = re.sub(r"(?m)^\s*(?:\\\\1|\x01)\s*", "", content)
        # remove any leading backslash-number artifacts like \1 that may appear at line starts
        content = re.sub(r'(?m)^\s*\\\d+\s*', '', content)
        # RISK block: flatten bullet lists into prose inside Sector view / Stock pick
        if tag == '[RISK]':
            content = _compress_bullets_in_views(content)
        content = content.replace("\\1", "").replace("\x01", "")
        # emphasize required sub-structure if present
        content = re.sub(r"(?m)^(Sector view:\s*)", r"<span class=\'subhead\'>\1</span>", content)
        content = re.sub(r"(?m)^(Stock pick:\s*)", r"<span class=\'subhead\'>\1</span>", content)
        content_html = "<div class='agent-content'>" + content.replace("\n", "<br>") + "</div>"

        if tag == "[SECTOR_OUTLOOK]":
            html += (
                f"<div class='{cls}' style='border-left:5px solid #00f2fe; margin-bottom:10px; padding:10px 12px;'>"
                f"<span class='orbitron' style='letter-spacing:0.8px; font-weight:900;'>{label}</span><br>{content_html}</div>"
            )
        elif tag == "[JUDGE]":
            # Judge: visually distinct
            html += (
                f"<div class='agent-row {cls}' style='margin-top:10px; padding:12px 14px; border:1px solid rgba(255,0,85,0.45); background: rgba(255,0,85,0.06);'>"
                f"<div class='agent-label' style='color:#ff0055; font-weight:800;'>{label}</div>{content_html}</div>"
            )
        else:
            html += f"<div class='agent-row {cls}'><div class='agent-label'>{label}</div>{content_html}</div>"

    return html
def run():
    # --- 1. INITIALIZE STATE ---
    if "system_logs" not in st.session_state: st.session_state.system_logs = []
    if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
    if "last_market_key" not in st.session_state: st.session_state.last_market_key = None
    if "last_lookback_key" not in st.session_state: st.session_state.last_lookback_key = None
    if "ai_nonce" not in st.session_state: st.session_state.ai_nonce = 0

    # --- UI STYLES ---
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@300;400;600;700&family=Orbitron:wght@400;600;900&family=JetBrains+Mono:wght@300;400;600&family=M+PLUS+1+Code:wght@300;400;700&display=swap');

:root{
  --bg:#000; --panel:#0a0a0a; --card:#111; --border:#333;
  --accent:#00f2fe; --accent2:#ff0055; --text:#e6e6e6;
  --fz-hero: clamp(28px, 3.2vw, 40px);
  --fz-h1: clamp(18px, 1.8vw, 24px);
  --fz-h2: clamp(15px, 1.4vw, 18px);
  --fz-body: clamp(12.5px, 1.05vw, 14px);
  --fz-note: clamp(10.5px, 0.95vw, 12px);
  --fz-table: 11px;
}

/* Base */
html, body, .stApp{
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Zen Kaku Gothic New', sans-serif !important;
  font-size: var(--fz-body) !important;
  line-height: 1.85 !important;
}
*{ letter-spacing: 0.02em !important; }

/* Headings / brand */
h1, h2, h3, .brand, .orbitron, div[data-testid="stMetricValue"]{
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase;
}
.brand{ 
  font-size: var(--fz-hero) !important;
  font-weight: 900 !important;
  background: linear-gradient(90deg, #00f2fe 0%, #e6e6e6 35%, #ff0055 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 18px rgba(0,242,254,0.12);
}

/* Notes / definitions */
.def-text{
  font-size: var(--fz-note) !important;
  color: #8a8a8a !important;
  line-height: 1.6 !important;
  border-bottom: 1px solid #333;
  padding-bottom: 8px;
  margin-bottom: 10px;
}
.caption-text{
  font-size: var(--fz-note) !important;
  color: #6f6f6f !important;
  font-family: 'Orbitron', sans-serif !important;
  letter-spacing: 0.05em !important;
}
div[data-testid="stCaptionContainer"] * { font-family:'Orbitron',sans-serif !important; letter-spacing:0.06em !important; }
div[data-testid="stMarkdownContainer"] small { font-family:'Orbitron',sans-serif !important; }

/* Data / numbers */
.mono, code, pre, div[data-testid="stDataFrame"] *{
  font-family: 'M PLUS 1 Code', monospace !important;
}
div[data-testid="stDataFrame"] *{
  font-size: var(--fz-table) !important;
  color: #f0f0f0 !important;
}

/* Report */
.report-box{
  background: #0a0a0a; border-top: 2px solid #00f2fe;
  padding: 14px; margin-top: 10px;
  font-size: var(--fz-body) !important;
  line-height: 1.75;
  color: #eee;
  white-space: pre-wrap;
}
.kpi-strip{
  font-family: 'M PLUS 1 Code', monospace !important;
  font-size: var(--fz-note) !important;
  color: #00f2fe !important;
  margin: 6px 0 10px 0;
}

/* Market Box */
.market-box{
  background:#080808; border:1px solid #333; padding:20px; margin:10px 0 18px 0;
}

/* Agent Council */
.agent-row{ display:flex; gap:10px; border:1px solid #222; padding:8px; margin:6px 0; background:#0b0b0b; width:100%; box-sizing:border-box; }
.agent-label{ flex:0 0 70px; min-width:70px; max-width:70px; font-family:'Orbitron',sans-serif !important; font-size:12px; color:#9adbe2; text-align:right; font-weight:700; word-break:break-word; line-height:1.15; padding-top:2px; }
.subhead{font-family:'JetBrains Mono',monospace; font-weight:700; color:#00f2fe;}
.agent-content{ flex:1 1 auto; min-width:0; white-space:pre-wrap; line-height:1.9; overflow-wrap:anywhere; }
.agent-verdict{ width:100%; box-sizing:border-box; overflow-wrap:anywhere; word-break:break-word; }
.agent-outlook{ border:1px solid #1d3c41; padding:12px; margin:8px 0; background:#061012; border-left:5px solid #00f2fe; }

/* Highlights */
.hl-pos{ color:#2cff7e; font-weight:800; }
.hl-neg{ color:#ff3b7a; font-weight:800; }
.hl-neutral{ color:#ffd166; font-weight:800; }

/* Buttons */
button{
  background:#111 !important;
  color: var(--accent) !important;
  border: 1px solid #444 !important;
  border-radius: 6px !important;
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 700 !important;
  font-size: 12px !important;
}
.action-call {
  font-family:'Orbitron',sans-serif; font-size:12px; color:#00f2fe; text-align:center;
  margin:8px 0 6px 0; padding:8px; border:1px solid #223; background:#050b0c;
}
/* Compact spacing */
.element-container{ margin-bottom: .35rem !important; }
.stMarkdown p{ margin: .25rem 0 !important; }


/* Controls: unify buttons + select boxes */
div.stButton > button, button[kind="primary"]{
  background: linear-gradient(90deg, rgba(0,242,254,0.22), rgba(255,0,85,0.14)) !important;
  border: 1px solid rgba(0,242,254,0.45) !important;
  color: var(--text) !important;
  border-radius: 14px !important;
  padding: 0.55rem 0.85rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em !important;
  box-shadow: 0 0 18px rgba(0,242,254,0.12) !important;
}
div.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 0 24px rgba(0,242,254,0.22) !important;
}
div.stButton > button:active{
  transform: translateY(0px);
  box-shadow: 0 0 10px rgba(255,0,85,0.18) !important;
}
div[data-baseweb="select"] > div{
  background: rgba(17,17,17,0.85) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
}
div[data-baseweb="select"] span{ color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)
    
    st.markdown("<h1 class='brand'>ALPHALENS</h1>", unsafe_allow_html=True)
    
    # 0. Controls
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1.0])
    with c1: market_key = st.selectbox("MARKET", list(MARKETS.keys()))
    with c2: lookback_key = st.selectbox("WINDOW", list(LOOKBACKS.keys()), index=1)
    with c3: st.caption(f"FETCH: {FETCH_PERIOD}"); st.progress(100)
    with c4:
        st.write("")
        run_ai = st.button("✨ GENERATE AI INSIGHTS", type="primary", use_container_width=True)
        refresh_prices = st.button("🔄 RELOAD MARKET DATA", use_container_width=True)
        qc_on = st.toggle("🛡️ AI output quality check", value=st.session_state.get("qc_on", True), help="Checks AI text for artifacts/overconfidence before showing. Does not add new facts.")
        st.session_state.qc_on = qc_on

    # Reset sector selection when MARKET/WINDOW changes
    prev_market = st.session_state.last_market_key
    prev_window = st.session_state.last_lookback_key
    market_changed = (prev_market != market_key)
    window_changed = (prev_window != lookback_key)

    if market_changed or window_changed:
        st.session_state.selected_sector = None
        # IMPORTANT: Market switch must not reuse previous market's cached data (causes BENCHMARK MISSING / SPY leakage in JP etc.)
        if market_changed:
            for k in ["core_df", "sec_df", "sec_stats", "news_cache", "ev_df", "audit"]:
                if k in st.session_state:
                    del st.session_state[k]
        st.session_state.last_market_key = market_key
        st.session_state.last_lookback_key = lookback_key

    if run_ai:
        # bust only AI cache (keeps price cache for speed)
        st.session_state.ai_nonce += 1
        st.toast("🤖 Running AI agents…", icon="🤖")

    if refresh_prices:
        # full refresh: clear cached price fetch + reset derived dfs
        try:
            st.cache_data.clear()
        except Exception:
            pass
        for k in ["core_df","sec_df","sec_stats","news_cache"]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.selected_sector = None
        st.toast("🔄 Refreshed prices", icon="🔄")

    m_cfg = MARKETS[market_key]
    win = LOOKBACKS[lookback_key]
    bench = m_cfg["bench"]
    # --- DATA FETCHING ---
    core_tickers = [bench] + list(m_cfg["sectors"].values())
    if refresh_prices or "core_df" not in st.session_state:
        with st.spinner("FETCHING MARKET DATA..."):
            raw = fetch_market_data(tuple(core_tickers), FETCH_PERIOD)
            st.session_state.core_df = extract_close_prices(raw, core_tickers)

    core_df = st.session_state.get("core_df", pd.DataFrame())
    if core_df.empty or len(core_df) < win + 1:
        st.warning("WAITING FOR DATA...")
        return

    audit = audit_data_availability(core_tickers, core_df, win)
    bench_used = bench
    # --- Benchmark robustness: best-effort alignment even when yfinance misses ---
    if bench not in audit.get("list", []):
        # Market-aware proxy candidates (avoid wrong-region proxies)
        proxy_by_market = {
            "US": ["^GSPC", "SPY", "VOO", "IVV"],
            "JP": ["^TOPX", "1306.T", "1321.T", "^N225"],
        }
        # Also allow bench-specific fallbacks
        proxy_by_bench = {
            "SPY": ["^GSPC", "VOO", "IVV"],
            "QQQ": ["^NDX", "^IXIC"],
            "1306.T": ["^TOPX", "1321.T", "^N225"],
        }
        mk = "JP" if str(market_key).endswith("JP") or "JP" in str(market_key) else "US"
        proxies = []
        proxies += proxy_by_bench.get(bench, [])
        proxies += proxy_by_market.get(mk, [])
        # Try already-fetched columns first
        for p in proxies:
            if p in core_df.columns and core_df[p].dropna().shape[0] >= win + 1:
                bench_used = p
                st.warning(f"BENCHMARK MISSING: using proxy {bench_used}. requested={bench} (best-effort; Market Pulse may be slightly degraded)")
                break
        else:
            # Last resort: pick any available series with sufficient history
            candidates = [c for c in core_df.columns if core_df[c].dropna().shape[0] >= win + 1]
            if candidates:
                bench_used = candidates[0]
                st.warning(f"BENCHMARK MISSING: using available series {bench_used}. requested={bench} (best-effort; Market Pulse may be degraded)")
            else:
                st.error("BENCHMARK MISSING: no usable series for the selected window.")
                return


    # 1. Market Pulse
    b_stats = calc_technical_metrics(core_df[bench_used], core_df[bench_used], win)
    if not b_stats: st.error("BENCH ERROR"); return

    regime, weight_mom = calculate_regime(core_df[bench_used].dropna())
    
    sec_rows = []
    for s_n, s_t in m_cfg["sectors"].items():
        if s_t in audit["list"]:
            res = calc_technical_metrics(core_df[s_t], core_df[bench_used], win)
            if res:
                res["Sector"] = s_n
                sec_rows.append(res)
    
    if not sec_rows:
        st.warning("SECTOR DATA INSUFFICIENT (continuing with degraded view)")
        sdf = pd.DataFrame()
    else:
        sdf = pd.DataFrame(sec_rows).sort_values("RS", ascending=True)

    # --- Spread robustness: ensure defined in all paths ---
    try:
        spread = float(sdf['RS'].max() - sdf['RS'].min()) if (not sdf.empty and 'RS' in sdf.columns) else 0.0
    except Exception:
        spread = 0.0

    s_date = core_df.index[-win-1].strftime('%Y/%m/%d')
    e_date = core_df.index[-1].strftime('%Y/%m/%d')
        # --- News robustness: never fail Market Pulse ---
    market_context, m_sent, m_meta = [], 0, {}
    try:
        _, market_context, m_sent, m_meta = get_news_consolidated(bench, m_cfg["name"], market_key)
    except Exception:
        market_context, m_sent, m_meta = [], 0, {}
    # News sentiment (robust defaults)
    try:
        s_score = int(np.clip(int(round(float(m_sent or 0))), -10, 10))
    except Exception:
        s_score = 0
    lbl = "Positive" if s_score > 0 else ("Negative" if s_score < 0 else "Neutral")
    hit_pos = int((m_meta or {}).get("pos", 0))
    hit_neg = int((m_meta or {}).get("neg", 0))
    s_cls = "hl-pos" if s_score > 0 else ("hl-neg" if s_score < 0 else "hl-neutral")

    
    # Definition Header (ORDER FIXED: Spread -> Regime -> NewsSent)
    index_name = get_name(bench)
    index_label = f"{index_name} ({bench})" if index_name else bench

    st.markdown(f"""
    <div class='market-box'>
    <b class='orbitron'>MARKET PULSE ({s_date} - {e_date})</b><br>
    <span class='caption-text'>Spread: {spread:.1f}pt | Regime: {regime} | NewsSent: <span class='{s_cls}'>{s_score:+d}</span> ({lbl}) [Hit:{hit_pos}/{hit_neg}]</span><br>
    """ + market_to_html(force_nonempty_outlook_market(
        enforce_market_format(enforce_index_naming(generate_ai_content("market", {
            "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
            "top": sdf.iloc[-1]["Sector"], "bot": sdf.iloc[0]["Sector"],
            "market_name": m_cfg["name"], "headlines": market_context,
            "date_slots": outlook_date_slots(),
            "index_label": index_label,
            "nonce": st.session_state.ai_nonce
        }), index_label)), regime, b_stats["Ret"], spread, market_key
    )) + f"""
    <div class='def-text'>
    <b>DEFINITIONS</b> |
    <b>Spread</b>: セクターRSの最大−最小(pt)。市場内の勝ち負けがどれだけ鮮明かを示す |
    <b>Regime</b>: 200DMA判定（終値&gt;200DMA=Bull / 終値&lt;200DMA=Bear） |
    <b>NewsSent</b>: 見出しキーワード命中（pos=+1/neg=−1）合計を−10〜+10にクリップ |
    <b>RS</b>: 相対リターン差(pt)=セクター(or銘柄)リターン−市場平均リターン
    </div>
    </div>""", unsafe_allow_html=True)

    
    # If sector data is unavailable, stop after Market Pulse (degraded but stable)
    if sdf is None or (isinstance(sdf, pd.DataFrame) and sdf.empty):
        st.info("Sector rotation / sector leaderboard unavailable (insufficient sector ETF history for the selected window). Try REFRESH PRICES or a longer WINDOW.")
        return

# 2. Sector Rotation
    st.subheader(f"SECTOR ROTATION ({s_date} - {e_date})")
    
    # Sort by Return for Display/Button (Requirement)
    sdf["Label"] = sdf["Sector"] + " (" + sdf["Ret"].apply(lambda x: f"{x:+.1f}%") + ")"
    # Sort Descending (Top=Max)
    sdf_disp = sdf.sort_values("Ret", ascending=False).reset_index(drop=True)
    
    # Default Selection: Max Return (Always Top)
    if not st.session_state.selected_sector:
        best_row = sdf_disp.iloc[0]
        st.session_state.selected_sector = best_row["Sector"]

    click_sec = st.session_state.selected_sector

    # --- Gradient coloring by RS (pro look) ---
    rs_vals = pd.to_numeric(sdf_disp["RS"], errors="coerce").fillna(0.0)
    cmin = float(rs_vals.min()) if len(rs_vals) else -1.0
    cmax = float(rs_vals.max()) if len(rs_vals) else 1.0
    # Avoid zero-range colorbar
    if abs(cmax - cmin) < 1e-9:
        cmin, cmax = cmin - 1.0, cmax + 1.0

    # Highlight selected sector with outline (no color override to keep gradient)
    line_w = [2.5 if s == click_sec else 0.0 for s in sdf_disp["Sector"].tolist()]
    line_c = ["#e6e6e6" if s == click_sec else "rgba(0,0,0,0)" for s in sdf_disp["Sector"].tolist()]

    # Plot
    fig = px.bar(sdf_disp, x="RS", y="Label", orientation='h', title=f"Relative Strength ({lookback_key})")
    fig.update_traces(
        customdata=np.stack([sdf_disp["Ret"]], axis=-1),
        hovertemplate="%{y}<br>Ret: %{customdata[0]:+.1f}%<br>RS: %{x:.2f}<extra></extra>",
        marker=dict(
            color=rs_vals,
            colorscale="RdYlGn",
            cmin=cmin,
            cmax=cmax,
            line=dict(color=line_c, width=line_w),
        ),
    )
# Fix Plotly sorting (array order)
    fig.update_layout(height=420, margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font_color='#e0e0e0', font_family="JetBrains Mono", 
                      xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True, categoryorder="array", categoryarray=sdf_disp["Label"].tolist()[::-1]))
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True, 'displayModeBar': False})
    
    st.markdown("<div class='action-call'>👇 Select a SECTOR to run AI agents (Top Pick)</div>", unsafe_allow_html=True)
    
    # Buttons
    st.write("SELECT SECTOR:")
    cols = st.columns(2)
    for i, row in enumerate(sdf_disp.itertuples()):
        s = row.Sector
        label = f"✅ {s} ({row.Ret:+.1f}%)" if s == st.session_state.selected_sector else f"{s} ({row.Ret:+.1f}%)"
        if cols[i%2].button(label, key=f"btn_{s}", use_container_width=True):
            st.session_state.selected_sector = s
            st.rerun()
            
    target_sector = st.session_state.selected_sector or sdf_disp.iloc[0]["Sector"]

    # 3. Sector Forensic
    st.markdown(f"<div id='sector_anchor'></div>", unsafe_allow_html=True)
    st.divider()
    st.subheader(f"SECTOR FORENSIC: {target_sector}")
    
    stock_list = m_cfg["stocks"].get(target_sector, [])
    if not stock_list: st.warning("No stocks."); return

    full_list = [bench] + stock_list
    cache_key = f"{market_key}_{target_sector}_{lookback_key}"
    
    if cache_key != st.session_state.get("sec_cache_key") or refresh_prices:
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
    
        # TopPickScore (momentum/news-centric): 3M & 1M dominate
    df["Apex"] = 0.45 * calculate_zscore(df["3M"]) + 0.35 * calculate_zscore(df["1M"]) + 0.15 * calculate_zscore(df["RS"]) + 0.05 * calculate_zscore(df["Accel"])
    df = df.sort_values("Apex", ascending=False)
    
    # 4. Top pick selection (fast)
    top3 = df.head(1).copy()  # keep variable name for downstream code
    neg = df.iloc[0:0].copy()  # empty
    # Fetch fundamentals for Top3 + Neg for debate context
    cand_tickers = top3["Ticker"].tolist()
    if not neg.empty: cand_tickers.append(neg.iloc[0]["Ticker"])
    cand_fund = fetch_fundamentals_batch(cand_tickers).reset_index()
    
    # Build context lines
    cand_lines = []
    for _, r in top3.iterrows():
        f = pick_fund_row(cand_fund, r["Ticker"])
        cand_lines.append(
            f"{r['Name']}({r['Ticker']}): Ret {r['Ret']:.1f}%, RS {r['RS']:.2f}, Accel {r['Accel']:.2f}, HighDist {r['HighDist']:.1f}%, "
            f"MCap {sfloat(f.get('MCap',0))/1e9:.1f}B, PER {dash(f.get('PER'))}, PBR {dash(f.get('PBR'))}"
        )
    if not neg.empty:
        nr = neg.iloc[0]
        f = pick_fund_row(cand_fund, nr["Ticker"])
        cand_lines.append(f"\n[AVOID] {nr['Name']}: Ret {nr['Ret']:.1f}%, RS {nr['RS']:.2f}, PER {dash(f.get('PER'))}")

    _, sec_news, _, _ = get_news_consolidated(m_cfg["sectors"][target_sector], target_sector, market_key, limit_each=3)
    
    # Sector Stats
    sector_stats = f"Universe:{len(stock_list)} Computable:{len(df)} MedianRS:{df['RS'].median():.2f} MedianRet:{df['Ret'].median():.1f}% SpreadRS:{(df['RS'].max()-df['RS'].min()):.2f}"

    # ---- stringify context for AI (never undefined) ----
    sector_stats_str = sector_stats

    # Candidates (top pick + optional avoid) as compact lines
    top3_str = "\n".join([x for x in cand_lines if isinstance(x, str) and x.strip()]) if 'cand_lines' in locals() else ""
    if not top3_str:
        try:
            top3_str = top_line
        except Exception:
            top3_str = ""

    # News to text (cap length, robust to schema)
    def _news_to_str(items, limit=6):
        if not items:
            return ""
        out = []
        for it in list(items)[:limit]:
            try:
                if isinstance(it, dict):
                    title = it.get("title") or it.get("Title") or it.get("headline") or ""
                    src = it.get("source") or it.get("Source") or it.get("publisher") or ""
                elif isinstance(it, (list, tuple)) and len(it) >= 1:
                    title = str(it[0])
                    src = str(it[1]) if len(it) > 1 else ""
                else:
                    title = str(it)
                    src = ""
                title = str(title).strip()
                src = str(src).strip()
                if title:
                    out.append(f"- {title}" + (f" ({src})" if src else ""))
            except Exception:
                continue
        return "\n".join(out)

    sec_news_str = _news_to_str(sec_news, limit=6) if 'sec_news' in locals() else ""
    
    # 🦅 🤖 AI AGENT SECTOR REPORT (fast, top-pick focused)
    tp = df.iloc[0]
    tp_f = pick_fund_row(cand_fund, tp["Ticker"])
    top_line = (
        f"[TOP] {tp['Name']} ({tp['Ticker']}): Ret {tp['Ret']:.1f}%, RS {tp['RS']:.2f}, Accel {tp['Accel']:.2f}, "
        f"HighDist {tp['HighDist']:.1f}%, MaxDD {tp['MaxDD']:.1f}%, "
        f"MCap {sfloat(tp_f.get('MCap',0))/1e9:.1f}B, PER {dash(tp_f.get('PER'))}, PBR {dash(tp_f.get('PBR'))}"
    )

    sec_ai_raw = generate_ai_content("sector_debate_fast", {
        "sec": target_sector,
        # Avoid NameError: use market config name directly (with safe fallback)
        "market_name": m_cfg.get("name", str(market_key)),
        "sector_stats": sector_stats_str,
        "top": top3_str,
        "news": sec_news_str,
        "nonce": st.session_state.ai_nonce
    })
    sec_ai_txt = quality_gate_text(enforce_da_dearu_soft(sec_ai_raw), enable=st.session_state.get('qc_on', True))
    sec_ai_html = parse_agent_debate(sec_ai_txt) if ("[FUNDAMENTAL]" in sec_ai_txt or "[SECTOR_OUTLOOK]" in sec_ai_txt) else sec_ai_txt
    st.markdown(f"<div class='report-box'><b>🦅 🤖 AI AGENT SECTOR REPORT</b><br>{sec_ai_html}</div>", unsafe_allow_html=True)
    # Download Council Log (before leaderboard)
    st.download_button("DOWNLOAD COUNCIL LOG", sec_ai_raw, f"council_log_target_sector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    ev_fund = fetch_fundamentals_batch(top3["Ticker"].tolist()).reset_index()
    ev_df = top3.merge(ev_fund, on="Ticker", how="left")
    for c in ["PER","PBR"]:
        if c not in ev_df.columns: ev_df[c] = np.nan
        ev_df[c] = ev_df[c].apply(lambda x: dash(x))
    for c in ["ROE","RevGrow","OpMargin"]:
        if c not in ev_df.columns: ev_df[c] = np.nan
        ev_df[c] = ev_df[c].apply(pct)
    if "Beta" not in ev_df.columns: ev_df["Beta"] = np.nan
    ev_df["Beta"] = ev_df["Beta"].apply(lambda x: dash(x, "%.2f"))
    
    cols = ['Name', 'Ticker', 'Apex', 'RS', 'Accel', 'Ret', '1M', '3M', 'HighDist', 'MaxDD', 'PER', 'PBR', 'ROE', 'RevGrow', 'OpMargin', 'Beta']
    cols = [c for c in cols if c in ev_df.columns]
    st.dataframe(ev_df[cols], hide_index=True, use_container_width=True)

    # 5. Leaderboard
    universe_cnt = len(stock_list)
    computable_cnt = len(df)
    up = int((df["Ret"] > 0).sum())
    down = computable_cnt - up
    st.markdown(f"##### LEADERBOARD (Universe: {universe_cnt} | Computable: {computable_cnt} | Up: {up} | Down: {down})")
    
    tickers_for_fund = df.head(20)["Ticker"].tolist()
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
    
    st.markdown("<div class='action-call'>👇 Select ONE stock to generate the AI agents' analysis note below</div>", unsafe_allow_html=True)
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
            "RevGrow": st.column_config.TextColumn("RevGrow"),
            "OpMargin": st.column_config.TextColumn("OpMargin"),
            "Beta": st.column_config.TextColumn("Beta"),
            "1M": st.column_config.NumberColumn(format="%.1f%%"),
            "12M": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="stock_table"
    )

    st.caption(
        "DEFINITIONS | Apex: zscore合成=weight_mom*z(RS)+(0.8-weight_mom)*z(Accel)+0.2*z(Ret) | "
        "RS: Ret(銘柄)−Ret(市場平均) | Accel: 直近半期間リターン−(全期間リターン/2) | "
        "HighDist: 直近価格の52週高値からの乖離(%) | MaxDD: 期間内最大ドローダウン(%) | "
        "PER/PBR/ROE等: yfinance.Ticker().info（負のPER/PBRは除外、欠損は'-'）"
    )
    st.caption(
        "SOURCE & NOTES | Price: yfinance.download(auto_adjust=True) | Fundamentals: yfinance.Ticker().info | "
        "Up/Down: 期間リターンが + の銘柄数 / それ以外（0以下）の銘柄数 | "
        "PER/PBR: 負値は除外 | ROE/RevGrow/OpMargin/Beta: 取得できる場合のみ表示 | "
        "Apex/RS/Accel等は本アプリ算出"
    )
    

    # 6. Deep Dive
    top = df_sorted.iloc[0]
    try:
        if hasattr(event, "selection") and event.selection:
            sel_rows = event.selection.get("rows", [])
            if sel_rows: top = df_sorted.iloc[sel_rows[0]]
    except: pass

    st.divider()
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"### 🦅 🤖 AI EQUITY ANALYST {top['Name']}")
    st.caption(f"Data Timestamp: {now_str} | Source: yfinance (PER/PBR exclude negatives)")
    
    news_items, news_context, _, _ = get_news_consolidated(top["Ticker"], top["Name"], market_key, limit_each=10)
    fund_data = get_fundamental_data(top["Ticker"])
    overview = ""
    try:
        bsum = str(fund_data.get("BusinessSummary") or fund_data.get("Summary") or "").strip()
        if len(bsum) > 280:
            bsum = bsum[:280].rstrip() + "…"
        sec_name = str(fund_data.get("Sector") or "-")
        ind_name = str(fund_data.get("Industry") or "-")
        mcap = fund_data.get("MCap") or fund_data.get("MarketCap") or 0
        mcap_disp = dash(mcap, "%.0f")
        if isinstance(mcap, (int, float)) and mcap:
            if mcap >= 1e12:
                mcap_disp = f"{mcap/1e12:.1f}T"
            elif mcap >= 1e9:
                mcap_disp = f"{mcap/1e9:.1f}B"
            elif mcap >= 1e6:
                mcap_disp = f"{mcap/1e6:.0f}M"
        overview = f"Sector:{sec_name} | Industry:{ind_name} | MCap:{mcap_disp} | Summary:{bsum}"
    except Exception:
        overview = ""
    # --- Company Overview (always visible) ---
    # --- Company Overview (always visible) ---
    if not overview:
        overview = "Sector:- | Industry:- | MCap:- | Summary:-"
    st.markdown(f"<div class='note-box'><b>Company Overview</b><br>{overview}</div>", unsafe_allow_html=True)
    # --- External chart links (1Y chart removed by design) ---
    try:
        def _tv_symbol(tk: str) -> str:
            # TradingView symbol mapping (best-effort)
            if tk.endswith(".T"):
                return "TSE:" + tk.replace(".T","")
            if tk.startswith("^"):
                # indices: use generic (may not resolve on TV)
                return tk.replace("^","")
            return tk
        yf_url = f"https://finance.yahoo.com/quote/{top['Ticker']}"
        tv_url = f"https://www.tradingview.com/symbols/{_tv_symbol(top['Ticker'])}/"
        st.markdown(f"<div class='mini-note'>Charts: <a href='{yf_url}' target='_blank'>Yahoo Finance</a> | <a href='{tv_url}' target='_blank'>TradingView</a></div>", unsafe_allow_html=True)
    except Exception:
        pass

    ed = fetch_earnings_dates(top["Ticker"]).get("EarningsDate", "-")
    bench_fd = get_fundamental_data(bench)
    
    # Price Action Pack
    pa = {}
    try:
        if "sec_df" in st.session_state and top["Ticker"] in st.session_state.sec_df.columns:
            pa = price_action_pack(st.session_state.sec_df[top["Ticker"]])
    except: pass
    
    price_act = ""
    if pa:
        price_act = f"Last {pa.get('Last',np.nan):.2f} | 1D {pa.get('1D',np.nan):+.2f}% | 1W {pa.get('1W',np.nan):+.2f}% | 1M {pa.get('1M',np.nan):+.2f}% | 3M {pa.get('3M',np.nan):+.2f}% | 200DMA {pa.get('200DMA_Dist',np.nan):+.1f}% | MaxDD(6M) {pa.get('MaxDD_6M',np.nan):.1f}%"

    st.markdown(f"<div class='kpi-strip mono'>{price_act}</div>", unsafe_allow_html=True)

    bench_per = dash(bench_fd.get("PER"))
    sector_per = dash(pd.to_numeric(df["PER"], errors="coerce").median())
    stock_per = dash(fund_data.get("PER"))
    m_comp = f"市場平均PER: {bench_per}倍 / セクター中央値PER: {sector_per}倍 / 当該銘柄PER: {stock_per}倍"
    
    fund_str = f"PER:{stock_per}, PBR:{dash(fund_data.get('PBR'))}, PEG:{dash(fund_data.get('PEG'))}, Target:{dash(fund_data.get('Target'))}"

    report_txt = generate_ai_content("stock_report", {
        "name": top["Name"], "ticker": top["Ticker"],
        "overview": overview, "fund_str": fund_str, "m_comp": m_comp, "news": news_context,
        "earnings_date": ed, "price_action": price_act, "nonce": st.session_state.ai_nonce
    })
    report_txt = clean_ai_text(report_txt)
    # Prepend Company Overview and Quantitative Summary (always) to the downloadable analyst note
    overview_plain = re.sub(r"<[^>]+>", "", overview).strip() if isinstance(overview, str) else ""
    if not overview_plain:
        overview_plain = "Sector:- | Industry:- | MCap:- | Summary:-"
    # --- Company Overview (always shown above the report as well) ---
    # --- Company Overview (always visible; markdown-based to avoid CSS/HTML issues) ---
    overview_md = f"""**Company Overview — {top['Name']} ({top['Ticker']})**
- Sector: {sec_name}
- Industry: {ind_name}
- Market Cap: {mcap_disp}
- Website: {fund_data.get('Website') or '-'}
- Summary: {bsum if bsum else '-'}"""
    st.markdown(overview_md)
    analyst_note_txt = (
        "Company Overview\n" + f"Name: {top['Name']} ({top['Ticker']})\nSector: {sec_name}\nIndustry: {ind_name}\nMarket Cap: {mcap_disp}\nWebsite: {fund_data.get('Website') or '-'}\nSummary: {bsum if bsum else '-'}" + "\n\n"
        "Quantitative Summary\n" + fund_str + "\n\n"
        + report_txt
    )
    report_txt_disp = quality_gate_text(enforce_da_dearu_soft(report_txt), enable=st.session_state.get('qc_on', True))
    
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        st.markdown(f"<div class='report-box'><b>AI EQUITY BRIEFING</b><br>{overview_plain}<br><br>{report_txt_disp}</div>", unsafe_allow_html=True)

        # Links
        links = build_ir_links(top["Name"], top["Ticker"], fund_data.get("Website"), market_key)
        lc1, lc2, lc3 = st.columns(3)
        with lc1: safe_link_button("OFFICIAL", links["official"], use_container_width=True)
        with lc2: safe_link_button("IR SEARCH", links["ir_search"], use_container_width=True)
        with lc3: safe_link_button("EARNINGS DECK", links["earnings_deck"], use_container_width=True)
        try:
            target_mcap = top["MCap"] if pd.notna(top["MCap"]) else 0
            df_peers_base = df_sorted.copy()
            df_peers_base["Dist"] = (pd.to_numeric(df_peers_base["MCap"], errors="coerce") - float(target_mcap or 0)).abs()
            df_peers = df_peers_base.sort_values("Dist").iloc[1:5]
            st.dataframe(df_peers[["Name", "ROE", "RevGrow", "PER", "PBR", "RS", "12M"]], hide_index=True)
        except: pass
        
        st.caption(
            "PEER LOGIC | Nearest Market Cap: |MCap(peer)−MCap(target)|が小さい順に抽出（同一セクター内） | "
            "SOURCE: yfinance.Ticker().info（欠損は'-'）"
        )
        st.download_button("DOWNLOAD ANALYST NOTE", analyst_note_txt, f"analyst_note_{top['Ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with nc2:
        st.caption("INTEGRATED NEWS FEED")
        for n in news_items[:20]:
            dt = datetime.fromtimestamp(n["pub"]).strftime("%Y/%m/%d") if n["pub"] else "-"
            st.markdown(f"- {dt} [{n['src']}] [{n['title']}]({n['link']})")

if __name__ == "__main__":
    run()
