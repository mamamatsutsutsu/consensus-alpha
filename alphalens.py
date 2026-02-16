import os
import time
import re
import math
import html as html_lib
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
BUILD_ID = "2026-02-16-r5"  # visible build stamp for ops/debug

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


def _norm_currency(cur: Optional[str]) -> str:
    try:
        if not cur:
            return ""
        c = str(cur).upper().strip()
        return c if 1 <= len(c) <= 6 else ""
    except Exception:
        return ""

def format_mcap(mcap: Any, currency: Optional[str] = None) -> str:
    """Human-friendly market cap with currency prefix when available (e.g., 'USD 626.0B')."""
    try:
        if pd.isna(mcap):
            return "-"
        v = float(mcap)
        if not (v > 0):
            return "-"
    except Exception:
        return "-"

    cur = _norm_currency(currency)
    if v >= 1e12:
        val = f"{v/1e12:.1f}T"
    elif v >= 1e9:
        val = f"{v/1e9:.1f}B"
    elif v >= 1e6:
        val = f"{v/1e6:.0f}M"
    else:
        val = f"{v:.0f}"

    return f"{cur} {val}".strip() if cur else val

def outlook_date_slots(days: List[int] = [7, 21, 35, 49, 63, 84]) -> List[str]:
    base = datetime.now().date()
    return [(base + timedelta(days=d)).strftime("%Y/%m/%d") for d in days]

def _safe_http_url(url: Any) -> str:
    """Allow only http/https URLs for clickable links (defense-in-depth).

    Also normalizes bare domains like 'example.com' into 'https://example.com'.
    """
    try:
        if not url:
            return ""
        u = str(url).strip()
        if not u:
            return ""
        # If scheme is missing but it looks like a domain, prefix https://
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
            if re.search(r"\s", u):
                return ""
            u2 = u.lstrip("/")
            if "." in u2:
                u = "https://" + u2
        p = urllib.parse.urlparse(u)
        if p.scheme not in ("http", "https"):
            return ""
        if not p.netloc:
            return ""
        return u
    except Exception:
        return ""

def safe_link_button(label: str, url: str, use_container_width: bool = True):
    """Render a high-contrast, mobile-friendly link button.

    We intentionally avoid st.link_button here because its styling can become unreadable
    on dark themes (especially on mobile). This uses our own CSS class `.link-btn`.
    """
    u = _safe_http_url(url)
    if not u:
        st.button(label, disabled=True, use_container_width=use_container_width)
        return

    lbl = html_lib.escape(str(label), quote=True)
    u_e = html_lib.escape(u, quote=True)
    width_style = "width:100%;" if use_container_width else ""
    st.markdown(
        f"<a class='link-btn' href='{u_e}' target='_blank' rel='noopener noreferrer' style='{width_style}'>{lbl} <span class='link-arrow'>↗</span></a>",
        unsafe_allow_html=True,
    )

def build_ir_links(name: str, ticker: str, website: Optional[str], market_key: str) -> Dict[str, str]:
    q_site = urllib.parse.quote(name)
    q_ir = urllib.parse.quote(f"{name} IR")
    if "US" in market_key:
        q_deck = urllib.parse.quote(f"{name} investor presentation earnings pdf")
    else:
        q_deck = urllib.parse.quote(f"{name} 決算説明資料 pdf")
            
    official = _safe_http_url(website) or f"https://www.google.com/search?q={q_site}+official+site"
    
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
    """Return a clean Close-price matrix with columns=tickers.

    yfinance.download(..., group_by="ticker") usually returns a MultiIndex column:
      (Field, Ticker) or (Ticker, Field). However, for a single ticker it may return a
      single-level column DataFrame with fields (Open/High/Low/Close/...).

    This helper supports both shapes and always returns a DataFrame of Close prices.
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()

    try:
        close = None

        if isinstance(df.columns, pd.MultiIndex):
            # Prefer the "Close" level regardless of which axis level it sits on
            if "Close" in df.columns.get_level_values(0):
                close = df.xs("Close", axis=1, level=0)
            elif "Close" in df.columns.get_level_values(1):
                close = df.xs("Close", axis=1, level=1)
            elif "Adj Close" in df.columns.get_level_values(0):
                close = df.xs("Adj Close", axis=1, level=0)
            elif "Adj Close" in df.columns.get_level_values(1):
                close = df.xs("Adj Close", axis=1, level=1)
            else:
                return pd.DataFrame()
        else:
            # Single ticker case: columns are fields
            if "Close" in df.columns:
                s = df["Close"]
            elif "Adj Close" in df.columns:
                s = df["Adj Close"]
            else:
                return pd.DataFrame()

            # Name the column as the (only) expected ticker if possible
            col_name = expected[0] if expected and len(expected) == 1 else "Close"
            close = s.to_frame(name=col_name)

        # Coerce numeric and drop fully-empty rows
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        # If we expected specific tickers, filter in that order when possible
        if expected:
            cols = [c for c in expected if c in close.columns]
            if cols:
                close = close[cols]
            elif close.shape[1] == 1 and len(expected) == 1:
                # If the single column name doesn't match, rename it (best-effort)
                close = close.rename(columns={close.columns[0]: expected[0]})
        return close
    except Exception:
        return pd.DataFrame()

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
    # MaxDD: max drawdown in the window (negative %, 0 is best)
    dd = float(((s_win / s_win.cummax() - 1) * 100).min())
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
                "Currency": i.get("currency") or i.get("financialCurrency"),
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

    # Summary: prefer faithful translation when available; otherwise fall back to conservative AI summary from FACTS only.
    summary = str(profile.get("BusinessSummary") or "").strip()

    # Translate to Japanese (translation only, no new facts) when AI is enabled and the source is not Japanese.
    if enable_ai and summary and (not _looks_japanese(summary)):
        tr = translate_summary_to_japanese_cached(summary[:1200], nonce=nonce)
        if tr:
            summary = tr

    # If yfinance has no summary, ask AI to produce a conservative one from facts only (still JP).
    if enable_ai and (not summary):
        facts = {
            "name": name,
            "sector": sector,
            "industry": industry,
            "country": country,
            "website": website or "-",
            "marketCap": mcap_disp,
            "yfinance_summary": "-",
        }
        ai_sum = ai_company_summary_cached(name, facts, nonce=nonce)
        if ai_sum:
            summary = ai_sum

    if not summary:
        summary = "-"

    # Keep "overview_html" summary-free (UI-safe), but keep "overview_plain" with Summary for AI context/logging.
    overview_html = f"Sector:{sector} | Industry:{industry} | MCap:{mcap_disp} | Country:{country}"
    overview_plain = f"Sector:{sector} | Industry:{industry} | MCap:{mcap_disp} | Summary:{summary}"

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "country": country,
        "website": website,
        "mcap": mcap,
        "mcap_disp": mcap_disp,
        "summary": summary,
        "overview_html": overview_html,
        "overview_plain": overview_plain,
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


def _looks_japanese(s: str) -> bool:
    """Heuristic: detect Japanese scripts (Hiragana/Katakana/Kanji)."""
    try:
        return bool(re.search(r"[ぁ-ゔァ-ヴー一-龥]", str(s)))
    except Exception:
        return False

def text_to_safe_html(text: Any) -> str:
    """Escape text for safe HTML embedding (prevents HTML injection).
    Converts newlines to <br> for display inside st.markdown(unsafe_allow_html=True) wrappers.
    """
    if text is None:
        return ""
    t = str(text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return html_lib.escape(t, quote=True).replace("\n", "<br>")


def split_snapshot_jp(text: str) -> Tuple[str, str]:
    """Extract a one-line Japanese snapshot from a report.

    Expected format (single line, no markdown):
      SNAPSHOT_JP: <JP 180-220 chars>

    Returns:
      (snapshot_text, body_without_snapshot_line)
    """
    try:
        if text is None:
            return "", ""
        t = str(text).replace("\r\n", "\n").replace("\r", "\n")
        m = re.search(r"(?im)^\s*SNAPSHOT_JP\s*[:：]\s*(.+?)\s*$", t)
        if not m:
            return "", t.strip()
        snap = m.group(1).strip()
        body = re.sub(r"(?im)^\s*SNAPSHOT_JP\s*[:：]\s*.+\s*\n?", "", t, count=1).strip()
        # Soft length guard (JP chars). If too long, truncate with ellipsis.
        if len(snap) > 220:
            snap = snap[:219] + "…"
        return snap, body
    except Exception:
        return "", str(text).strip()

@st.cache_data(ttl=24*3600, show_spinner=False)
def translate_summary_to_japanese_cached(summary_text: str, nonce: int = 0) -> str:
    """Translate an English business summary into Japanese (faithful, no new facts).
    Returns empty string on failure (fail-closed).
    """
    try:
        if summary_text is None:
            return ""
        src = str(summary_text).strip()
        if not src or src in ("-", "nan", "None"):
            return ""
        if _looks_japanese(src):
            return src
        if not HAS_LIB or not API_KEY:
            return ""
        m = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
あなたは金融文書の翻訳者。
以下の英文のCompany Summaryを、日本語に忠実に翻訳せよ。

禁止: 新しい事実の追加、推測、内容の増補、誇張、具体例の創作。
厳守: 数字・地名・事業セグメント名・会社名・ティッカーは意味を変えず保持する。
文体: だ・である調。
出力: 翻訳本文のみ（箇条書きや見出しは不要）。

英文: {src}
"""
        out = m.generate_content(prompt).text or ""
        out = clean_ai_text(out)
        return out if _looks_japanese(out) else ""
    except Exception:
        return ""


def quality_gate_text(text: str, enable: bool = True) -> str:
    """Lightweight, safe post-processor for AI text before rendering.
    - Removes meta/preambles
    - Removes stray artifacts (e.g., \1)
    - (Optional) Softens overconfident claims
    - Keeps structure but does NOT add any new facts
    """
    t = clean_ai_text(text)

    # If QC is disabled, we still do basic cleanup (no opinionated edits).
    if not enable:
        return re.sub(r"\n{2,}", "\n", t).strip()

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
    """Render Market Pulse text safely inside HTML wrappers.

    - Escapes any HTML from the model (prevents injection)
    - Keeps our own highlighting spans for (+)/(−) lines
    """
    safe = html_lib.escape(str(text) if text is not None else "", quote=True)
    safe = re.sub(r"(^\(\+\s*\).*$)", r"<span class='hl-pos'>\1</span>", safe, flags=re.MULTILINE)
    safe = re.sub(r"(^\(\-\s*\).*$)", r"<span class='hl-neg'>\1</span>", safe, flags=re.MULTILINE)
    return safe.replace("\n", "<br>")

@st.cache_data(ttl=1800)
def get_news_consolidated(ticker: str, name: str, market_key: str, limit_each: int = 10) -> Tuple[List[dict], str, int, Dict[str,int]]:
    """Collect news from Yahoo + Google News RSS + a few public RSS feeds.

    Speed: public RSS sources are fetched in parallel.
    Output:
      - news_items: list of {title, link, pub, src}
      - context_str: up to 15 lines, newest first, in the form '- [SRC YYYY/MM/DD] Title'
      - sentiment_score: lightweight headline sentiment (pos/neg keywords, recency weighted)
      - meta: counts for source hits + pos/neg hits
    """
    pos_words = ["増益", "最高値", "好感", "上昇", "自社株買い", "上方修正", "急騰", "beat", "high", "jump", "record"]
    neg_words = ["減益", "安値", "嫌気", "下落", "下方修正", "急落", "赤字", "miss", "low", "drop", "warn"]

    def _fetch_yahoo() -> List[dict]:
        items: List[dict] = []
        try:
            raw = yf.Ticker(ticker).news or []
            for n in raw[:limit_each]:
                t = n.get("title", "") or ""
                l = n.get("link", "") or ""
                p = n.get("providerPublishTime", 0) or 0
                items.append({"title": t, "link": l, "pub": int(p) if p else 0, "src": "Yahoo"})
        except Exception:
            pass
        return items

    def _fetch_google() -> List[dict]:
        items: List[dict] = []
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
                t = i.findtext("title") or ""
                l = i.findtext("link") or ""
                d = i.findtext("pubDate") or ""
                try:
                    pub = int(email.utils.parsedate_to_datetime(d).timestamp())
                except Exception:
                    pub = 0
                items.append({"title": t, "link": l, "pub": pub, "src": "Google"})
        except Exception:
            pass
        return items

    def _fetch_rss(src: str, url2: str, n: int) -> List[dict]:
        items: List[dict] = []
        try:
            with urllib.request.urlopen(url2, timeout=3) as r:
                root = ET.fromstring(r.read())
            for it in root.findall(".//item")[:n]:
                t2 = it.findtext("title") or ""
                l2 = it.findtext("link") or ""
                d2 = it.findtext("pubDate") or ""
                try:
                    pub2 = int(email.utils.parsedate_to_datetime(d2).timestamp())
                except Exception:
                    pub2 = 0
                if not t2:
                    continue
                items.append({"title": t2, "link": l2, "pub": pub2, "src": src})
        except Exception:
            pass
        return items

    rss_sources = [
        ("Reuters Markets", "https://feeds.reuters.com/reuters/marketsNews"),
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ]
    rss_n = max(3, int(limit_each // 3))

    all_items: List[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_fetch_yahoo), ex.submit(_fetch_google)]
        for src, url2 in rss_sources:
            futs.append(ex.submit(_fetch_rss, src, url2, rss_n))
        for fu in futs:
            try:
                res = fu.result()
                if res:
                    all_items.extend(res)
            except Exception:
                continue

    # --- Dedup & sort ---
    seen = set()
    news_items: List[dict] = []
    for it in all_items:
        try:
            key = (str(it.get("title", "")).strip(), str(it.get("src", "")).strip())
            if not key[0]:
                continue
            if key in seen:
                continue
            seen.add(key)
            news_items.append(it)
        except Exception:
            continue

    news_items.sort(key=lambda x: int(x.get("pub", 0) or 0), reverse=True)

    # --- Build context + lightweight sentiment ---
    sentiment_score = 0
    meta: Dict[str, int] = {"yahoo": 0, "google": 0, "rss": 0, "pos": 0, "neg": 0}
    context_lines: List[str] = []

    now_ts = time.time()
    for it in news_items[:15]:
        title = str(it.get("title", "")).strip()
        pub = int(it.get("pub", 0) or 0)
        src = str(it.get("src", "")).strip() or "-"
        dt = datetime.fromtimestamp(pub).strftime("%Y/%m/%d") if pub else "-"
        context_lines.append(f"- [{src} {dt}] {title}")

        if src == "Yahoo":
            meta["yahoo"] += 1
        elif src == "Google":
            meta["google"] += 1
        else:
            meta["rss"] += 1

        weight = 2 if (pub and (now_ts - pub) < 172800) else 1
        if any(w in title for w in pos_words):
            sentiment_score += 1 * weight
            meta["pos"] += 1
        if any(w in title for w in neg_words):
            sentiment_score -= 1 * weight
            meta["neg"] += 1

    return news_items, "\n".join(context_lines), sentiment_score, meta

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
    
    # Model preference: lite for faster macro/sector agents, full flash for deeper reports.
    # Model routing: keep quality for deep analysis, but prefer speed for lightweight prompts.
    if prompt_key in {"market", "sector_debate_fast", "company_snapshot_jp"}:
        models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]
    else:
        models = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]
    if prompt_key in ("market", "sector_debate_fast", "company_snapshot_jp"):
        models = ["gemini-2.0-flash-lite", "gemini-2.0-flash"]
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

        出力フォーマット（タグ厳守。全体で750〜1150字目安）:
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
    elif prompt_key == "company_snapshot_jp":
        p = f"""
        現在: {today_str}
        銘柄:{context['name']} ({context['ticker']})
        企業概要:{context.get('overview','')}
        基礎データ:{context.get('fund_str','')}
        市場・セクター比較:{context.get('m_comp','')}
        株価動向:{context.get('price_action','')}
        ニュース:{context.get('news','')}
        Nonce:{context.get('nonce',0)}
        
        タスク:
        - プロ向けに「約200字（180〜220字）」の日本語スナップショットを1段落で作れ。
        - 事業の核 + 収益ドライバー + 足元の材料（ニュース/株価動向） + 今後3ヶ月の注目点を、因果で1〜2文に圧縮する。
        
        厳守:
        - 文体は「だ・である」。
        - 改行禁止、箇条書き禁止。
        - 記号(「**」や「""」)は禁止。
        - 与えた情報の範囲で推論し、新しい事実の創作はしない。
        
        出力: 本文のみ。
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
        分量: 850-1200字程度。冗長な言い換え禁止。各段落は新情報/新しい推論のみ。
        
        追加ルール（最重要）:
        - 最初の1行は必ず次の形式で出力せよ（改行禁止）:
          SNAPSHOT_JP: <日本語180〜220文字。事業の核 + 強み + 今後3ヶ月の注目点を1〜2文でまとめる。推測の断言は禁止。>
        - SNAPSHOT_JP行の直後に本文を続ける（空行禁止）。
        
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

    Security: model-provided content is HTML-escaped (no injection).
    Only our own minimal tags (<div>/<span>/<br>) are emitted.
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
    clean = clean_ai_text(str(text).replace("```html", "").replace("```", ""))
    parts = re.split(r'(\[[A-Z_]+\])', clean)

    buckets: Dict[str, str] = {}
    cur = None
    buf: List[str] = []
    for p in parts:
        if p in mapping:
            if cur is not None and buf:
                buckets[cur] = (buckets.get(cur, "") + "\n" + "".join(buf)).strip()
            cur = p
            buf = []
        else:
            buf.append(p)
    if cur is not None and buf:
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

    out_html = ""
    for tag in order:
        if tag not in buckets or not buckets[tag].strip():
            continue
        cls, label = mapping[tag]
        content = buckets[tag].strip()

        if tag == "[JUDGE]":
            # Judge should not show a separate 'Sector view:' block (avoid duplication)
            content = re.sub(r"(?mi)^Sector view:\s*.*?(\n\s*\n|\n(?=Stock pick:)|\Z)", "", content).strip()

        # Remove extra headings like 'トリガー:' / 'Triggers:'
        content = re.sub(r"(?mi)^(?:トリガー|トリガー\s*\(.*?\)|Triggers?)\s*[:：]\s*", "", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        # remove regex backref artifacts + SOH
        content = re.sub(r"(?m)^\s*(?:\\\\1|\x01)\s*", "", content)
        content = re.sub(r"(?m)^\s*\\\d+\s*", "", content)
        if tag == "[RISK]":
            content = _compress_bullets_in_views(content)
        content = content.replace("\\1", "").replace("\x01", "")

        # --- HTML sanitization: escape model content ---
        safe = html_lib.escape(content, quote=True)

        # Re-add our own safe emphasis (after escaping)
        safe = re.sub(r"(?m)^(Sector view:\s*)", r"<span class='subhead'>\1</span>", safe)
        safe = re.sub(r"(?m)^(Stock pick:\s*)", r"<span class='subhead'>\1</span>", safe)

        content_html = "<div class='agent-content'>" + safe.replace("\n", "<br>") + "</div>"

        if tag == "[SECTOR_OUTLOOK]":
            out_html += (
                f"<div class='{cls}' style='border-left:5px solid #00f2fe; margin-bottom:10px; padding:10px 12px;'>"
                f"<span class='orbitron' style='letter-spacing:0.8px; font-weight:900;'>{label}</span><br>{content_html}</div>"
            )
        elif tag == "[JUDGE]":
            out_html += (
                f"<div class='agent-row {cls}' style='margin-top:10px; padding:12px 14px; border:1px solid rgba(255,0,85,0.45); background: rgba(255,0,85,0.06);'>"
                f"<div class='agent-label' style='color:#ff0055; font-weight:800;'>{label}</div>{content_html}</div>"
            )
        else:
            out_html += f"<div class='agent-row {cls}'><div class='agent-label'>{label}</div>{content_html}</div>"

    return out_html
def run():
    # --- 1. INITIALIZE STATE ---
    if "system_logs" not in st.session_state: st.session_state.system_logs = []
    if "selected_sector" not in st.session_state: st.session_state.selected_sector = None
    if "last_market_key" not in st.session_state: st.session_state.last_market_key = None
    if "last_lookback_key" not in st.session_state: st.session_state.last_lookback_key = None
    if "ai_nonce" not in st.session_state: st.session_state.ai_nonce = 0
    if "ai_panels" not in st.session_state: st.session_state.ai_panels = {}
    if "peer_cache" not in st.session_state: st.session_state.peer_cache = {}
    if "news_panels" not in st.session_state: st.session_state.news_panels = {}
    if "fund_panels" not in st.session_state: st.session_state.fund_panels = {}
    if "jp_snapshots" not in st.session_state: st.session_state.jp_snapshots = {}

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
.snapshot-box{
  margin-top: 10px;
  margin-bottom: 10px;
  padding: 12px 14px;
  border: 1px solid rgba(0, 242, 254, 0.38);
  border-left: 4px solid #00f2fe;
  background: linear-gradient(135deg, rgba(0,242,254,0.12), rgba(255,0,85,0.06));
  box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset;
}
.snapshot-title{
  font-family:'Orbitron',sans-serif !important;
  font-weight: 900;
  letter-spacing: 0.10em;
  color: #00f2fe;
  font-size: 12px;
  margin-bottom: 6px;
}
.snapshot-text{
  font-family:'Zen Kaku Gothic New', sans-serif !important;
  font-size: var(--fz-body) !important;
  line-height: 1.8;
  color: #f2f2f2;
  overflow-wrap: anywhere;
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

/* Select readability (fix iOS/Safari black text after selection) */
div[data-baseweb="select"] *{
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}
div[data-baseweb="select"] input{
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  caret-color: var(--accent) !important;
}
div[data-baseweb="select"] svg{
  fill: var(--text) !important;
}
div[data-baseweb="select"] > div:hover{
  border-color: rgba(0,242,254,0.55) !important;
}
div[data-baseweb="menu"]{
  background: #0a0a0a !important;
}
div[data-baseweb="menu"] span{
  color: var(--text) !important;
}

/* Hint bars */
.hint-bar{
  font-family:'Orbitron',sans-serif;
  font-size: 11.5px;
  color: #dfeff2;
  background: rgba(0,242,254,0.06);
  border: 1px solid rgba(0,242,254,0.22);
  padding: 10px 12px;
  border-radius: 14px;
  margin: 6px 0 10px 0;
  line-height: 1.55;
}
.hint-bar b{ color:#00f2fe; }
.hint-bar .dim{ color:#9fb3b6; font-weight:700; }

/* Link buttons (HTML anchors) */
.link-btn{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  width: 100%;
  padding: 0.62rem 0.9rem;
  margin: 0.1rem 0;
  border-radius: 14px;
  border: 1px solid rgba(0,242,254,0.45);
  background: linear-gradient(90deg, rgba(0,242,254,0.22), rgba(255,0,85,0.14));
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  text-decoration: none !important;
  font-family: 'Orbitron', sans-serif !important;
  font-weight: 900 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase;
  box-shadow: 0 0 18px rgba(0,242,254,0.12) !important;
}
.link-btn:hover{
  transform: translateY(-1px);
  box-shadow: 0 0 24px rgba(0,242,254,0.22) !important;
}
.link-btn:active{
  transform: translateY(0px);
  box-shadow: 0 0 10px rgba(255,0,85,0.18) !important;
}
.link-arrow{ opacity:0.9; font-weight:900; }
@media (max-width: 640px){
  .link-btn{
    font-size: 12px;
    padding: 0.75rem 0.85rem;
  }
}

/* Notes */
.note-box{
  background: #0a0a0a;
  border: 1px solid #333;
  border-left: 3px solid #00f2fe;
  padding: 10px 12px;
  margin-top: 10px;
  margin-bottom: 10px;
  font-size: var(--fz-note) !important;
  line-height: 1.6;
  color: #eee;
  white-space: pre-wrap;
}
.note-box a{
  color: #00f2fe !important;
  text-decoration: none;
  border-bottom: 1px dotted rgba(0,242,254,0.65);
}
.note-box a:hover{
  text-decoration: underline;
}
.mini-note{
  font-size: var(--fz-note) !important;
  color: #bbb;
  margin-top: 6px;
  margin-bottom: 6px;
}
.mini-note a{
  color: #00f2fe !important;
  text-decoration: none;
}
.mini-note a:hover{
  text-decoration: underline;
}

</style>
""", unsafe_allow_html=True)
    
    st.markdown("<h1 class='brand'>ALPHALENS</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='dim' style='margin-top:-10px; margin-bottom:6px;'>Build: {BUILD_ID}</div>", unsafe_allow_html=True)
    
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

    st.markdown(
        "<div class='hint-bar'>"
        "<b>MARKET</b> selects the universe. <b>WINDOW</b> sets the scoring horizon. "
        "<span class='dim'>RELOAD MARKET DATA</span> refetches prices & recalculates tables. "
        "<span class='dim'>GENERATE AI INSIGHTS</span> refreshes AI narratives (cached by default).</div>",
        unsafe_allow_html=True,
    )


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


    # 1. Market Pulse (AI on-demand for fast startup)
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

    # Safe top/bottom sector names (avoid crash when sector data is missing)
    top_sector = sdf.iloc[-1]["Sector"] if (isinstance(sdf, pd.DataFrame) and (not sdf.empty) and "Sector" in sdf.columns) else "-"
    bot_sector = sdf.iloc[0]["Sector"] if (isinstance(sdf, pd.DataFrame) and (not sdf.empty) and "Sector" in sdf.columns) else "-"

    # --- Spread robustness: ensure defined in all paths ---
    try:
        spread = float(sdf['RS'].max() - sdf['RS'].min()) if (not sdf.empty and 'RS' in sdf.columns) else 0.0
    except Exception:
        spread = 0.0

    s_date = core_df.index[-win-1].strftime('%Y/%m/%d')
    e_date = core_df.index[-1].strftime('%Y/%m/%d')

    # Definition Header (ORDER FIXED: Spread -> Regime -> NewsSent)
    index_name = get_name(bench)
    index_label = f"{index_name} ({bench})" if index_name else bench

    # --- AI is on-demand: no headline fetch / no LLM call until user explicitly requests it ---
    mp_ctx_id = f"{market_key}|{lookback_key}|{bench_used}|{s_date}|{e_date}|{st.session_state.ai_nonce}"
    mp_saved = st.session_state.ai_panels.get(("market", mp_ctx_id), {}) if isinstance(st.session_state.get("ai_panels"), dict) else {}

    # Default header values (no news fetch yet)
    s_score = 0
    lbl = "Neutral"
    hit_pos = 0
    hit_neg = 0
    s_cls = "hl-neutral"

    # Default body (fast placeholder)
    mp_body_html = "<span class='dim'>AI narrative is <b>on-demand</b>. Open the panel below and click <b>GENERATE MARKET PULSE</b>.</span>"

    if isinstance(mp_saved, dict) and mp_saved.get("html"):
        mp_body_html = mp_saved.get("html", mp_body_html)
        s_score = int(mp_saved.get("s_score", s_score) or 0)
        lbl = str(mp_saved.get("lbl", lbl) or lbl)
        hit_pos = int(mp_saved.get("hit_pos", hit_pos) or 0)
        hit_neg = int(mp_saved.get("hit_neg", hit_neg) or 0)
        s_cls = str(mp_saved.get("s_cls", s_cls) or s_cls)

    st.markdown(
        f"""
        <div class='market-box'>
          <b class='orbitron'>MARKET PULSE ({s_date} - {e_date})</b><br>
          <span class='caption-text'>Spread: {spread:.1f}pt | Regime: {regime} | NewsSent: <span class='{s_cls}'>{s_score:+d}</span> ({lbl}) [Hit:{hit_pos}/{hit_neg}]</span><br>
          {mp_body_html}
          <div class='def-text'>
            <b>DEFINITIONS</b> |
            <b>Spread</b>: セクターRSの最大−最小(pt)。市場内の勝ち負けがどれだけ鮮明かを示す |
            <b>Regime</b>: 200DMA判定（終値&gt;200DMA=Bull / 終値&lt;200DMA=Bear） |
            <b>NewsSent</b>: 見出しキーワード命中（pos=+1/neg=−1）合計を−10〜+10にクリップ |
            <b>RS</b>: 相対リターン差(pt)=セクター(or銘柄)リターン−指数リターン（指数名は本文に明記）
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🧠 MARKET PULSE — AI Narrative (on-demand)", expanded=False):
        st.markdown(
            "<div class='hint-bar'>"
            "This panel fetches headlines and generates the Market Pulse narrative only when requested. "
            "<span class='dim'>Tip:</span> Use <b>✨ GENERATE AI INSIGHTS</b> above to force-refresh (nonce)."
            "</div>",
            unsafe_allow_html=True,
        )
        gen_mp = st.button(
            "GENERATE MARKET PULSE",
            key=f"gen_market_pulse_{market_key}_{lookback_key}_{bench_used}",
            type="primary",
            use_container_width=True,
        )
        if gen_mp:
            with st.spinner("Generating Market Pulse..."):
                # Headlines + lightweight sentiment (cached 30m)
                try:
                    _, market_context, m_sent, m_meta = get_news_consolidated(bench, m_cfg["name"], market_key)
                except Exception:
                    market_context, m_sent, m_meta = "", 0, {}
                try:
                    s_score2 = int(np.clip(int(round(float(m_sent or 0))), -10, 10))
                except Exception:
                    s_score2 = 0
                lbl2 = "Positive" if s_score2 > 0 else ("Negative" if s_score2 < 0 else "Neutral")
                hit_pos2 = int((m_meta or {}).get("pos", 0))
                hit_neg2 = int((m_meta or {}).get("neg", 0))
                s_cls2 = "hl-pos" if s_score2 > 0 else ("hl-neg" if s_score2 < 0 else "hl-neutral")

                raw_mp = generate_ai_content("market", {
                    "s_date": s_date, "e_date": e_date, "ret": b_stats["Ret"],
                    "top": top_sector, "bot": bot_sector,
                    "market_name": m_cfg["name"],
                    "headlines": (market_context[:900] if isinstance(market_context, str) else market_context),
                    "slot_line": ", ".join(outlook_date_slots()),
                    "index_label": index_label,
                    "nonce": st.session_state.ai_nonce
                })
                raw_mp = enforce_index_naming(raw_mp, index_label)
                raw_mp = enforce_market_format(raw_mp)
                raw_mp = force_nonempty_outlook_market(raw_mp, regime, b_stats["Ret"], spread, market_key)
                mp_html2 = market_to_html(raw_mp)

                # Store for this context (no heavy work unless requested)
                if isinstance(st.session_state.get("ai_panels"), dict):
                    st.session_state.ai_panels[("market", mp_ctx_id)] = {
                        "html": mp_html2,
                        "s_score": s_score2,
                        "lbl": lbl2,
                        "hit_pos": hit_pos2,
                        "hit_neg": hit_neg2,
                        "s_cls": s_cls2,
                    }
                st.toast("✅ Market Pulse updated", icon="✅")
                st.rerun()
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

    sector_ticker = m_cfg.get("sectors", {}).get(target_sector)
    full_list = list(dict.fromkeys([bench] + ([sector_ticker] if sector_ticker else []) + stock_list))
    cache_key = f"{market_key}_{target_sector}_{lookback_key}"
    
    if cache_key != st.session_state.get("sec_cache_key") or refresh_prices:
        with st.spinner(f"ANALYZING {len(stock_list)} STOCKS..."):
            raw_s = fetch_market_data(tuple(full_list), FETCH_PERIOD)
            st.session_state.sec_df = extract_close_prices(raw_s, full_list)
            st.session_state.sec_cache_key = cache_key
            
    sec_df = st.session_state.sec_df
    s_audit = audit_data_availability(full_list, sec_df, win)
    
    results = []
    # Pre-compute sector return for sector-relative strength (RS_Sec)
    sector_ret = np.nan
    try:
        if sector_ticker and sector_ticker in sec_df.columns:
            sec_series = sec_df[sector_ticker].dropna()
            if len(sec_series) >= win + 1:
                sec_win = sec_df[sector_ticker].ffill().tail(win+1)
                if not sec_win.isna().iloc[0]:
                    sector_ret = float((sec_win.iloc[-1] / sec_win.iloc[0] - 1) * 100)
    except Exception:
        sector_ret = np.nan

    for t in [x for x in s_audit["list"] if x not in {bench, sector_ticker}]:
        stats = calc_technical_metrics(sec_df[t], sec_df[bench], win)
        if stats:
            # RS_Sec: stock return minus sector return over the same window
            try:
                stats["RS_Sec"] = float(stats.get("Ret", np.nan) - sector_ret) if pd.notna(sector_ret) else np.nan
            except Exception:
                stats["RS_Sec"] = np.nan
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
            f"{r['Name']}({r['Ticker']}): Ret {r['Ret']:.1f}%, RS(Mkt) {r['RS']:.2f}, "
            f"RS(Sec) {dash(r.get('RS_Sec'), '%.2f')}, Accel {r['Accel']:.2f}, HighDist {r['HighDist']:.1f}%, "
            f"MCap {format_mcap(f.get('MCap',0), f.get('Currency'))}, PER {dash(f.get('PER'))}, PBR {dash(f.get('PBR'))}"
        )
    if not neg.empty:
        nr = neg.iloc[0]
        f = pick_fund_row(cand_fund, nr["Ticker"])
        cand_lines.append(f"\n[AVOID] {nr['Name']}: Ret {nr['Ret']:.1f}%, RS {nr['RS']:.2f}, PER {dash(f.get('PER'))}")

    # --- AI Sector Council is on-demand (keeps sector scan fast) ---
    sec_items, sec_context = [], ""  # placeholder; real fetch happens only when user requests AI

    # Sector Stats
    try:
        med_rs_sec = df['RS_Sec'].median() if 'RS_Sec' in df.columns else np.nan
    except Exception:
        med_rs_sec = np.nan
    sec_ret_disp = f"{sector_ret:+.1f}%" if pd.notna(sector_ret) else "-"
    sector_stats = (
        f"Universe:{len(stock_list)} Computable:{len(df)} "
        f"MedianRS(Mkt):{df['RS'].median():.2f} "
        + (f"MedianRS(Sec):{med_rs_sec:.2f} " if pd.notna(med_rs_sec) else "")
        + f"MedianRet:{df['Ret'].median():.1f}% "
        + f"SpreadRS(Mkt):{(df['RS'].max()-df['RS'].min()):.2f} "
        + (f"Sector({sector_ticker})Ret:{sec_ret_disp}" if sector_ticker else "")
    )

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

    # Top pick line (for display & context)
    tp = df.iloc[0]
    tp_f = pick_fund_row(cand_fund, tp["Ticker"])
    top_line = (
        f"[TOP] {tp['Name']} ({tp['Ticker']}): Ret {tp['Ret']:.1f}%, RS(Mkt) {tp['RS']:.2f}, "
        f"RS(Sec) {dash(tp.get('RS_Sec'), '%.2f')}, Accel {tp['Accel']:.2f}, "
        f"HighDist {tp['HighDist']:.1f}%, MaxDD {tp['MaxDD']:.1f}%, "
        f"MCap {format_mcap(tp_f.get('MCap',0), tp_f.get('Currency'))}, PER {dash(tp_f.get('PER'))}, PBR {dash(tp_f.get('PBR'))}"
    )

    # --- Render panel (cached display) ---
    sec_ai_ctx_id = f"{market_key}|{target_sector}|{lookback_key}|{st.session_state.ai_nonce}"
    sec_saved = st.session_state.ai_panels.get(("sector", sec_ai_ctx_id), {}) if isinstance(st.session_state.get("ai_panels"), dict) else {}
    sec_box_body = "<span class='dim'>AI council is <b>on-demand</b>. Open the panel below and click <b>GENERATE SECTOR COUNCIL</b>.</span>"
    sec_ai_raw_cached = ""
    if isinstance(sec_saved, dict) and sec_saved.get("html"):
        sec_box_body = sec_saved.get("html", sec_box_body)
        sec_ai_raw_cached = str(sec_saved.get("raw", "") or "")

    st.markdown(f"<div class='report-box'><b>🦅 🤖 AI AGENT SECTOR REPORT</b><br>{sec_box_body}</div>", unsafe_allow_html=True)

    if sec_ai_raw_cached:
        st.download_button(
            "DOWNLOAD COUNCIL LOG",
            sec_ai_raw_cached,
            f"council_log_target_sector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

    with st.expander("🦅 🤖 AI AGENT SECTOR REPORT — Generate (on-demand)", expanded=False):
        st.markdown(
            "<div class='hint-bar'>"
            "Runs a 5-agent council for the selected sector. "
            "Fetches a small headline set and generates the narrative only when requested."
            "</div>",
            unsafe_allow_html=True,
        )
        gen_sec = st.button(
            "GENERATE SECTOR COUNCIL",
            key=f"gen_sector_council_{market_key}_{target_sector}_{lookback_key}",
            type="primary",
            use_container_width=True,
        )
        if gen_sec:
            with st.spinner("Running sector council..."):
                try:
                    sec_items2, sec_context2, _, _ = get_news_consolidated(
                        m_cfg["sectors"][target_sector], target_sector, market_key, limit_each=3
                    )
                except Exception:
                    sec_items2, sec_context2 = [], ""

                sec_news_str = (sec_context2[:1500] if isinstance(sec_context2, str) else _news_to_str(sec_items2, limit=6))

                sec_ai_raw2 = generate_ai_content("sector_debate_fast", {
                    "sec": target_sector,
                    "market_name": m_cfg.get("name", str(market_key)),
                    "sector_stats": sector_stats_str,
                    "top": top3_str,
                    "news": sec_news_str,
                    "nonce": st.session_state.ai_nonce
                })
                sec_ai_txt2 = quality_gate_text(enforce_da_dearu_soft(sec_ai_raw2), enable=st.session_state.get('qc_on', True))
                if ("[FUNDAMENTAL]" in sec_ai_txt2 or "[SECTOR_OUTLOOK]" in sec_ai_txt2):
                    sec_ai_html2 = parse_agent_debate(sec_ai_txt2)
                else:
                    sec_ai_html2 = text_to_safe_html(sec_ai_txt2)

                if isinstance(st.session_state.get("ai_panels"), dict):
                    st.session_state.ai_panels[("sector", sec_ai_ctx_id)] = {"raw": sec_ai_raw2, "html": sec_ai_html2}

                st.toast("✅ Sector council updated", icon="✅")
                st.rerun()
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
    
    cols = ['Name', 'Ticker', 'Apex', 'RS', 'RS_Sec', 'Accel', 'Ret', '1M', '3M', 'HighDist', 'MaxDD', 'PER', 'PBR', 'ROE', 'RevGrow', 'OpMargin', 'Beta']
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
    def fmt_mcap_row(mcap, cur):
        return format_mcap(mcap, cur)
    
    df["MCapDisp"] = [format_mcap(m, c) for m, c in zip(df.get('MCap', [np.nan]*len(df)), df.get('Currency', [None]*len(df)))]
    
    df_disp = df.copy()
    for c in ["PER", "PBR"]: df_disp[c] = df_disp[c].apply(lambda x: dash(x))
    for c in ["ROE", "RevGrow", "OpMargin"]: df_disp[c] = df_disp[c].apply(pct)
    df_disp["Beta"] = df_disp["Beta"].apply(lambda x: dash(x, "%.2f"))

    df_sorted = df_disp.sort_values("MCap", ascending=False)
    
    st.markdown("<div class='action-call'>👇 Select ONE stock to generate the AI agents' analysis note below</div>", unsafe_allow_html=True)
    event = st.dataframe(
        df_sorted[["Name", "Ticker", "MCapDisp", "ROE", "RevGrow", "PER", "PBR", "Apex", "RS", "RS_Sec", "1M", "12M"]],
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
        "DEFINITIONS | Apex: 0.45*z(3M)+0.35*z(1M)+0.15*z(RS)+0.05*z(Accel)（zscore合成） | "
        "RS: Ret(銘柄)−Ret(市場平均) | RS_Sec: Ret(銘柄)−Ret(セクターETF) | Accel: 直近半期間リターン−(全期間リターン/2) | "
        "HighDist: 直近価格の52週高値からの乖離(%) | MaxDD: 期間内最大ドローダウン(%)（負値。0に近いほど良い） | "
        "MCap: 通貨付き（例 USD 626.0B） | PER/PBR/ROE等: yfinance.Ticker().info（負のPER/PBRは除外、欠損は'-'）"
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
    
    # --- NEWS + AI are on-demand (fast initial view) ---
    stock_ai_ctx_id = f"{market_key}|{target_sector}|{lookback_key}|{top['Ticker']}|{st.session_state.ai_nonce}"
    saved_report = st.session_state.ai_panels.get(("stock", stock_ai_ctx_id), {}) if isinstance(st.session_state.get("ai_panels"), dict) else {}

    news_ctx_id = f"{market_key}|{top['Ticker']}"
    saved_news = st.session_state.news_panels.get(news_ctx_id, {}) if isinstance(st.session_state.get("news_panels"), dict) else {}
    news_items = saved_news.get("items", []) if isinstance(saved_news, dict) else []
    news_context = saved_news.get("context", "") if isinstance(saved_news, dict) else ""

    # Fundamentals are fully on-demand (no yfinance.info call on selection)
    ticker_sel = str(top["Ticker"])
    # Light fundamentals from the table (no network)
    fund_data = {
        "Ticker": ticker_sel,
        "Name": (top.get("Name") if hasattr(top, "get") else ticker_sel),
        "Sector": (top.get("Sector", "-") if hasattr(top, "get") else "-"),
        "Industry": (top.get("Industry", "-") if hasattr(top, "get") else "-"),
        "MCap": (top.get("MCap", np.nan) if hasattr(top, "get") else np.nan),
        "Currency": (top.get("Currency") if hasattr(top, "get") else None),
        "PER": (top.get("PER", np.nan) if hasattr(top, "get") else np.nan),
        "PBR": (top.get("PBR", np.nan) if hasattr(top, "get") else np.nan),
        "PEG": (top.get("PEG", np.nan) if hasattr(top, "get") else np.nan),
        "Target": (top.get("Target", np.nan) if hasattr(top, "get") else np.nan),
        "Website": None,
        "Summary": "",
    }
    # Merge on-demand loaded fundamentals (if any)
    fund_extra = st.session_state.fund_panels.get(ticker_sel, {}) if isinstance(st.session_state.get("fund_panels"), dict) else {}
    if isinstance(fund_extra, dict) and fund_extra:
        # Prefer non-empty values from extra
        for k, v in fund_extra.items():
            if v not in (None, "", "-", "nan"):
                fund_data[k] = v

    # --- Company Overview (display vs AI context) ---
    sec_name = str(fund_data.get("Sector") or "-")
    ind_name = str(fund_data.get("Industry") or "-")
    cur = fund_data.get("Currency")
    mcap = fund_data.get("MCap") or fund_data.get("MarketCap") or np.nan
    mcap_disp = format_mcap(mcap, cur)

    website = fund_data.get("Website") or ""
    website_url = _safe_http_url(website)
    if website_url:
        website_disp = website_url.replace("https://", "").replace("http://", "")
        escaped_url = html_lib.escape(website_url, quote=True)
        escaped_disp = html_lib.escape(website_disp, quote=True)
        website_html = f'<a href="{escaped_url}" target="_blank" rel="noopener noreferrer">{escaped_disp} <span class="link-arrow">↗</span></a>'
    else:
        website_html = "<span class='dim'>— (load fundamentals)</span>"

    # Keep summary ONLY as hidden AI context (not shown as a duplicated UI block)
    bsum_raw = str(fund_data.get("BusinessSummary") or fund_data.get("Summary") or "").strip()
    bsum_raw = re.sub(r"\s+", " ", bsum_raw).strip()
    bsum_src = (bsum_raw[:1200] if bsum_raw else "")
    bsum = bsum_src
    if len(bsum) > 500:
        bsum = bsum[:500].rstrip() + "…"

    overview_for_ai = f"Sector:{sec_name} | Industry:{ind_name} | MCap:{mcap_disp} | Summary:{bsum if bsum else '-'}"
    overview_display = f"Sector: {sec_name} | Industry: {ind_name} | Market Cap: {mcap_disp}"

    overview_display_html = html_lib.escape(str(overview_display), quote=True)

    st.markdown(
        f"<div class='note-box'><b>Company Overview</b><br>{overview_display_html}<br>Website: {website_html}</div>",
        unsafe_allow_html=True,
    )

    # Fundamentals loader (on-demand)
    with st.expander("📌 FUNDAMENTALS — Load (on-demand)", expanded=False):
        st.markdown(
            "<div class='hint-bar'>Loads <b>yfinance.Ticker().info</b> only when requested (keeps selection snappy). "
            "After loading, the Website line becomes a direct clickable link.</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "LOAD FUNDAMENTALS (yfinance.info)",
            key=f"load_fund_{market_key}_{target_sector}_{lookback_key}_{ticker_sel}",
            use_container_width=True,
        ):
            with st.spinner("Loading fundamentals..."):
                try:
                    fd = get_fundamental_data(ticker_sel)
                except Exception:
                    fd = {}
                if isinstance(st.session_state.get("fund_panels"), dict):
                    st.session_state.fund_panels[ticker_sel] = fd
            st.toast("✅ Fundamentals loaded", icon="✅")
            st.rerun()

    # --- External chart links ---
    try:
        def _tv_symbol(tk: str) -> str:
            if tk.endswith(".T"):
                return "TSE:" + tk.replace(".T","")
            if tk.startswith("^"):
                return tk.replace("^","")
            return tk
        yf_url = f"https://finance.yahoo.com/quote/{top['Ticker']}"
        tv_url = f"https://www.tradingview.com/symbols/{_tv_symbol(top['Ticker'])}/"
        yf_url_e = html_lib.escape(str(yf_url), quote=True)
        tv_url_e = html_lib.escape(str(tv_url), quote=True)
        st.markdown(f"<div class='mini-note'>Charts: <a href='{yf_url_e}' target='_blank'>Yahoo Finance</a> | <a href='{tv_url_e}' target='_blank'>TradingView</a></div>", unsafe_allow_html=True)
    except Exception:
        pass

    ed = "-"  # Earnings date is fetched on-demand during AI generation
    bench_fd = {}  # Benchmark fundamentals are fetched on-demand during AI generation

    # Price Action Pack
    pa = {}
    try:
        if "sec_df" in st.session_state and top["Ticker"] in st.session_state.sec_df.columns:
            pa = price_action_pack(st.session_state.sec_df[top["Ticker"]])
    except Exception:
        pa = {}

    price_act = ""
    if pa:
        price_act = (
            f"Last {pa.get('Last',np.nan):.2f} | 1D {pa.get('1D',np.nan):+.2f}% | 1W {pa.get('1W',np.nan):+.2f}% | "
            f"1M {pa.get('1M',np.nan):+.2f}% | 3M {pa.get('3M',np.nan):+.2f}% | 200DMA {pa.get('200DMA_Dist',np.nan):+.1f}% | "
            f"MaxDD(6M) {pa.get('MaxDD_6M',np.nan):.1f}%"
        )
    price_act_html = html_lib.escape(str(price_act), quote=True)
    st.markdown(f"<div class='kpi-strip mono'>{price_act_html}</div>", unsafe_allow_html=True)

    bench_per = dash(bench_fd.get("PER"))
    sector_per = dash(pd.to_numeric(df["PER"], errors="coerce").median())
    stock_per = dash(fund_data.get("PER"))
    rs_mkt = dash(top.get("RS"), "%.2f")
    rs_sec = dash(top.get("RS_Sec"), "%.2f")
    m_comp = f"市場平均PER: {bench_per}倍 / セクター中央値PER: {sector_per}倍 / 当該銘柄PER: {stock_per}倍 | RS(Mkt): {rs_mkt} | RS(Sec): {rs_sec}"

    fund_str = f"PER:{stock_per}, PBR:{dash(fund_data.get('PBR'))}, PEG:{dash(fund_data.get('PEG'))}, Target:{dash(fund_data.get('Target'))}"

    # --- Read cached AI report if available ---
    snap_jp = str(saved_report.get("snap", "") or "") if isinstance(saved_report, dict) else ""
    if not snap_jp:
        try:
            snap_jp = str(st.session_state.jp_snapshots.get(ticker_sel, {}).get("text", "") or "")
        except Exception:
            snap_jp = ""
    report_body = str(saved_report.get("body", "") or "") if isinstance(saved_report, dict) else ""
    analyst_note_txt = str(saved_report.get("analyst_note", "") or "") if isinstance(saved_report, dict) else ""
    generated_at = str(saved_report.get("generated_at", "") or "") if isinstance(saved_report, dict) else ""

    # UI
    nc1, nc2 = st.columns([1.5, 1])
    with nc1:
        # --- JP Company Snapshot (embedded at the top of AI EQUITY BRIEFING) ---
        if snap_jp:
            snap_disp = quality_gate_text(enforce_da_dearu_soft(snap_jp), enable=st.session_state.get('qc_on', True))
            snapshot_inline_html = (
                "<div class='snapshot-box'><div class='snapshot-title'>AI SNAPSHOT (JP)</div>"
                f"<div class='snapshot-text'>{text_to_safe_html(snap_disp)}</div></div>"
            )
        else:
            snapshot_inline_html = (
                "<div class='snapshot-box'><div class='snapshot-title'>AI SNAPSHOT (JP)</div>"
                "<div class='dim'>Not generated yet. Open the panel below and click <b>GENERATE JP SNAPSHOT</b>.</div></div>"
            )

        with st.expander("🇯🇵 AI COMPANY SNAPSHOT (JP) — Generate (on-demand)", expanded=False):
            st.markdown(
                "<div class='hint-bar'>"
                "Generates a tight Japanese snapshot (~200 chars): core business, near-term catalyst, and 3‑month watchpoints. "
                "Runs <b>only when requested</b> to keep the initial load fast."
                "</div>",
                unsafe_allow_html=True,
            )
            include_news_snap = st.checkbox(
                "Include fresh news (slower)",
                value=False,
                key=f"snap_news_{market_key}_{target_sector}_{lookback_key}_{ticker_sel}",
            )
            gen_snap = st.button(
                "GENERATE JP SNAPSHOT (~200 chars)",
                key=f"gen_snap_{market_key}_{target_sector}_{lookback_key}_{ticker_sel}",
                use_container_width=True,
            )
            if gen_snap:
                with st.spinner("Generating JP snapshot..."):
                    # 1) Load fundamentals on-demand (for Summary/Website/Industry)
                    try:
                        fd_full = get_fundamental_data(ticker_sel)
                    except Exception:
                        fd_full = {}
                    if isinstance(st.session_state.get("fund_panels"), dict) and isinstance(fd_full, dict):
                        st.session_state.fund_panels[ticker_sel] = fd_full

                    sec_name2 = str((fd_full.get("Sector") if isinstance(fd_full, dict) else None) or sec_name)
                    ind_name2 = str((fd_full.get("Industry") if isinstance(fd_full, dict) else None) or ind_name)
                    cur2 = (fd_full.get("Currency") if isinstance(fd_full, dict) else None) or cur
                    mcap2 = (fd_full.get("MCap") if isinstance(fd_full, dict) else None) or (fd_full.get("MarketCap") if isinstance(fd_full, dict) else None) or mcap
                    mcap_disp2 = format_mcap(mcap2, cur2)

                    bsum_raw2 = str((fd_full.get("BusinessSummary") if isinstance(fd_full, dict) else None) or (fd_full.get("Summary") if isinstance(fd_full, dict) else None) or "").strip()
                    bsum_raw2 = re.sub(r"\s+", " ", bsum_raw2).strip()
                    bsum2 = bsum_raw2
                    if len(bsum2) > 500:
                        bsum2 = bsum2[:500].rstrip() + "…"

                    overview_for_ai2 = f"Sector:{sec_name2} | Industry:{ind_name2} | MCap:{mcap_disp2} | Summary:{bsum2 if bsum2 else '-'}"

                    # Light valuation string (avoid extra network; benchmark PER omitted here)
                    per_val = (fd_full.get("PER") if isinstance(fd_full, dict) else None)
                    if pd.isna(per_val):
                        per_val = fund_data.get("PER")
                    pbr_val = (fd_full.get("PBR") if isinstance(fd_full, dict) else None)
                    if pd.isna(pbr_val):
                        pbr_val = fund_data.get("PBR")
                    peg_val = (fd_full.get("PEG") if isinstance(fd_full, dict) else np.nan)
                    target_val = (fd_full.get("Target") if isinstance(fd_full, dict) else np.nan)
                    stock_per2 = dash(per_val)
                    stock_pbr2 = dash(pbr_val)
                    fund_str2 = f"PER:{stock_per2}, PBR:{stock_pbr2}, PEG:{dash(peg_val)}, Target:{dash(target_val)}"

                    sector_per2 = dash(pd.to_numeric(df.get("PER"), errors="coerce").median()) if isinstance(df, pd.DataFrame) else "-"
                    rs_mkt2 = dash(top.get("RS"), "%.2f")
                    rs_sec2 = dash(top.get("RS_Sec"), "%.2f")
                    m_comp2 = f"セクター中央値PER: {sector_per2}倍 / 当該銘柄PER: {stock_per2}倍 | RS(Mkt): {rs_mkt2} | RS(Sec): {rs_sec2}"

                    # 2) News (optional)
                    news_for_snap = news_context
                    if include_news_snap:
                        try:
                            news_items3, news_context3, _, _ = get_news_consolidated(ticker_sel, top["Name"], market_key, limit_each=10)
                            news_for_snap = news_context3
                            if isinstance(st.session_state.get("news_panels"), dict):
                                st.session_state.news_panels[news_ctx_id] = {"items": news_items3, "context": news_context3}
                        except Exception:
                            pass

                    # 3) LLM call (lite model by routing)
                    snap_txt = generate_ai_content("company_snapshot_jp", {
                        "name": top["Name"], "ticker": ticker_sel,
                        "overview": overview_for_ai2,
                        "fund_str": fund_str2,
                        "m_comp": m_comp2,
                        "news": (news_for_snap[:1800] if isinstance(news_for_snap, str) else news_for_snap),
                        "price_action": price_act,
                        "nonce": st.session_state.ai_nonce,
                    })
                    snap_txt = clean_ai_text(snap_txt)
                    snap_txt = re.sub(r"\s+", " ", str(snap_txt)).strip()
                    if snap_txt and len(snap_txt) > 220:
                        snap_txt = snap_txt[:219] + "…"

                    if isinstance(st.session_state.get("jp_snapshots"), dict):
                        st.session_state.jp_snapshots[ticker_sel] = {
                            "text": str(snap_txt or ""),
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                    st.toast("✅ JP snapshot updated", icon="✅")
                    st.rerun()

        # --- AI equity briefing ---
        if report_body:
            report_txt_disp = quality_gate_text(enforce_da_dearu_soft(report_body), enable=st.session_state.get('qc_on', True))
            report_html = text_to_safe_html(report_txt_disp)
            gen_stamp = f"<div class='dim' style='margin-top:6px;'>Generated: {html_lib.escape(generated_at, quote=True)}</div>" if generated_at else ""
            st.markdown(f"<div class='report-box'><b>AI EQUITY BRIEFING</b><br>{snapshot_inline_html}{report_html}{gen_stamp}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='report-box'><b>AI EQUITY BRIEFING</b><br>"
                f"{snapshot_inline_html}"
                "<span class='dim'>AI briefing is <b>on-demand</b>. Open the panel below and click <b>GENERATE AI EQUITY BRIEFING</b>.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("🦅 🤖 AI EQUITY BRIEFING — Generate (on-demand)", expanded=False):
            st.markdown(
                "<div class='hint-bar'>"
                "Generates the analyst memo only when requested (keeps initial load fast). "
                "It also refreshes the integrated news cache for this ticker."
                "</div>",
                unsafe_allow_html=True,
            )
            gen_stock = st.button(
                "GENERATE AI EQUITY BRIEFING",
                key=f"gen_stock_report_{market_key}_{target_sector}_{lookback_key}_{top['Ticker']}",
                type="primary",
                use_container_width=True,
            )
            if gen_stock:
                with st.spinner("Generating analyst briefing..."):
                    # 0) Fundamentals + earnings date (on-demand)
                    ticker_sel2 = str(top["Ticker"])
                    try:
                        fd_full = get_fundamental_data(ticker_sel2)
                    except Exception:
                        fd_full = {}
                    if isinstance(st.session_state.get("fund_panels"), dict) and isinstance(fd_full, dict):
                        st.session_state.fund_panels[ticker_sel2] = fd_full

                    # Earnings date (best-effort; cached)
                    try:
                        ed2 = fetch_earnings_dates(ticker_sel2).get("EarningsDate", "-")
                    except Exception:
                        ed2 = "-"

                    # Benchmark PER (best-effort; cached)
                    bench_per2 = "-"
                    try:
                        bfd = get_fundamental_data(bench)
                        bench_per2 = dash(bfd.get("PER"))
                    except Exception:
                        bench_per2 = "-"

                    sec_name2 = str((fd_full.get("Sector") if isinstance(fd_full, dict) else None) or sec_name)
                    ind_name2 = str((fd_full.get("Industry") if isinstance(fd_full, dict) else None) or ind_name)
                    cur2 = (fd_full.get("Currency") if isinstance(fd_full, dict) else None) or cur
                    mcap2 = (fd_full.get("MCap") if isinstance(fd_full, dict) else None) or (fd_full.get("MarketCap") if isinstance(fd_full, dict) else None) or mcap
                    mcap_disp2 = format_mcap(mcap2, cur2)
                    website2 = (fd_full.get("Website") if isinstance(fd_full, dict) else None) or (fund_data.get("Website") or "")

                    bsum_raw2 = str((fd_full.get("BusinessSummary") if isinstance(fd_full, dict) else None) or (fd_full.get("Summary") if isinstance(fd_full, dict) else None) or "").strip()
                    bsum_raw2 = re.sub(r"\s+", " ", bsum_raw2).strip()
                    bsum2 = bsum_raw2
                    if len(bsum2) > 500:
                        bsum2 = bsum2[:500].rstrip() + "…" 
                    overview_for_ai2 = f"Sector:{sec_name2} | Industry:{ind_name2} | MCap:{mcap_disp2} | Summary:{bsum2 if bsum2 else '-'}"

                    per_val2 = (fd_full.get("PER") if isinstance(fd_full, dict) else None)
                    if pd.isna(per_val2):
                        per_val2 = fund_data.get("PER")
                    pbr_val2 = (fd_full.get("PBR") if isinstance(fd_full, dict) else None)
                    if pd.isna(pbr_val2):
                        pbr_val2 = fund_data.get("PBR")
                    peg_val2 = (fd_full.get("PEG") if isinstance(fd_full, dict) else np.nan)
                    target_val2 = (fd_full.get("Target") if isinstance(fd_full, dict) else np.nan)
                    stock_per2 = dash(per_val2)
                    sector_per2 = dash(pd.to_numeric(df.get("PER"), errors="coerce").median()) if isinstance(df, pd.DataFrame) else "-"
                    rs_mkt2 = dash(top.get("RS"), "%.2f")
                    rs_sec2 = dash(top.get("RS_Sec"), "%.2f")
                    m_comp2 = f"市場平均PER: {bench_per2}倍 / セクター中央値PER: {sector_per2}倍 / 当該銘柄PER: {stock_per2}倍 | RS(Mkt): {rs_mkt2} | RS(Sec): {rs_sec2}"
                    fund_str2 = f"PER:{stock_per2}, PBR:{dash(pbr_val2)}, PEG:{dash(peg_val2)}, Target:{dash(target_val2)}"

                    # 1) Fetch news (cached 30m)
                    try:
                        news_items2, news_context2, _, _ = get_news_consolidated(top["Ticker"], top["Name"], market_key, limit_each=10)
                    except Exception:
                        news_items2, news_context2 = [], ""
                    if isinstance(st.session_state.get("news_panels"), dict):
                        st.session_state.news_panels[news_ctx_id] = {"items": news_items2, "context": news_context2}

                    # 2) Main report
                    report_txt2 = generate_ai_content("stock_report", {
                        "name": top["Name"], "ticker": top["Ticker"],
                        "overview": overview_for_ai2,
                        "fund_str": fund_str2,
                        "m_comp": m_comp2,
                        "news": (news_context2[:1800] if isinstance(news_context2, str) else news_context2),
                        "earnings_date": ed2,
                        "price_action": price_act,
                        "nonce": st.session_state.ai_nonce
                    })
                    report_txt2 = clean_ai_text(report_txt2)

                    # Extract a compact JP snapshot (first line) from the report, and keep the body clean.
                    snap_jp2, report_body2 = split_snapshot_jp(report_txt2)
                    report_body2 = str(report_body2 or "").strip()

                    # Fallback: if the model did not emit SNAPSHOT_JP, generate snapshot separately (rare).
                    if not snap_jp2:
                        try:
                            snap_jp2 = generate_ai_content("company_snapshot_jp", {
                                "name": top["Name"], "ticker": top["Ticker"],
                                "overview": overview_for_ai2,
                                "fund_str": fund_str2,
                                "m_comp": m_comp2,
                                "news": (news_context2[:1800] if isinstance(news_context2, str) else news_context2),
                                "price_action": price_act,
                                "nonce": st.session_state.ai_nonce
                            })
                            snap_jp2 = clean_ai_text(snap_jp2)
                            snap_jp2 = re.sub(r"\s+", " ", str(snap_jp2)).strip()
                        except Exception:
                            snap_jp2 = ""

                    # Hard cap (never exceed 220 chars in UI)
                    try:
                        if snap_jp2 and len(snap_jp2) > 220:
                            snap_jp2 = snap_jp2[:219] + "…"
                    except Exception:
                        pass

                    # Build downloadable analyst note (no duplicated Summary block in the header)
                    analyst_note_txt2 = (
                        "Company Overview\n"
                        + f"Name: {top['Name']} ({top['Ticker']})\n"
                        + f"Sector: {sec_name2}\nIndustry: {ind_name2}\nMarket Cap: {mcap_disp2}\nWebsite: {website2}\n\n"
                        + ("AI Snapshot (JP)\n" + str(snap_jp2) + "\n\n" if snap_jp2 else "")
                        + "Quantitative Summary\n"
                        + fund_str2
                        + "\n\n"
                        + report_body2
                    )


                    # Persist snapshot separately so it is always visible at the top
                    if snap_jp2 and isinstance(st.session_state.get("jp_snapshots"), dict):
                        st.session_state.jp_snapshots[ticker_sel2] = {
                            "text": str(snap_jp2),
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                    if isinstance(st.session_state.get("ai_panels"), dict):
                        st.session_state.ai_panels[("stock", stock_ai_ctx_id)] = {
                            "snap": str(snap_jp2 or ""),
                            "body": report_body2,
                            "analyst_note": analyst_note_txt2,
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }

                    st.toast("✅ Equity briefing updated", icon="✅")
                    st.rerun()

        # Links (always available)
        st.markdown(
            "<div class='hint-bar'>Select buttons to open a new tab: <b>OFFICIAL</b> (company site), <b>IR SEARCH</b>, <b>EARNINGS DECK</b>.</div>",
            unsafe_allow_html=True,
        )
        links = build_ir_links(top["Name"], top["Ticker"], fund_data.get("Website"), market_key)
        lc1, lc2, lc3 = st.columns(3)
        with lc1: safe_link_button("OFFICIAL", links["official"], use_container_width=True)
        with lc2: safe_link_button("IR SEARCH", links["ir_search"], use_container_width=True)
        with lc3: safe_link_button("EARNINGS DECK", links["earnings_deck"], use_container_width=True)

        # --- PRO PEER DASHBOARD (visual) ---
        with st.expander("📊 PEER DASHBOARD — Sector peers (pro)", expanded=False):
            st.markdown(
                "<div class='hint-bar'>"
                "<b>What you get:</b> percentile ranks, a <b>Peer Map</b> with pro-axis switch (e.g., RS×Valuation PER/PBR), and a ranked peer table. "
                "All peers are within the selected sector."
                "</div>",
                unsafe_allow_html=True,
            )
            peer_mode = st.selectbox(
                "Peer Set",
                ["Top Momentum (Apex)", "Top Sector Winners (RS_Sec)", "Size Peers (Market Cap)"],
                index=0,
                key=f"peer_mode_{market_key}_{target_sector}_{lookback_key}_{top['Ticker']}",
            )
            peer_n = st.slider(
                "Peers",
                min_value=8,
                max_value=25,
                value=12,
                step=1,
                key=f"peer_n_{market_key}_{target_sector}_{lookback_key}_{top['Ticker']}",
            )
            st.markdown(
                "<div class='mini-note'><b>Peer Map Axis (Pro)</b> — Switch to RS×Valuation (PER/PBR) etc.</div>",
                unsafe_allow_html=True,
            )
            axis_opts = {
                "RS vs Sector (RS_Sec)": "RS_Sec",
                "RS vs Market (RS)": "RS",
                "Valuation: P/E (PER)": "PER",
                "Valuation: P/B (PBR)": "PBR",
                "Apex Score": "Apex",
                "Resilience (−MaxDD)": "Resilience",
                "Return (Lookback)": "Ret",
                "1M Return": "1M",
                "3M Return": "3M",
                "HighDist (to 52W High)": "HighDist",
                "Accel": "Accel",
                "Volatility (ann%)": "Vol(ann)",
                "Beta (calc)": "Beta(calc)",
            }
            axis_labels = list(axis_opts.keys())
            _def_x = axis_labels.index("RS vs Sector (RS_Sec)") if "RS vs Sector (RS_Sec)" in axis_labels else 0
            _def_y = axis_labels.index("Resilience (−MaxDD)") if "Resilience (−MaxDD)" in axis_labels else 0
            x_axis_label = st.selectbox(
                "Peer Map — X Axis",
                axis_labels,
                index=_def_x,
                key=f"peer_x_{market_key}_{target_sector}_{lookback_key}_{top['Ticker']}",
            )
            y_axis_label = st.selectbox(
                "Peer Map — Y Axis",
                axis_labels,
                index=_def_y,
                key=f"peer_y_{market_key}_{target_sector}_{lookback_key}_{top['Ticker']}",
            )
            x_col = axis_opts.get(x_axis_label, "RS_Sec")
            y_col = axis_opts.get(y_axis_label, "Resilience")

            peer_ctx = f"{market_key}|{target_sector}|{lookback_key}|{top['Ticker']}|{peer_mode}|{peer_n}"
            build_peers = st.button(
                "BUILD PEER DASHBOARD",
                key=f"peer_build_{peer_ctx}",
                use_container_width=True,
            )
            if build_peers and isinstance(st.session_state.get("peer_cache"), dict):
                st.session_state.peer_cache[peer_ctx] = True
                st.rerun()

            if isinstance(st.session_state.get("peer_cache"), dict) and st.session_state.peer_cache.get(peer_ctx):
                # Universe (numeric)
                uni = df.copy()
                # Basic safety conversions
                for c in ["Apex","RS_Sec","RS","Ret","1M","3M","MaxDD","HighDist","Accel","MCap","PER","PBR","ROE","RevGrow","OpMargin","Beta"]:
                    if c in uni.columns:
                        uni[c] = pd.to_numeric(uni[c], errors="coerce")

                target_ticker = str(top["Ticker"])
                target_mcap = float(mcap) if pd.notna(mcap) else np.nan

                # Peer selection
                peer_tickers = []
                try:
                    if peer_mode.startswith("Top Momentum"):
                        peer_tickers = uni.sort_values("Apex", ascending=False)["Ticker"].head(peer_n).tolist()
                    elif peer_mode.startswith("Top Sector"):
                        peer_tickers = uni.sort_values("RS_Sec", ascending=False)["Ticker"].head(peer_n).tolist()
                    else:
                        # Size peers: fetch MCap for a capped candidate set, then find nearest.
                        cand = uni["Ticker"].tolist()
                        cap_n = 80
                        if len(cand) > cap_n:
                            cand = uni.sort_values("Apex", ascending=False)["Ticker"].head(cap_n).tolist()
                            st.caption(f"Size peers use a capped candidate set: top {cap_n} by Apex (for speed).")
                        if target_ticker not in cand:
                            cand = [target_ticker] + cand[:-1]
                        cand_f = fetch_fundamentals_batch(cand).reset_index()
                        cand_f["MCap"] = pd.to_numeric(cand_f.get("MCap"), errors="coerce")
                        t_m = float(cand_f.loc[cand_f["Ticker"] == target_ticker, "MCap"].iloc[0]) if (not cand_f.empty and (cand_f["Ticker"] == target_ticker).any()) else target_mcap
                        if pd.isna(t_m): t_m = target_mcap
                        cand_f["Dist"] = (cand_f["MCap"] - float(t_m or 0)).abs()
                        peer_tickers = cand_f.sort_values("Dist")["Ticker"].tolist()
                        peer_tickers = [t for t in peer_tickers if t != target_ticker][:peer_n]
                except Exception:
                    peer_tickers = uni.sort_values("Apex", ascending=False)["Ticker"].head(peer_n).tolist()

                # Ensure target is included
                if target_ticker not in peer_tickers:
                    peer_tickers = [target_ticker] + peer_tickers[: max(0, peer_n-1)]
                else:
                    # Move target to top
                    peer_tickers = [target_ticker] + [t for t in peer_tickers if t != target_ticker]

                peers = uni[uni["Ticker"].isin(peer_tickers)].copy()
                # Keep peer order
                peers["__ord"] = peers["Ticker"].apply(lambda x: peer_tickers.index(x) if x in peer_tickers else 9999)
                peers = peers.sort_values("__ord").drop(columns=["__ord"])

                # Fundamentals for peers (small batch; cached)
                try:
                    pf = fetch_fundamentals_batch(peer_tickers).reset_index()
                except Exception:
                    pf = pd.DataFrame({"Ticker": peer_tickers})
                peers = peers.merge(pf, on="Ticker", how="left", suffixes=("", "_f"))

                # Market cap / valuation fallbacks
                if "MCap_f" in peers.columns and "MCap" in peers.columns:
                    peers["MCap"] = peers["MCap"].fillna(peers["MCap_f"])
                if "Currency_f" in peers.columns and "Currency" in peers.columns:
                    peers["Currency"] = peers["Currency"].fillna(peers["Currency_f"])
                for c in ["PER","PBR","ROE","RevGrow","OpMargin","Beta"]:
                    if f"{c}_f" in peers.columns and c in peers.columns:
                        peers[c] = peers[c].fillna(peers[f"{c}_f"])

                # Extra risk metrics from prices (Vol/BetaCalc)
                bench_series = None
                try:
                    if "sec_df" in st.session_state:
                        sec_df = st.session_state.sec_df
                        if bench in sec_df.columns:
                            bench_series = sec_df[bench]
                except Exception:
                    bench_series = None

                def _realized_vol(price: pd.Series, win: int = 63) -> float:
                    try:
                        p = price.ffill().tail(win+1)
                        r = p.pct_change().dropna()
                        if r.empty: return np.nan
                        return float(r.std(ddof=0) * (252 ** 0.5) * 100)
                    except Exception:
                        return np.nan

                def _beta_calc(price: pd.Series, bench_p: pd.Series, win: int = 63) -> float:
                    try:
                        if bench_p is None: return np.nan
                        p = price.ffill().tail(win+1)
                        b = bench_p.ffill().tail(win+1)
                        r = p.pct_change()
                        rb = b.pct_change()
                        tmp = pd.concat([r, rb], axis=1).dropna()
                        if tmp.shape[0] < 3: return np.nan
                        cov = float(tmp.iloc[:,0].cov(tmp.iloc[:,1]))
                        var = float(tmp.iloc[:,1].var())
                        return cov/var if var > 0 else np.nan
                    except Exception:
                        return np.nan

                peers["Vol(ann)"] = np.nan
                peers["Beta(calc)"] = np.nan
                try:
                    if "sec_df" in st.session_state:
                        sec_df = st.session_state.sec_df
                        for t in peer_tickers:
                            if t in sec_df.columns:
                                peers.loc[peers["Ticker"] == t, "Vol(ann)"] = _realized_vol(sec_df[t], win=min(LOOKBACKS.get(lookback_key, 63), 126))
                                peers.loc[peers["Ticker"] == t, "Beta(calc)"] = _beta_calc(sec_df[t], bench_series, win=min(LOOKBACKS.get(lookback_key, 63), 126))
                except Exception:
                    pass

                # Add helper columns
                peers["Resilience"] = -pd.to_numeric(peers.get("MaxDD"), errors="coerce")
                peers["MCapDisp"] = peers.apply(lambda r: format_mcap(r.get("MCap", 0), r.get("Currency")), axis=1)

                # Percentile table (sector universe, technical only)
                pct_rows = []
                pct_metrics = [
                    ("Apex", True),
                    ("RS_Sec", True),
                    ("RS", True),
                    ("Ret", True),
                    ("1M", True),
                    ("3M", True),
                    ("MaxDD", True),  # higher (closer to 0) is better
                    ("HighDist", True),
                    ("Accel", True),
                ]
                for met, higher_better in pct_metrics:
                    if met not in uni.columns: 
                        continue
                    s = pd.to_numeric(uni[met], errors="coerce")
                    if s.dropna().empty:
                        continue
                    v = pd.to_numeric(peers.loc[peers["Ticker"] == target_ticker, met], errors="coerce")
                    v = float(v.iloc[0]) if (not v.empty and pd.notna(v.iloc[0])) else np.nan
                    if pd.isna(v):
                        continue
                    try:
                        idx0 = uni.index[uni["Ticker"] == target_ticker]
                        if len(idx0) > 0:
                            pct = float((s.rank(pct=True) * 100).loc[idx0].iloc[0])
                        else:
                            pct = float((s < v).mean() * 100)
                    except Exception:
                        pct = float((s < v).mean() * 100)
                    pct_rows.append({"Metric": met, "Value": v, "Percentile": pct})
                if pct_rows:
                    pct_df = pd.DataFrame(pct_rows)
                    # Formatting
                    try:
                        pct_df["Value"] = pct_df.apply(lambda r: f"{r['Value']:.2f}" if r["Metric"] in ["Apex","RS_Sec","RS","Accel"] else f"{r['Value']:+.1f}%", axis=1)
                    except Exception:
                        pass
                    try:
                        pct_df["Percentile"] = pct_df["Percentile"].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
                    except Exception:
                        pass
                    st.markdown("<div class='mini-note'><b>Target Percentile (within sector)</b></div>", unsafe_allow_html=True)
                    st.dataframe(pct_df, hide_index=True, use_container_width=True)

                # Visuals
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go

                    plot_df = peers.copy()
                    # Bubble size (log cap)
                    plot_df["MCapSize"] = pd.to_numeric(plot_df["MCap"], errors="coerce")
                    plot_df["MCapSize"] = plot_df["MCapSize"].fillna(plot_df["MCapSize"].median())
                    plot_df["MCapSize"] = np.log10(plot_df["MCapSize"].clip(lower=1))
                    # Selected axes (pro)
                    if x_col not in plot_df.columns or y_col not in plot_df.columns:
                        st.warning("Selected axes are not available for this peer set.")
                    else:
                        try:
                            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
                            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
                        except Exception:
                            pass

                        title_txt = f"Peer Map: {x_axis_label} × {y_axis_label}"

                        hover_data = {
                            "Ticker": True,
                            "Apex": ":.2f",
                            "RS_Sec": ":.2f",
                            "RS": ":.2f",
                            "PER": ":.1f",
                            "PBR": ":.2f",
                            "Ret": ":.1f",
                            "MaxDD": ":.1f",
                            "MCapDisp": True,
                        }
                        # Keep only columns that exist
                        try:
                            hover_data = {k: v for k, v in hover_data.items() if (k in plot_df.columns) or (k in ["Ticker", "MCapDisp"])}
                        except Exception:
                            pass

                        fig = px.scatter(
                            plot_df,
                            x=x_col,
                            y=y_col,
                            size="MCapSize",
                            color="Apex",
                            hover_name="Name",
                            hover_data=hover_data,
                            title=title_txt,
                        )
                        fig.update_layout(
                            template="plotly_dark",
                            height=420,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e6e6e6",
                            margin=dict(l=10, r=10, t=50, b=10),
                            showlegend=False,
                        )
                        # Highlight target
                        trow = plot_df.loc[plot_df["Ticker"] == target_ticker]
                        if not trow.empty:
                            try:
                                fig.add_trace(
                                    go.Scatter(
                                        x=trow[x_col],
                                        y=trow[y_col],
                                        mode="markers+text",
                                        text=["TARGET"],
                                        textposition="top center",
                                        marker=dict(size=16, symbol="diamond", line=dict(width=2, color="#00f2fe"), color="#00f2fe"),
                                        hoverinfo="skip",
                                    )
                                )
                            except Exception:
                                pass

                        # Reference lines: 0-line for RS/Resilience-like axes; median line for valuation axes
                        try:
                            if x_col in ["RS_Sec", "RS", "Accel", "Resilience"]:
                                fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="rgba(255,255,255,0.25)")
                        except Exception:
                            pass
                        try:
                            if y_col in ["RS_Sec", "RS", "Accel", "Resilience"]:
                                fig.add_hline(y=0, line_dash="dot", line_width=1, line_color="rgba(255,255,255,0.25)")
                        except Exception:
                            pass
                        try:
                            if x_col in ["PER", "PBR"]:
                                mx = pd.to_numeric(plot_df[x_col], errors="coerce").median()
                                if pd.notna(mx):
                                    fig.add_vline(x=float(mx), line_dash="dot", line_width=1, line_color="rgba(255,255,255,0.25)")
                        except Exception:
                            pass
                        try:
                            if y_col in ["PER", "PBR"]:
                                my = pd.to_numeric(plot_df[y_col], errors="coerce").median()
                                if pd.notna(my):
                                    fig.add_hline(y=float(my), line_dash="dot", line_width=1, line_color="rgba(255,255,255,0.25)")
                        except Exception:
                            pass

                        st.plotly_chart(fig, use_container_width=True)

                    # Apex rank bar
                    bar_df = peers.copy()
                    bar_df["Label"] = bar_df["Name"] + " (" + bar_df["Ticker"] + ")"
                    bar_df = bar_df.sort_values("Apex", ascending=True)
                    fig2 = px.bar(
                        bar_df,
                        x="Apex",
                        y="Label",
                        orientation="h",
                        title="Peer Ranking: Apex (Momentum Composite)",
                    )
                    fig2.update_layout(
                        template="plotly_dark",
                        height=420,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e6e6e6",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.info(f"Peer visuals unavailable: {e}")

                # Peer table (pro view)
                show_cols = [
                    "Name","Ticker","MCapDisp","Apex","RS","RS_Sec","Ret","1M","3M","MaxDD","Resilience","Vol(ann)","Beta(calc)","PER","PBR","ROE","RevGrow"
                ]
                show_cols = [c for c in show_cols if c in peers.columns]
                disp = peers[show_cols].copy()
                # friendly formatting
                for c in ["Ret","1M","3M","MaxDD","Resilience","Vol(ann)"]:
                    if c in disp.columns:
                        disp[c] = pd.to_numeric(disp[c], errors="coerce").apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
                for c in ["Apex","RS","RS_Sec","Beta(calc)"]:
                    if c in disp.columns:
                        disp[c] = pd.to_numeric(disp[c], errors="coerce").apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                for c in ["PER","PBR"]:
                    if c in disp.columns:
                        disp[c] = pd.to_numeric(disp[c], errors="coerce").apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                for c in ["ROE","RevGrow"]:
                    if c in disp.columns:
                        disp[c] = pd.to_numeric(disp[c], errors="coerce").apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-")

                st.markdown("<div class='mini-note'><b>Peer Table</b></div>", unsafe_allow_html=True)
                st.dataframe(disp, hide_index=True, use_container_width=True)
                st.caption(
                    "NOTES | Resilience = −MaxDD（大きいほどドローダウン耐性が高い） | "
                    "Vol(ann): 期間内の日次リターンから年率換算 | Beta(calc): 同期間の指数に対する回帰β（価格系列から算出）"
                )
            else:
                st.info("Click BUILD PEER DASHBOARD to render comparisons.")

        if analyst_note_txt:
            st.download_button(
                "DOWNLOAD ANALYST NOTE",
                analyst_note_txt,
                f"analyst_note_{top['Ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )

    with nc2:
        st.caption("INTEGRATED NEWS FEED")
        load_news = st.button(
            "LOAD NEWS FEED",
            key=f"load_news_{market_key}_{top['Ticker']}",
            use_container_width=True,
        )
        if load_news:
            with st.spinner("Loading news..."):
                try:
                    news_items2, news_context2, _, _ = get_news_consolidated(top["Ticker"], top["Name"], market_key, limit_each=10)
                except Exception:
                    news_items2, news_context2 = [], ""
                if isinstance(st.session_state.get("news_panels"), dict):
                    st.session_state.news_panels[news_ctx_id] = {"items": news_items2, "context": news_context2}
                st.toast("🗞️ News updated", icon="🗞️")
                st.rerun()

        if news_items:
            for n in news_items[:20]:
                dt = datetime.fromtimestamp(n.get("pub", 0)).strftime("%Y/%m/%d") if n.get("pub") else "-"
                src = n.get("src", "-")
                title = n.get("title", "-")
                link = n.get("link", "")
                try:
                    st.markdown(f"- {dt} [{src}] [{title}]({link})")
                except Exception:
                    st.markdown(f"- {dt} [{src}] {title}")
        else:
            st.markdown("<div class='mini-note'><span class='dim'>No news loaded yet. Click <b>LOAD NEWS FEED</b> or generate the AI briefing (which also loads news).</span></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    run()
