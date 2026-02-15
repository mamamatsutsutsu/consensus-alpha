# THEMELENS — AI-first Theme Portfolio Builder (Gemini) v12 (AI-only universe)
# -----------------------------------------------------------------------------
# What changed vs v11:
# - No ETF screening. Candidate universe is proposed directly by Gemini based on theme/region/N.
# - Prompt is engineered to prioritize completeness of *Theme Market Cap* leaders (mktcap × TRR).
# - Quick Draft is still fast: 1st call = candidates + TRR; 2nd call (small) = summaries for Top-N only.
# - Robust JSON parsing + one retry with a minimal schema. If still fails, we show a clean error (no crash).
# - Controls UI is redesigned for visibility (bigger fields + better spacing + mobile-friendly).
# - Definitions are always displayed (not hidden).
#
# Deliverable: Ranked list by Theme Market Cap (month-end) = Free-float MktCap (proxy) × TRR.

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import concurrent.futures as cf
import hashlib
import json
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# =============================================================================
# Streamlit rerun helper (version compatibility)
# =============================================================================
def _rerun() -> None:
    fn = getattr(st, "rerun", None)
    if callable(fn):
        fn(); return
    fn = getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn(); return
    raise RuntimeError("Streamlit rerun is not available in this environment.")


# =============================================================================
# Styling — AlphaLens-like, but boost control visibility (mobile-friendly)
# =============================================================================
THEMELENS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

:root{
  --al-cyan:#00f2fe;
  --al-muted:rgba(255,255,255,0.72);
  --al-border:rgba(255,255,255,0.10);
  --al-bg:rgba(0,0,0,0.58);
  --al-bg2:rgba(0,0,0,0.35);
}

/* Do NOT touch padding-top (app.py owns safe-top spacing) */
div[data-testid="stAppViewContainer"] .block-container{
  padding-bottom: 2.0rem;
  max-width: 1400px;
}

/* Title */
.tl-title{
  font-family:Orbitron, ui-sans-serif, system-ui;
  letter-spacing:0.14em;
  color:var(--al-cyan);
  margin: 0 0 12px 0;
  font-size: 28px;
}

/* Control bar */
.tl-panel{
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: var(--al-bg);
  backdrop-filter: blur(10px);
  padding: 14px 14px;
}

/* Make inputs bigger & clearer */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea{
  background: rgba(255,255,255,0.06) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  font-size: 16px !important;
  padding: 12px 12px !important;
}

div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.06) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

div[role="radiogroup"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 8px 10px;
  border-radius: 14px;
}

label{
  font-weight: 800 !important;
  letter-spacing: 0.02em;
}

/* Buttons */
.stButton>button{
  border-radius: 14px;
  font-weight: 900;
  letter-spacing: 0.04em;
  padding: 0.72rem 1.0rem;
  font-size: 15px;
}
.stButton>button[kind="primary"]{
  box-shadow: 0 0 18px rgba(0,242,254,0.18);
  border: 1px solid rgba(0,242,254,0.40);
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border-radius:14px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.07);
}

/* Small chips */
.tl-chip{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.05);
  color: rgba(255,255,255,0.80);
  font-size: 12px;
}
</style>
"""


# =============================================================================
# Types
# =============================================================================
RegionMode = Literal["Global", "Japan", "US", "Europe", "China"]


@dataclass(frozen=True)
class ThemeInput:
    theme_text: str
    region_mode: RegionMode
    top_n: int


# =============================================================================
# Utils
# =============================================================================
def _get_secret(key: str) -> Optional[str]:
    try:
        v = st.secrets.get(key)  # type: ignore[attr-defined]
        if v is not None and str(v).strip():
            return str(v).strip()
    except Exception:
        pass
    v = os.getenv(key)
    if v and v.strip():
        return v.strip()
    return None


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def clamp01(x: Any) -> float:
    try:
        return max(0.0, min(float(x), 1.0))
    except Exception:
        return 0.0


def most_recent_month_end(today: date) -> date:
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = next_month - timedelta(days=1)
    if today >= month_end:
        return month_end
    return today.replace(day=1) - timedelta(days=1)


def fmt_money(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        v = float(x)
        if abs(v) >= 1e12:
            return f"{v/1e12:.2f}T"
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.2f}M"
        return f"{v:,.0f}"
    except Exception:
        return "-"


def _region_definition(region: RegionMode) -> str:
    if region == "Global":
        return "Developed markets (US, Japan, Western Europe incl. UK & Switzerland) + China (Mainland + HK). Avoid frontier/illiquid markets."
    if region == "Japan":
        return "Japan listed equities (TSE Prime etc)."
    if region == "US":
        return "United States listed equities (NYSE/NASDAQ etc)."
    if region == "Europe":
        return "Developed Europe: EU + UK + Switzerland (exclude Russia)."
    if region == "China":
        return "China large-caps: Mainland A-shares + Hong Kong listed China large-caps."
    return "Global"


def _large_cap_threshold_usd(region: RegionMode) -> float:
    # pragmatic threshold for "major index sized" companies
    if region in ("US", "Global"):
        return 10e9
    if region == "Japan":
        return 5e9
    if region in ("Europe", "China"):
        return 8e9
    return 8e9


def _candidate_pool(top_n: int) -> int:
    n = max(1, min(int(top_n), 30))
    # enough buffer to reduce "missed leaders" risk, but keep speed
    return int(min(60, max(25, 2 * n + 18)))


# =============================================================================
# Market data (month-end free-float-ish market cap proxy)
# =============================================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _yf_download_cached(tickers: Tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _ticker_info_cached(ticker: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _fx_usd_per_ccy_cached(ccy: str, asof_iso: str) -> Optional[float]:
    ccy = (ccy or "USD").upper()
    if ccy == "USD":
        return 1.0
    asof = date.fromisoformat(asof_iso)
    candidates = [f"{ccy}USD=X", f"USD{ccy}=X"]
    for sym in candidates:
        try:
            df = _yf_download_cached((sym,), start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat())
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                s = df[(sym, "Close")].dropna() if (sym, "Close") in df.columns else pd.Series(dtype=float)
            else:
                s = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
            if s.empty:
                continue
            rate = float(s.iloc[-1])
            if not (rate > 0):
                continue
            usd_per_ccy = rate
            if sym.startswith("USD") or rate > 20:
                usd_per_ccy = 1.0 / rate
            if 0 < usd_per_ccy < 100:
                return float(usd_per_ccy)
        except Exception:
            continue
    return None


def _batch_infos(tickers: List[str], max_workers: int = 14) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not tickers:
        return out
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_ticker_info_cached, t): t for t in tickers}
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                out[t] = fut.result() or {}
            except Exception:
                out[t] = {}
    return out


def month_end_close_batch(tickers: List[str], asof: date) -> Dict[str, Optional[float]]:
    if not tickers:
        return {}
    start = (asof - timedelta(days=10)).isoformat()
    end = (asof + timedelta(days=1)).isoformat()
    df = _yf_download_cached(tuple(tickers), start=start, end=end)
    out: Dict[str, Optional[float]] = {t: None for t in tickers}
    if df is None or df.empty:
        return out

    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            col = (t, "Close")
            if col in df.columns:
                s = df[col].dropna()
                out[t] = float(s.iloc[-1]) if not s.empty else None
    else:
        if "Close" in df.columns and len(tickers) == 1:
            s = df["Close"].dropna()
            out[tickers[0]] = float(s.iloc[-1]) if not s.empty else None

    return out


def batch_mktcap_asof_usd(tickers: List[str], asof: date) -> pd.DataFrame:
    tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker","currency","mcap_usd","mcap_quality"])

    close_map = month_end_close_batch(tickers, asof)
    info_map = _batch_infos(tickers, max_workers=14)

    currencies = set(str((info_map.get(t, {}) or {}).get("currency") or "USD").upper() for t in tickers)
    fx_map = {ccy: _fx_usd_per_ccy_cached(ccy, asof.isoformat()) for ccy in currencies}

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        info = info_map.get(t, {}) or {}
        ccy = str(info.get("currency") or "USD").upper()
        px = close_map.get(t)

        mcap_local = None
        quality = "missing"

        if px is not None and px > 0:
            float_sh = info.get("floatShares")
            sh_out = info.get("sharesOutstanding")
            if float_sh and float(float_sh) > 0:
                mcap_local = float(float_sh) * float(px)
                quality = "proxy_free_float(floatShares×Close)"
            elif sh_out and float(sh_out) > 0:
                mcap_local = float(float(sh_out) * float(px))
                quality = "proxy_total(sharesOutstanding×Close)"
            else:
                mcap_now = info.get("marketCap")
                px_now = info.get("regularMarketPrice") or info.get("currentPrice")
                if mcap_now and px_now and float(px_now) > 0:
                    shares = float(mcap_now) / float(px_now)
                    mcap_local = shares * float(px)
                    quality = "proxy_derived(marketCap/currentPrice×Close)"
                elif mcap_now and float(mcap_now) > 0:
                    mcap_local = float(mcap_now)
                    quality = "proxy_field(marketCap)"
        else:
            mcap_now = info.get("marketCap")
            if mcap_now and float(mcap_now) > 0:
                mcap_local = float(mcap_now)
                quality = "proxy_field(marketCap_no_asof_price)"

        fx = fx_map.get(ccy)
        mcap_usd = (float(mcap_local) * float(fx)) if (mcap_local is not None and fx is not None) else None

        rows.append({
            "ticker": t,
            "currency": ccy,
            "mcap_usd": mcap_usd,
            "mcap_quality": quality + ("" if fx is not None else "|fx_unavailable"),
            "_country": str(info.get("country") or ""),
            "_exchange": str(info.get("exchange") or ""),
            "_shortName": str(info.get("shortName") or info.get("longName") or ""),
            "_sector": str(info.get("sector") or ""),
            "_industry": str(info.get("industry") or ""),
        })

    df = pd.DataFrame(rows)
    df["mcap_usd"] = pd.to_numeric(df["mcap_usd"], errors="coerce")
    return df


# =============================================================================
# Disk cache (AI)
# =============================================================================
def _ai_cache_dir(data_dir: str) -> Path:
    p = Path(data_dir) / "ai_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(prefix: str, *parts: str) -> str:
    raw = prefix + "|" + "|".join([p or "" for p in parts])
    return _sha12(raw)


def _cache_load(data_dir: str, key: str, max_age_days: int = 30) -> Optional[Dict[str, Any]]:
    path = _ai_cache_dir(data_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        ts = obj.get("__cached_at__", None)
        if ts:
            cached_at = datetime.fromisoformat(ts)
            if (datetime.utcnow() - cached_at) > timedelta(days=max_age_days):
                return None
        return obj.get("payload")
    except Exception:
        return None


def _cache_save(data_dir: str, key: str, payload: Dict[str, Any]) -> None:
    path = _ai_cache_dir(data_dir) / f"{key}.json"
    obj = {"__cached_at__": datetime.utcnow().isoformat(timespec="seconds"), "payload": payload}
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# Gemini JSON — robust parsing + REST-first
# =============================================================================
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_json_any(txt: str) -> Any:
    txt = _strip_code_fences(str(txt or ""))
    if not txt:
        raise ValueError("Empty model response.")
    # direct
    try:
        return json.loads(txt)
    except Exception:
        pass
    # extract object
    lo, hi = txt.find("{"), txt.rfind("}")
    if lo != -1 and hi != -1 and hi > lo:
        frag = txt[lo:hi+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    # extract array
    lo, hi = txt.find("["), txt.rfind("]")
    if lo != -1 and hi != -1 and hi > lo:
        frag = txt[lo:hi+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    raise ValueError("No JSON object/array found.")


def gemini_generate_json(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    system_txt = ""
    user_parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower().strip()
        content = str(m.get("content") or "")
        if role == "system" and not system_txt:
            system_txt = content
        elif role == "user":
            user_parts.append(content)
    user_txt = "\n\n".join([p for p in user_parts if p]).strip()
    if not user_txt:
        raise RuntimeError("Gemini call: missing user prompt.")

    api_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")

    # REST-first (timeout reliable)
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_txt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
                "responseMimeType": "application/json",
            },
        }
        if system_txt:
            payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_txt}]}

        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=int(timeout_s))
        r.raise_for_status()
        data = r.json()
        txt = ""
        try:
            cand0 = (data.get("candidates") or [])[0]
            content = (cand0.get("content") or {})
            parts = (content.get("parts") or [])
            if parts:
                txt = "\n".join([str(p.get("text","") or "") for p in parts if isinstance(p, dict)])
        except Exception:
            txt = ""
        obj = _parse_json_any(txt)
        return obj if isinstance(obj, dict) else {"status": "ok", "rows": obj}

    # SDK fallback (Vertex/ADC)
    try:
        from google.genai import types  # type: ignore
        from google import genai  # type: ignore
        client = genai.Client()
        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
            system_instruction=system_txt or None,
            response_mime_type="application/json",
            max_output_tokens=int(max_output_tokens),
        )
        resp = client.models.generate_content(model=model, contents=user_txt, config=cfg)
        txt = getattr(resp, "text", "") or ""
        if callable(txt):
            txt = txt()
        obj = _parse_json_any(str(txt))
        return obj if isinstance(obj, dict) else {"status": "ok", "rows": obj}
    except Exception as e:
        raise RuntimeError(f"Gemini call failed. Configure GEMINI_API_KEY/GOOGLE_API_KEY. Error: {e}")


# =============================================================================
# Prompts (AI-only universe)
# =============================================================================
def _ai_system_prompt() -> str:
    return (
        "You are a senior global equity portfolio manager and sector analyst. "
        "Return ONLY valid JSON. No markdown. "
        "Never invent URLs or quotes. "
        "If you are uncertain, label TRR as method='estimated' confidence='Low'."
    )


def build_messages_candidates(theme: str, region: RegionMode, asof: date, top_n: int, pool: int) -> List[Dict[str, str]]:
    region_def = _region_definition(region)

    schema = {
        "status": "ok|ambiguous|error",
        "theme_definition": "string",
        "notes": "string",
        "reference_etfs": ["string"],
        "companies": [
            {
                "company_name": "string",
                "ticker": "string",
                "listed_country": "string",
                "primary_exchange": "string",
                "theme_revenue_ratio": 0.0,
                "method": "disclosed|proxy|estimated",
                "confidence": "High|Med|Low",
            }
        ],
    }

    user_prompt = f"""
Task: Propose a large-cap candidate universe for a theme portfolio.

Inputs:
- Theme text: "{theme}"
- Region: {region}
- Region definition: {region_def}
- Target Top N: {top_n}
- Candidate pool size to return: {pool}
- Market-cap as-of date: {asof.isoformat()}

Portfolio objective (IMPORTANT):
We will compute month-end Theme Market Cap locally and rank by:
Theme Market Cap = Free-float Market Cap (proxy, USD) × TRR
So the candidate list must be COMPLETE for likely top Theme Market Cap leaders.

Rules (IMPORTANT):
- Only large, liquid companies typical of major indices in the region.
- Avoid small caps and obscure listings.
- Provide yfinance-compatible tickers WITH correct suffix when needed (examples: 0700.HK, 9984.T, NESN.SW, HSBA.L).
- TRR (theme_revenue_ratio) is 0-1 and represents theme-related revenue / total revenue.
- If theme is ambiguous / not investable, return status="ambiguous" with theme_definition and 3-8 reference_etfs; companies=[].

Output format:
Return ONLY JSON following this schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


def build_messages_candidates_minimal_retry(theme: str, region: RegionMode, asof: date, top_n: int, pool: int) -> List[Dict[str, str]]:
    # second attempt: minimal schema to improve JSON reliability
    region_def = _region_definition(region)
    schema = {"status": "ok|ambiguous|error", "theme_definition": "string", "reference_etfs": ["string"], "tickers": ["string"], "trr": [0.0]}
    user_prompt = f"""
Return ONLY JSON.

Theme: "{theme}"
Region: {region} (definition: {region_def})
Need {pool} LARGE-CAP tickers (yfinance format). Avoid small caps.

Also return a parallel array trr[] of same length with TRR estimates (0-1). If unsure, use a conservative estimate.

If ambiguous: status="ambiguous", tickers=[], trr=[] and provide reference_etfs.

Schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


def build_messages_summaries(theme: str, region: RegionMode, tickers: List[str]) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    schema = {"status": "ok|error", "rows": [{"ticker": "string", "theme_ja": "string", "non_theme_ja": "string"}]}
    user_prompt = f"""
Task: Write very short Japanese summaries (institutional tone). No disclaimers.
Theme: "{theme}"
Region: {region} (definition: {region_def})
Tickers: {json.dumps(tickers, ensure_ascii=False)}

Constraints:
- theme_ja: 1 sentence, <= 120 Japanese characters
- non_theme_ja: 1 sentence, <= 90 Japanese characters

Return ONLY JSON:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


def build_messages_refine_evidence(theme: str, region: RegionMode, asof: date, rows_payload: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    schema = {
        "status": "ok|error",
        "notes": "string",
        "rows": [
            {
                "ticker": "string",
                "theme_revenue_ratio": 0.0,
                "method": "disclosed|proxy|estimated",
                "confidence": "High|Med|Low",
                "sources": [
                    {"title": "string", "publisher": "string", "year": None, "url": None, "locator": "string", "excerpt": ""}
                ],
                "theme_ja": "string",
                "non_theme_ja": "string",
            }
        ],
    }

    user_prompt = f"""
Task: REFINE (Top-N only). Add TRR evidence metadata and improve summaries.

Theme: "{theme}"
Region: {region} (definition: {region_def})
As-of (market cap): {asof.isoformat()}
Top-N payload:
{json.dumps(rows_payload, ensure_ascii=False)}

Rules:
- NEVER invent URLs or quotes.
- If you cannot provide a verified URL+exact excerpt, set url=null and excerpt="" and keep method="estimated" confidence="Low".
- Prefer filings/annual report/IR decks. Provide up to 2 sources per company.

Return ONLY JSON:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


# =============================================================================
# Normalizers
# =============================================================================
def _normalize_candidates(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "companies": [], "notes": "LLM output not a JSON object."}
    status = str(obj.get("status", "error"))
    if status not in ("ok", "ambiguous", "error"):
        status = "error"
    companies = obj.get("companies", []) or []
    if not isinstance(companies, list):
        companies = []
    out = []
    for c in companies:
        if not isinstance(c, dict):
            continue
        t = str(c.get("ticker", "") or "").strip().upper()
        if not t:
            continue
        out.append({
            "company_name": str(c.get("company_name","") or "").strip(),
            "ticker": t,
            "listed_country": str(c.get("listed_country","") or "").strip(),
            "primary_exchange": str(c.get("primary_exchange","") or "").strip(),
            "theme_revenue_ratio": clamp01(c.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(c.get("method", "estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(c.get("confidence", "Low") or "Low"),
        })
    return {
        "status": status,
        "theme_definition": str(obj.get("theme_definition","") or ""),
        "notes": str(obj.get("notes","") or ""),
        "reference_etfs": obj.get("reference_etfs", []) if isinstance(obj.get("reference_etfs", []), list) else [],
        "companies": out,
    }


def _normalize_candidates_minimal(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "tickers": [], "trr": []}
    status = str(obj.get("status", "error"))
    if status not in ("ok", "ambiguous", "error"):
        status = "error"
    tickers = obj.get("tickers", []) or []
    trr = obj.get("trr", []) or []
    if not isinstance(tickers, list):
        tickers = []
    if not isinstance(trr, list):
        trr = []
    tickers2 = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    trr2 = [clamp01(x) for x in trr]
    # align length
    L = min(len(tickers2), len(trr2))
    tickers2, trr2 = tickers2[:L], trr2[:L]
    ref = obj.get("reference_etfs", []) if isinstance(obj.get("reference_etfs", []), list) else []
    return {
        "status": status,
        "theme_definition": str(obj.get("theme_definition","") or ""),
        "reference_etfs": ref,
        "rows": [{"ticker": tickers2[i], "theme_revenue_ratio": trr2[i]} for i in range(L)],
    }


def _normalize_summaries(obj: Any) -> Dict[str, Dict[str, str]]:
    if not isinstance(obj, dict) or str(obj.get("status")) != "ok":
        return {}
    rows = obj.get("rows", []) or []
    if not isinstance(rows, list):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker","") or "").strip().upper()
        if not t:
            continue
        out[t] = {
            "theme_ja": str(r.get("theme_ja","") or "").strip(),
            "non_theme_ja": str(r.get("non_theme_ja","") or "").strip(),
        }
    return out


def _normalize_refine(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "rows": []}
    status = str(obj.get("status","error"))
    if status != "ok":
        return {"status": status, "rows": [], "notes": str(obj.get("notes","") or "")}
    rows = obj.get("rows", []) or []
    if not isinstance(rows, list):
        rows = []
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker","") or "").strip().upper()
        if not t:
            continue
        srcs = r.get("sources", []) or []
        if not isinstance(srcs, list):
            srcs = []
        out_rows.append({
            "ticker": t,
            "theme_revenue_ratio": clamp01(r.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(r.get("method","estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("confidence","Low") or "Low"),
            "theme_revenue_sources": srcs[:2],
            "theme_business_summary_ja": str(r.get("theme_ja","") or "").strip(),
            "non_theme_business_summary_ja": str(r.get("non_theme_ja","") or "").strip(),
        })
    return {"status": "ok", "rows": out_rows, "notes": str(obj.get("notes","") or "")}


# =============================================================================
# Executor + cached AI call
# =============================================================================
@st.cache_resource
def _executor() -> cf.ThreadPoolExecutor:
    return cf.ThreadPoolExecutor(max_workers=1)


def _call_ai_cached(
    *,
    data_dir: str,
    cache_key: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    cached = _cache_load(data_dir, cache_key)
    if cached:
        return {"status": "ok", "cached": True, "payload": cached}

    if cancel_event.is_set():
        return {"status": "cancelled"}

    try:
        payload = gemini_generate_json(
            messages=messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if cancel_event.is_set():
        return {"status": "cancelled"}

    if isinstance(payload, dict):
        _cache_save(data_dir, cache_key, payload)
    return {"status": "ok", "cached": False, "payload": payload}


# =============================================================================
# Quick Draft (AI-only universe)
# =============================================================================
def quick_draft_job(*, inp: ThemeInput, data_dir: str, cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())
    pool = _candidate_pool(top_n)

    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"

    # 1) candidates + TRR
    key1 = _cache_key("cand_v12", theme, region, asof.isoformat(), str(pool), model)
    msgs = build_messages_candidates(theme, region, asof, top_n, pool)

    call1 = _call_ai_cached(
        data_dir=data_dir,
        cache_key=key1,
        messages=msgs,
        model=model,
        temperature=0.05,
        max_tokens=1600,
        timeout_s=12,
        cancel_event=cancel_event,
    )
    if call1.get("status") != "ok":
        return {"status": call1.get("status","error"), "stage": "candidates", "notes": call1.get("error","")}

    raw1 = call1.get("payload", {})
    cand = _normalize_candidates(raw1)

    # retry with minimal schema if JSON shape is bad or empty companies
    if cand.get("status") == "error" or (cand.get("status") == "ok" and not cand.get("companies")):
        key1b = _cache_key("cand_min_v12", theme, region, asof.isoformat(), str(pool), model)
        msgs2 = build_messages_candidates_minimal_retry(theme, region, asof, top_n, pool)
        call2 = _call_ai_cached(
            data_dir=data_dir,
            cache_key=key1b,
            messages=msgs2,
            model=model,
            temperature=0.05,
            max_tokens=900,
            timeout_s=10,
            cancel_event=cancel_event,
        )
        if call2.get("status") != "ok":
            return {"status": "error", "stage": "candidates_retry", "notes": f"AI output not usable. {call2.get('error','')}"}

        raw2 = call2.get("payload", {})
        norm2 = _normalize_candidates_minimal(raw2)
        if norm2.get("status") == "ambiguous":
            return {
                "status": "ambiguous",
                "asof": asof.isoformat(),
                "theme_definition": norm2.get("theme_definition",""),
                "reference_etfs": norm2.get("reference_etfs", []),
                "notes": "",
            }
        rows2 = norm2.get("rows", []) or []
        companies2 = []
        for r in rows2:
            companies2.append({
                "company_name": "",
                "ticker": str(r.get("ticker","") or "").strip().upper(),
                "listed_country": "",
                "primary_exchange": "",
                "theme_revenue_ratio": clamp01(r.get("theme_revenue_ratio", 0.0)),
                "theme_revenue_ratio_method": "estimated",
                "theme_revenue_ratio_confidence": "Low",
            })
        cand = {"status": "ok", "companies": companies2, "theme_definition": "", "notes": "Used minimal retry schema.", "reference_etfs": norm2.get("reference_etfs", [])}

    if cand.get("status") == "ambiguous":
        return {
            "status": "ambiguous",
            "asof": asof.isoformat(),
            "theme_definition": cand.get("theme_definition",""),
            "reference_etfs": cand.get("reference_etfs", []),
            "notes": cand.get("notes",""),
        }

    if cand.get("status") != "ok" or not cand.get("companies"):
        return {"status": "error", "stage": "candidates", "notes": "AI returned no candidates."}

    companies = cand.get("companies", [])
    tickers = [c["ticker"] for c in companies if c.get("ticker")]
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))

    # 2) month-end market cap ranking
    mkt = batch_mktcap_asof_usd(tickers, asof)
    thr = _large_cap_threshold_usd(region)

    mkt = mkt.dropna(subset=["mcap_usd"])
    mkt = mkt[mkt["mcap_usd"] >= thr]

    base = pd.DataFrame(companies)
    base["theme_revenue_ratio"] = pd.to_numeric(base["theme_revenue_ratio"], errors="coerce").fillna(0.0).clip(0,1)

    df = base.merge(mkt[["ticker","mcap_usd","mcap_quality","_country","_exchange","_shortName","_sector","_industry"]], on="ticker", how="inner")
    df["theme_mktcap_usd"] = pd.to_numeric(df["mcap_usd"], errors="coerce") * pd.to_numeric(df["theme_revenue_ratio"], errors="coerce")
    df = df.dropna(subset=["theme_mktcap_usd"])
    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df = df.head(top_n).copy()

    if df.empty:
        return {"status": "error", "stage": "ranking", "notes": "No rows survived market-cap filter. Try Global or broaden the theme."}

    # Fill missing names/country/exchange from yfinance info
    df["company_name"] = df.apply(lambda r: (r.get("company_name") or "") if str(r.get("company_name") or "").strip() else (r.get("_shortName") or r.get("ticker")), axis=1)
    df["listed_country"] = df.apply(lambda r: (r.get("listed_country") or "") if str(r.get("listed_country") or "").strip() else (r.get("_country") or ""), axis=1)
    df["primary_exchange"] = df.apply(lambda r: (r.get("primary_exchange") or "") if str(r.get("primary_exchange") or "").strip() else (r.get("_exchange") or ""), axis=1)

    # 3) Summaries for Top-N only (small call)
    top_tickers = df["ticker"].astype(str).tolist()
    key_sum = _cache_key("sum_v12", theme, region, asof.isoformat(), _sha12(",".join(top_tickers)), model)
    msgs_sum = build_messages_summaries(theme, region, top_tickers)

    sum_call = _call_ai_cached(
        data_dir=data_dir,
        cache_key=key_sum,
        messages=msgs_sum,
        model=model,
        temperature=0.15,
        max_tokens=900,
        timeout_s=10,
        cancel_event=cancel_event,
    )

    smap: Dict[str, Dict[str, str]] = {}
    if sum_call.get("status") == "ok":
        smap = _normalize_summaries(sum_call.get("payload", {}))

    def _fallback_theme_summary(t: str, sector: str, industry: str) -> str:
        if industry:
            return f"業種: {industry}。テーマ関連は推計（Quick Draft）。"
        if sector:
            return f"セクター: {sector}。テーマ関連は推計（Quick Draft）。"
        return "テーマ関連は推計（Quick Draft）。"

    def _fallback_non_theme_summary(t: str, sector: str) -> str:
        return f"非テーマ事業: {sector}中心の他領域（要確認）。" if sector else "非テーマ事業: 多角化（要確認）。"

    df["theme_business_summary"] = df["ticker"].apply(lambda t: smap.get(t, {}).get("theme_ja","").strip())
    df["non_theme_business_summary"] = df["ticker"].apply(lambda t: smap.get(t, {}).get("non_theme_ja","").strip())
    df["theme_business_summary"] = df.apply(lambda r: r["theme_business_summary"] if r["theme_business_summary"] else _fallback_theme_summary(r["ticker"], str(r.get("_sector") or ""), str(r.get("_industry") or "")), axis=1)
    df["non_theme_business_summary"] = df.apply(lambda r: r["non_theme_business_summary"] if r["non_theme_business_summary"] else _fallback_non_theme_summary(r["ticker"], str(r.get("_sector") or "")), axis=1)

    # Evidence placeholders (Refine fills)
    df["trr_source_title"] = ""
    df["trr_source_publisher"] = ""
    df["trr_source_year"] = None
    df["trr_source_url"] = None
    df["trr_source_locator"] = ""
    df["trr_source_excerpt"] = ""
    df["trr_sources_full"] = ""

    df["mktcap_asof_date"] = asof.isoformat()

    return {
        "status": "ok",
        "mode": "quick",
        "asof": asof.isoformat(),
        "ranked": df.to_dict(orient="records"),
        "notes": str(cand.get("notes","") or ""),
        "theme_definition": str(cand.get("theme_definition","") or ""),
        "reference_etfs": cand.get("reference_etfs", []) or [],
        "model": model,
        "threshold_usd": thr,
        "cache_keys": {"candidates": key1, "summaries": key_sum},
        "counts": {"ai_candidates": len(tickers), "ranked": len(df)},
    }


# =============================================================================
# Refine (Top-N evidence)
# =============================================================================
def refine_job(*, inp: ThemeInput, data_dir: str, current_rows: List[Dict[str, Any]], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"

    payload = []
    for r in current_rows[:top_n]:
        payload.append({
            "ticker": r.get("ticker"),
            "company_name": r.get("company_name"),
            "listed_country": r.get("listed_country"),
            "current_trr": r.get("theme_revenue_ratio"),
            "mcap_usd": r.get("mcap_usd"),
            "theme_mktcap_usd": r.get("theme_mktcap_usd"),
        })

    key_ref = _cache_key("refine_v12", theme, region, asof.isoformat(), _sha12(json.dumps(payload, ensure_ascii=False)), model)
    msgs = build_messages_refine_evidence(theme, region, asof, payload)

    call = _call_ai_cached(
        data_dir=data_dir,
        cache_key=key_ref,
        messages=msgs,
        model=model,
        temperature=0.10,
        max_tokens=2400,
        timeout_s=16,
        cancel_event=cancel_event,
    )
    if call.get("status") != "ok":
        return {"status": "error", "stage": "refine", "notes": call.get("error","")}

    norm = _normalize_refine(call.get("payload", {}))
    if norm.get("status") != "ok":
        return {"status": "error", "stage": "refine", "notes": norm.get("notes","")}

    upd = {r["ticker"]: r for r in (norm.get("rows") or []) if isinstance(r, dict) and r.get("ticker")}
    df = pd.DataFrame(current_rows)
    if df.empty:
        return {"status": "error", "stage": "merge", "notes": "Current rows empty."}

    def u(t: str, k: str, default: Any) -> Any:
        return upd.get(t, {}).get(k, default)

    df["theme_revenue_ratio"] = df["ticker"].apply(lambda t: clamp01(u(t, "theme_revenue_ratio", df.loc[df["ticker"]==t, "theme_revenue_ratio"].values[0])))
    df["theme_revenue_ratio_method"] = df["ticker"].apply(lambda t: str(u(t, "theme_revenue_ratio_method", df.loc[df["ticker"]==t, "theme_revenue_ratio_method"].values[0])))
    df["theme_revenue_ratio_confidence"] = df["ticker"].apply(lambda t: str(u(t, "theme_revenue_ratio_confidence", df.loc[df["ticker"]==t, "theme_revenue_ratio_confidence"].values[0])))

    df["theme_business_summary"] = df["ticker"].apply(lambda t: str(u(t, "theme_business_summary_ja", df.loc[df["ticker"]==t, "theme_business_summary"].values[0]) or ""))
    df["non_theme_business_summary"] = df["ticker"].apply(lambda t: str(u(t, "non_theme_business_summary_ja", df.loc[df["ticker"]==t, "non_theme_business_summary"].values[0]) or ""))

    def first_source(t: str) -> Dict[str, Any]:
        srcs = u(t, "theme_revenue_sources", [])
        if isinstance(srcs, list) and srcs and isinstance(srcs[0], dict):
            return srcs[0]
        return {}

    df["trr_source_title"] = df["ticker"].apply(lambda t: str(first_source(t).get("title","") or ""))
    df["trr_source_publisher"] = df["ticker"].apply(lambda t: str(first_source(t).get("publisher","") or ""))
    df["trr_source_year"] = df["ticker"].apply(lambda t: first_source(t).get("year", None))
    df["trr_source_url"] = df["ticker"].apply(lambda t: first_source(t).get("url", None))
    df["trr_source_locator"] = df["ticker"].apply(lambda t: str(first_source(t).get("locator","") or ""))
    df["trr_source_excerpt"] = df["ticker"].apply(lambda t: str(first_source(t).get("excerpt","") or ""))
    df["trr_sources_full"] = df["ticker"].apply(lambda t: json.dumps(u(t, "theme_revenue_sources", []), ensure_ascii=False))

    # recompute and rerank
    df["theme_mktcap_usd"] = pd.to_numeric(df.get("mcap_usd"), errors="coerce") * pd.to_numeric(df.get("theme_revenue_ratio"), errors="coerce")
    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return {
        "status": "ok",
        "mode": "refine",
        "asof": asof.isoformat(),
        "ranked": df.to_dict(orient="records"),
        "notes": str(norm.get("notes","") or ""),
        "model": model,
        "cache_keys": {"refine": key_ref},
    }


# =============================================================================
# Definitions (always displayed)
# =============================================================================
def build_definitions_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Field": "Theme Market Cap", "Definition": "Theme Market Cap = Free-float Market Cap (proxy, USD, as-of month-end) × TRR"},
        {"Field": "TRR (Theme Revenue Ratio)", "Definition": "TRR = テーマ関連売上 ÷ 総売上（0〜1）"},
        {"Field": "Free-float Market Cap (proxy)", "Definition": "原則: floatShares×月末Close。欠損時: sharesOutstanding×Close / derived shares / marketCap を近似利用"},
        {"Field": "As-of date", "Definition": "時価総額の基準日（原則：直近月末）"},
        {"Field": "Method", "Definition": "disclosed=一次情報で明確 / proxy=代理指標 / estimated=推計（推計は明示）"},
        {"Field": "Confidence", "Definition": "High/Med/Low（根拠の強さ。Lowは推計寄り）"},
        {"Field": "Sources (Refine)", "Definition": "title/publisher/year/url/locator/excerpt。urlが無い場合 excerpt は空文字"},
    ])


# =============================================================================
# UI
# =============================================================================
def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(THEMELENS_CSS, unsafe_allow_html=True)
    st.markdown('<div class="tl-title">THEMELENS</div>', unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("tl_rows", None)
    ss.setdefault("tl_meta", {})
    ss.setdefault("tl_job", None)
    ss.setdefault("tl_future", None)
    ss.setdefault("tl_cancel_event", None)

    # Top controls (no sidebar)
    st.markdown('<div class="tl-panel">', unsafe_allow_html=True)
    with st.form("tl_controls", clear_on_submit=False):
        # Row 1: theme
        theme_text = st.text_input("Theme", value=ss.get("tl_theme_text", "半導体"), placeholder="例: 半導体 / ゲーム / サイバー / 生成AI / 防衛 ...")
        # Row 2: region + N + buttons
        c1, c2, c3, c4 = st.columns([2.2, 1.0, 1.2, 1.2])
        with c1:
            region_mode = st.radio("Region", ["Global","Japan","US","Europe","China"], horizontal=True,
                                   index=["Global","Japan","US","Europe","China"].index(ss.get("tl_region_mode","Global")))
        with c2:
            top_n = st.number_input("Top N", min_value=1, max_value=30, value=int(ss.get("tl_top_n", 10)), step=1)
        with c3:
            quick = st.form_submit_button("Quick Draft", type="primary", use_container_width=True)
        with c4:
            refine = st.form_submit_button("Refine", type="secondary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ss["tl_theme_text"] = theme_text
    ss["tl_region_mode"] = region_mode
    ss["tl_top_n"] = int(top_n)

    if refine and not ss.get("tl_rows"):
        st.warning("Run Quick Draft first.")
        refine = False

    inp = ThemeInput(theme_text=str(theme_text or "").strip(), region_mode=region_mode, top_n=int(top_n))

    # Start jobs
    if quick:
        ss["tl_rows"] = None
        ss["tl_meta"] = {}
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "quick", "status": "running", "started_at": time.time()}
        ss["tl_future"] = _executor().submit(quick_draft_job, inp=inp, data_dir=data_dir, cancel_event=cancel_event)
        _rerun()

    if refine:
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "refine", "status": "running", "started_at": time.time()}
        current_rows = ss.get("tl_rows") or []
        ss["tl_future"] = _executor().submit(refine_job, inp=inp, data_dir=data_dir, current_rows=current_rows, cancel_event=cancel_event)
        _rerun()

    # Running state
    job = ss.get("tl_job")
    fut = ss.get("tl_future")
    if job and job.get("status") == "running" and fut is not None:
        elapsed = time.time() - float(job.get("started_at", time.time()))
        mode = job.get("mode", "quick")

        if mode == "quick":
            st.info("Quick Draft: Gemini proposes candidates + TRR, then we re-rank by month-end Theme Market Cap.")
        else:
            st.info("Refine: add TRR evidence (sources) and improve summaries, then re-rank.")

        cA, cB, cC = st.columns([1.0, 1.0, 3.0])
        with cA:
            if st.button("Cancel", use_container_width=True):
                ev = ss.get("tl_cancel_event")
                if ev:
                    try:
                        ev.set()
                    except Exception:
                        pass
                try:
                    fut.cancel()
                except Exception:
                    pass
                ss["tl_job"] = None
                ss["tl_future"] = None
                st.warning("Cancelled.")
                st.stop()
        with cB:
            st.metric("Elapsed", f"{elapsed:.1f}s")
        with cC:
            st.caption("If it takes too long: Cancel → Quick Draft again. (Caching helps on repeated runs.)")

        budget = 18 if mode == "quick" else 26
        if elapsed > budget:
            st.warning("This is taking longer than expected. You can cancel and retry.")

        if fut.done():
            try:
                res = fut.result()
            except Exception as e:
                ss["tl_job"] = None
                ss["tl_future"] = None
                st.error(f"Job failed: {e}")
                st.stop()

            ss["tl_job"] = None
            ss["tl_future"] = None
            if isinstance(res, dict):
                ss["tl_meta"] = res
                ss["tl_rows"] = res.get("ranked") if res.get("status") == "ok" else None
            else:
                ss["tl_meta"] = {"status": "error", "notes": "Unexpected result type."}
                ss["tl_rows"] = None
            _rerun()

        time.sleep(0.45)
        _rerun()

    # Output
    meta = ss.get("tl_meta") or {}
    rows = ss.get("tl_rows")

    # Ambiguous theme
    if isinstance(meta, dict) and meta.get("status") == "ambiguous":
        st.error("Theme is ambiguous / not investable (boundary unclear).")
        if meta.get("theme_definition"):
            st.write(str(meta.get("theme_definition")))
        ref = meta.get("reference_etfs", []) or []
        if ref:
            st.caption("Reference ETFs: " + ", ".join([str(x) for x in ref[:8]]))
        if meta.get("notes"):
            st.caption(str(meta.get("notes")))
        return

    # Error state
    if isinstance(meta, dict) and meta.get("status") == "error" and not rows:
        st.error("Could not generate a ranked list.")
        note = str(meta.get("notes","") or "")
        if note:
            st.write(note)
        st.caption("Tip: try a clearer theme phrase, or broaden region to Global.")
        return

    # No list yet
    if not rows:
        asof = most_recent_month_end(date.today())
        st.caption(f"As-of (market cap): {asof.isoformat()}  •  Click Quick Draft to generate the ranked list.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No rows returned.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("As-of (MktCap)", str(df.get("mktcap_asof_date", pd.Series([most_recent_month_end(date.today()).isoformat()])).iloc[0]))
    with k2:
        st.metric("Region", region_mode)
    with k3:
        st.metric("Top N", int(top_n))
    with k4:
        st.metric("Mode", str(meta.get("mode","")))

    notes = str(meta.get("notes","") or "").strip()
    if notes:
        st.caption(notes)

    # Deliverable
    st.subheader("Deliverable: Ranked List (Theme Market Cap, desc)")

    view = df.copy()
    view["TRR"] = (pd.to_numeric(view["theme_revenue_ratio"], errors="coerce") * 100).round(1).astype(str) + "%"
    view["Free-float MktCap (USD)"] = pd.to_numeric(view["mcap_usd"], errors="coerce").apply(fmt_money)
    view["Theme MktCap (USD)"] = pd.to_numeric(view["theme_mktcap_usd"], errors="coerce").apply(fmt_money)

    # keep table readable; evidence details are in expander
    show_cols = [
        "rank",
        "company_name",
        "ticker",
        "listed_country",
        "theme_business_summary",
        "TRR",
        "Free-float MktCap (USD)",
        "Theme MktCap (USD)",
        "non_theme_business_summary",
        "trr_source_title",
        "trr_source_publisher",
        "trr_source_year",
        "trr_source_locator",
    ]
    existing = [c for c in show_cols if c in view.columns]
    st.dataframe(view[existing], use_container_width=True, height=560)

    # Exports
    asof = str(df.get("mktcap_asof_date", pd.Series([most_recent_month_end(date.today()).isoformat()])).iloc[0])
    sid = _sha12(f"{theme_text}|{region_mode}|{asof}|{top_n}")
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"themelens_ranked_list_{sid}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download JSON (snapshot)",
        data=json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"themelens_snapshot_{sid}.json",
        mime="application/json",
    )

    # Evidence details (shows method/confidence too, without cluttering the main table)
    with st.expander("Evidence details (TRR)", expanded=False):
        for _, r in df.sort_values("rank").iterrows():
            conf = str(r.get("theme_revenue_ratio_confidence","") or "")
            method = str(r.get("theme_revenue_ratio_method","") or "")
            title = f"{int(r['rank'])}. {r.get('company_name','')} ({r.get('ticker','')}) — TRR {float(r.get('theme_revenue_ratio',0))*100:.1f}% [{method}/{conf}]"
            with st.expander(title, expanded=False):
                st.write("**Theme business**")
                st.write(r.get("theme_business_summary","") or "-")
                st.write("**Non-theme business**")
                st.write(r.get("non_theme_business_summary","") or "-")
                st.divider()
                st.write("**TRR / Evidence**")
                st.json({
                    "TRR": r.get("theme_revenue_ratio"),
                    "method": r.get("theme_revenue_ratio_method"),
                    "confidence": r.get("theme_revenue_ratio_confidence"),
                    "source_title": r.get("trr_source_title"),
                    "publisher": r.get("trr_source_publisher"),
                    "year": r.get("trr_source_year"),
                    "url": r.get("trr_source_url"),
                    "locator": r.get("trr_source_locator"),
                    "excerpt": r.get("trr_source_excerpt"),
                    "all_sources": r.get("trr_sources_full"),
                    "mktcap_quality": r.get("mcap_quality"),
                })

    # Definitions (always visible)
    st.markdown("### Definitions")
    st.markdown(
        "- **Theme Market Cap** = *Free-float Market Cap (proxy, USD, month-end)* × **TRR**\n"
        "- **TRR (Theme Revenue Ratio)** = テーマ関連売上 ÷ 総売上（0〜1）\n"
        "- **As-of date** = 時価総額の基準日（原則：直近月末）\n"
        "- **Refine** = TRRの根拠（ソース/locators）を付与し、必要ならTRRと要約を更新して再ランキング"
    )
    st.dataframe(build_definitions_table(), use_container_width=True, height=320)
