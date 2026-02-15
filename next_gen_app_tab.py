# THEMELENS — AI Theme Portfolio Builder (Gemini, Mobile-first v9)
# -----------------------------------------------------------------------------
# Design goals (latest):
# - Match ALPHALENS look: minimal header (no long description block).
# - NO sidebar. Controls at top.
# - Data rigor / Confidence are NOT user controls; the AI labels them per-name.
# - Deliverable: ranked list by Theme Market Cap (month-end, free-float proxy).
# - Speed: aggressive caching + smaller default candidate pools + timeout + cancel.
#
# How to call from app.py:
#   from next_gen_app_tab import render_next_gen_tab
#   render_next_gen_tab(data_dir="data")
#
# Secrets / env (Gemini):
#   GEMINI_API_KEY or GOOGLE_API_KEY
#   GEMINI_MODEL (optional; defaults to a fast model)
#
# NOTE:
# - This module asks Gemini to propose candidate tickers & TRR (Theme Revenue Ratio) with evidence metadata.
# - We do the market-cap pull + re-ranking locally (month-end market cap proxy via yfinance).

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import time
import threading

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# =============================================================================
# Styling (keep it light; do NOT override padding-top -> avoids mobile tab click issues)
# =============================================================================
ALPHALENS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

:root{
  --al-cyan:#00f2fe;
  --al-muted:rgba(255,255,255,0.70);
  --al-border:rgba(255,255,255,0.10);
  --al-bg:rgba(0,0,0,0.58);
}

div[data-testid="stAppViewContainer"] .block-container{
  padding-bottom: 2.0rem;
  max-width: 1400px;
}

/* Minimal, AlphaLens-like header */
.themelens-title{
  font-family:Orbitron, sans-serif;
  letter-spacing:0.12em;
  color:var(--al-cyan);
  margin: 0 0 2px 0;
  font-size: 28px;
}
.themelens-sub{
  margin: 0 0 14px 0;
  color: var(--al-muted);
  font-size: 12.5px;
}

/* Light glass panel */
.tl-panel{
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: var(--al-bg);
  backdrop-filter: blur(10px);
  padding: 12px 12px;
}

/* DataFrame polish */
[data-testid="stDataFrame"]{
  border-radius:14px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.07);
}

.stButton>button{
  border-radius: 14px;
  font-weight: 800;
  letter-spacing: 0.03em;
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
    fast_mode: bool = True  # speed vs depth trade-off (still AI-first)


# =============================================================================
# Helpers
# =============================================================================
def _get_secret(key: str) -> Optional[str]:
    # streamlit secrets -> env fallback
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


def _sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


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
        return "China large-caps: Mainland A-shares (Shanghai/Shenzhen) + Hong Kong listed China large-caps."
    return "Global"


# =============================================================================
# Market data (free-float-ish market cap proxy)
# =============================================================================
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _yf_download_cached(tickers: Tuple[str, ...], start: str, end: str, auto_adjust: bool, group_by: str = "ticker") -> pd.DataFrame:
    return yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        group_by=group_by,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _ticker_info_cached(ticker: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


class MarketProviderYF:
    def info(self, ticker: str) -> Dict[str, Any]:
        return _ticker_info_cached(ticker)

    def fx_usd_per_ccy(self, ccy: str, asof: date) -> Tuple[Optional[float], str]:
        ccy = (ccy or "USD").upper()
        if ccy == "USD":
            return 1.0, "exact"

        candidates = [f"{ccy}USD=X", f"USD{ccy}=X"]
        for sym in candidates:
            try:
                df = _yf_download_cached((sym,), start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False, group_by="column")
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].dropna()
                if close.empty:
                    continue
                rate = float(close.iloc[-1])
                if not (rate > 0):
                    continue
                usd_per_ccy = rate
                if sym.startswith("USD") or rate > 20:
                    usd_per_ccy = 1.0 / rate
                if 0 < usd_per_ccy < 100:
                    return float(usd_per_ccy), f"best_effort({sym})"
            except Exception:
                continue
        return None, "unavailable"

    def _close_on_or_before(self, ticker: str, asof: date) -> Optional[float]:
        try:
            df = _yf_download_cached((ticker,), start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False, group_by="column")
            if df is None or df.empty or "Close" not in df.columns:
                return None
            close = df["Close"].dropna()
            if close.empty:
                return None
            return float(close.iloc[-1])
        except Exception:
            return None

    def free_float_mktcap_asof(self, ticker: str, asof: date) -> Tuple[Optional[float], str, Optional[float], str]:
        info = self.info(ticker)
        currency = str(info.get("currency") or "USD").upper()

        px = self._close_on_or_before(ticker, asof)
        if px is None or px <= 0:
            # fallback: yfinance marketCap (current-ish)
            mcap = info.get("marketCap")
            if mcap and float(mcap) > 0:
                usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
                mcap_usd = (float(mcap) * usd_per_ccy) if usd_per_ccy else None
                return float(mcap), currency, mcap_usd, f"proxy_field(marketCap),{q}"
            return None, currency, None, "missing_price"

        float_shares = info.get("floatShares")
        shares_out = info.get("sharesOutstanding")

        if float_shares and float(float_shares) > 0:
            mcap_local = float(float_shares) * px
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_free_float(floatShares×Close),{q}"

        if shares_out and float(shares_out) > 0:
            mcap_local = float(float(shares_out) * px)
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_total(sharesOutstanding×Close),{q}"

        mcap = info.get("marketCap")
        if mcap and float(mcap) > 0:
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (float(mcap) * usd_per_ccy) if usd_per_ccy else None
            return float(mcap), currency, mcap_usd, f"proxy_field(marketCap),{q}"

        return None, currency, None, "missing_shares"


# =============================================================================
# Gemini JSON call (best-effort compatibility)
# =============================================================================
def _parse_json_from_text(txt: str) -> Dict[str, Any]:
    if not txt:
        raise ValueError("Empty model response.")
    # attempt direct json
    try:
        return json.loads(txt)
    except Exception:
        pass
    # extract first {...} block
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response.")
    return json.loads(m.group(0))


def _get_google_genai_client(api_key: Optional[str]):
    # Vertex AI mode can work without api_key
    try:
        from google import genai  # type: ignore
        if api_key:
            return genai.Client(api_key=api_key)
        return genai.Client()
    except Exception:
        return None


def gemini_generate_json(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    """
    Preference order:
      1) google-genai SDK (if installed)
      2) legacy google.generativeai (if installed + api_key)
      3) REST (requires api_key)
    """
    system_txt = ""
    user_txt_parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower().strip()
        content = str(m.get("content") or "")
        if role == "system" and not system_txt:
            system_txt = content
        elif role == "user":
            user_txt_parts.append(content)

    user_txt = "\n\n".join([p for p in user_txt_parts if p]).strip()
    if not user_txt:
        raise RuntimeError("Gemini call: missing user prompt text.")

    api_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
    last_err: Optional[Exception] = None

    # 1) google-genai
    try:
        from google.genai import types  # type: ignore
        client = _get_google_genai_client(api_key)
        if client is None:
            raise RuntimeError("google-genai client unavailable")

        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
            system_instruction=system_txt or None,
            response_mime_type="application/json",
            max_output_tokens=int(max_output_tokens),
        )
        # NOTE: SDK may not honor timeout; keep this for compatibility.
        resp = client.models.generate_content(model=model, contents=user_txt, config=cfg)
        txt = getattr(resp, "text", "") or ""
        if callable(txt):
            txt = txt()
        return _parse_json_from_text(str(txt))
    except Exception as e:
        last_err = e

    # 2) legacy google.generativeai
    try:
        import google.generativeai as genai_old  # type: ignore
        if not api_key:
            raise RuntimeError("legacy google.generativeai requires api_key")
        genai_old.configure(api_key=api_key)
        generation_config = {
            "temperature": float(temperature),
            "response_mime_type": "application/json",
            "max_output_tokens": int(max_output_tokens),
        }
        model_obj = genai_old.GenerativeModel(
            model_name=model,
            system_instruction=system_txt or None,
            generation_config=generation_config,
        )
        resp = model_obj.generate_content(user_txt)
        txt = getattr(resp, "text", "") or ""
        return _parse_json_from_text(str(txt))
    except Exception as e:
        last_err = e

    # 3) REST fallback
    if not api_key:
        raise RuntimeError(
            "Gemini is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY, or install google-genai for Vertex AI. "
            f"Last error: {last_err}"
        )

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
            txt = parts[0].get("text", "") or ""
    except Exception:
        txt = ""
    return _parse_json_from_text(str(txt))


# =============================================================================
# AI Prompt (reworked: no data-rigor controls; AI labels method/confidence itself)
# =============================================================================
def _ai_system_prompt() -> str:
    return (
        "You are a senior global equity portfolio manager and sector analyst. "
        "You build investable theme portfolios for institutional clients. "
        "Be conservative: NEVER invent URLs or quotes. "
        "If you cannot provide a verified URL and exact excerpt, set url=null and excerpt='' and label the TRR as method='estimated' with confidence='Low'. "
        "Return ONLY valid JSON. No markdown."
    )


def _candidate_pool(top_n: int, fast: bool) -> int:
    n = max(1, min(int(top_n), 30))
    if fast:
        return int(min(45, max(25, 2 * n + 6)))
    return int(min(80, max(60, 3 * n + 10)))


def build_ai_messages(inp: ThemeInput, asof_mktcap_date: date) -> List[Dict[str, str]]:
    region_def = _region_definition(inp.region_mode)

    pool = _candidate_pool(inp.top_n, inp.fast_mode)

    schema = {
        "status": "ok | ambiguous | error",
        "theme_definition": "string",
        "reference_etfs": ["string"],
        "notes": "string",
        "companies": [
            {
                "company_name": "string",
                "ticker": "string (yfinance-compatible; include suffix like 0700.HK, 9984.T, NESN.SW, HSBA.L when needed)",
                "listed_country": "string",
                "primary_exchange": "string",
                "theme_business_summary": "string (Japanese; 1-2 sentences)",
                "non_theme_business_summary": "string (Japanese; 1 sentence)",
                "theme_revenue_ratio": "number (0-1)",
                "theme_revenue_ratio_year": "number or null",
                "theme_revenue_ratio_method": "disclosed | estimated | proxy",
                "theme_revenue_ratio_confidence": "High | Med | Low",
                "theme_revenue_sources": [
                    {
                        "source_title": "string",
                        "publisher": "string",
                        "year": "number or null",
                        "url": "string or null",
                        "locator": "string (page/section/table)",
                        "excerpt": "string (<=25 words; exact quote ONLY if url is present)",
                    }
                ],
                "theme_profit_ratio": "number (0-1) or null",
                "theme_profit_ratio_year": "number or null",
            }
        ],
    }

    user_prompt = f"""
# Task
Propose an investable large-cap candidate list for a theme portfolio (institutional-grade).

## Inputs
- Theme text: "{inp.theme_text}"
- Region filter: {inp.region_mode}
- Region definition: {region_def}
- Target Top N: {inp.top_n}
- Candidate pool to return: {pool}
- Market-cap as-of date: {asof_mktcap_date.isoformat()}

## Universe / size constraints (IMPORTANT)
- Only large, liquid listed companies typical of major indices in the region.
- Do NOT include small caps or obscure listings.

## Deliverable concept (IMPORTANT)
We will compute month-end market caps locally and re-rank by:
Theme Market Cap = Free-float Market Cap (proxy) × TRR
So you MUST provide TRR for each company (0-1).

## TRR rules (commercial-grade)
- TRR (Theme Revenue Ratio) = (revenue from theme-related business) / (total company revenue).
- If no reliable disclosure exists, you may estimate, BUT you must label:
  - theme_revenue_ratio_method = "estimated"
  - theme_revenue_ratio_confidence = "Low"
  - url = null and excerpt = ""
- Prefer evidence in this order:
  1) Annual report / 10-K / 20-F / regulatory filing segment notes
  2) Official earnings deck / IR presentation (segment mix tables)
  3) Company website segment revenue mix (only if specific)
  4) Reputable third-party research (only if primary not available)
  5) Estimation (last resort; must be clearly labeled)

## Theme ambiguity handling
If the theme is too vague / not investable (boundary unclear), return:
- status="ambiguous"
- theme_definition: why it is ambiguous
- reference_etfs: suggest 3-8 closest ETFs (tickers)
- companies: []

## Output format (STRICT)
Return ONLY JSON with this exact schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    return [
        {"role": "system", "content": _ai_system_prompt()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def _normalize_ai_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "notes": "LLM output not a JSON object", "companies": []}

    status = str(obj.get("status", "error"))
    if status not in ("ok", "ambiguous", "error"):
        status = "error"

    companies = obj.get("companies", [])
    if not isinstance(companies, list):
        companies = []

    # minimal cleanup
    norm_companies: List[Dict[str, Any]] = []
    for c in companies:
        if not isinstance(c, dict):
            continue
        t = str(c.get("ticker", "") or "").strip()
        if not t:
            continue
        c2 = dict(c)
        c2["ticker"] = t
        c2["company_name"] = str(c.get("company_name", "") or "").strip()
        c2["listed_country"] = str(c.get("listed_country", "") or "").strip()
        c2["primary_exchange"] = str(c.get("primary_exchange", "") or "").strip()
        c2["theme_business_summary"] = str(c.get("theme_business_summary", "") or "").strip()
        c2["non_theme_business_summary"] = str(c.get("non_theme_business_summary", "") or "").strip()
        c2["theme_revenue_ratio"] = clamp01(c.get("theme_revenue_ratio", 0.0))
        c2["theme_revenue_ratio_method"] = str(c.get("theme_revenue_ratio_method", "estimated") or "estimated")
        c2["theme_revenue_ratio_confidence"] = str(c.get("theme_revenue_ratio_confidence", "Low") or "Low")
        if not isinstance(c2.get("theme_revenue_sources", []), list):
            c2["theme_revenue_sources"] = []
        norm_companies.append(c2)

    obj2 = dict(obj)
    obj2["status"] = status
    obj2["companies"] = norm_companies
    if "reference_etfs" in obj2 and not isinstance(obj2["reference_etfs"], list):
        obj2["reference_etfs"] = []
    return obj2


# =============================================================================
# Local caching for AI outputs (speed)
# =============================================================================
def _ai_cache_dir(data_dir: str) -> Path:
    p = Path(data_dir) / "ai_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ai_cache_key(theme: str, region: str, pool: int, model: str, fast: bool) -> str:
    base = f"{theme}|{region}|{pool}|{model}|fast={int(fast)}"
    return _sha12(base)


def _ai_cache_load(data_dir: str, key: str, max_age_days: int = 30) -> Optional[Dict[str, Any]]:
    path = _ai_cache_dir(data_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        ts = obj.get("__cached_at__", None)
        if ts:
            cached_at = datetime.fromisoformat(ts)
            age = datetime.utcnow() - cached_at
            if age > timedelta(days=max_age_days):
                return None
        return obj.get("payload")
    except Exception:
        return None


def _ai_cache_save(data_dir: str, key: str, payload: Dict[str, Any]) -> None:
    path = _ai_cache_dir(data_dir) / f"{key}.json"
    obj = {"__cached_at__": datetime.utcnow().isoformat(timespec="seconds"), "payload": payload}
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # fail silently (cache is optional)
        pass


# =============================================================================
# Async execution (Cancel/Timeout)
# =============================================================================
@st.cache_resource
def _executor() -> cf.ThreadPoolExecutor:
    return cf.ThreadPoolExecutor(max_workers=1)


def _ai_worker(
    *,
    inp: ThemeInput,
    asof: date,
    data_dir: str,
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    # Load cache first (instant)
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    pool = _candidate_pool(inp.top_n, inp.fast_mode)
    key = _ai_cache_key(inp.theme_text.strip(), inp.region_mode, pool, model, inp.fast_mode)
    cached = _ai_cache_load(data_dir, key)
    if cached:
        return {"status": "ok", "cached": True, "ai": cached, "cache_key": key, "model": model, "pool": pool}

    if cancel_event.is_set():
        return {"status": "cancelled"}

    # AI call
    messages = build_ai_messages(inp, asof)
    # Speed knobs
    temperature = 0.15 if inp.fast_mode else 0.25
    max_tokens = 2600 if inp.fast_mode else 5200
    timeout_s = 18 if inp.fast_mode else 40  # hard request timeout (REST). SDK may not honor.

    ai_raw = gemini_generate_json(
        messages=messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout_s=timeout_s,
    )
    if cancel_event.is_set():
        return {"status": "cancelled"}

    ai = _normalize_ai_output(ai_raw)
    _ai_cache_save(data_dir, key, ai)
    return {"status": "ok", "cached": False, "ai": ai, "cache_key": key, "model": model, "pool": pool}


# =============================================================================
# Deliverable build (ranking)
# =============================================================================
def _large_cap_threshold_usd(region: RegionMode) -> float:
    # "major index constituents size" proxy; keep simple.
    # You can tune via data snapshot later (commercial).
    if region in ("US", "Global"):
        return 10e9
    if region == "Japan":
        return 5e9
    if region == "Europe":
        return 8e9
    if region == "China":
        return 8e9
    return 8e9


def build_ranked_list_from_ai(ai: Dict[str, Any], inp: ThemeInput, asof: date) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if ai.get("status") != "ok":
        return pd.DataFrame(), warnings

    companies = ai.get("companies", []) or []
    if not companies:
        return pd.DataFrame(), ["AI returned zero companies."]

    market = MarketProviderYF()
    thr = _large_cap_threshold_usd(inp.region_mode)

    rows: List[Dict[str, Any]] = []
    for c in companies:
        ticker = str(c.get("ticker") or "").strip()
        if not ticker:
            continue

        # yfinance likes uppercase, but keep suffix case
        ticker_norm = ticker.upper()

        mcap_local, ccy, mcap_usd, mcap_q = market.free_float_mktcap_asof(ticker_norm, asof)
        trr = clamp01(c.get("theme_revenue_ratio", 0.0))
        theme_mcap = (mcap_usd * trr) if (mcap_usd is not None) else None

        srcs = c.get("theme_revenue_sources", []) or []
        src0 = srcs[0] if (isinstance(srcs, list) and srcs) else {}

        rows.append({
            "company_name": str(c.get("company_name") or "").strip(),
            "ticker": ticker_norm,
            "listed_country": str(c.get("listed_country") or "").strip(),
            "primary_exchange": str(c.get("primary_exchange") or "").strip(),

            "theme_business_summary": str(c.get("theme_business_summary") or "").strip(),
            "non_theme_business_summary": str(c.get("non_theme_business_summary") or "").strip(),

            "theme_revenue_ratio": float(trr),
            "theme_revenue_ratio_year": c.get("theme_revenue_ratio_year", None),
            "theme_revenue_ratio_method": str(c.get("theme_revenue_ratio_method") or "").strip(),
            "theme_revenue_ratio_confidence": str(c.get("theme_revenue_ratio_confidence") or "").strip(),

            "trr_source_title": str(src0.get("source_title") or "").strip(),
            "trr_source_publisher": str(src0.get("publisher") or "").strip(),
            "trr_source_year": src0.get("year", None),
            "trr_source_url": src0.get("url", None),
            "trr_source_locator": str(src0.get("locator") or "").strip(),
            "trr_source_excerpt": str(src0.get("excerpt") or "").strip(),

            "mktcap_asof_date": asof.isoformat(),
            "free_float_mktcap_usd": mcap_usd,
            "free_float_mktcap_ccy": ccy,
            "free_float_mktcap_local": mcap_local,
            "mktcap_quality": mcap_q,
            "theme_mktcap_usd": theme_mcap,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, ["No usable rows after normalization."]

    df["free_float_mktcap_usd"] = pd.to_numeric(df["free_float_mktcap_usd"], errors="coerce")
    df["theme_mktcap_usd"] = pd.to_numeric(df["theme_mktcap_usd"], errors="coerce")

    # Large-cap filter (avoid tiny names)
    before = len(df)
    df = df.dropna(subset=["free_float_mktcap_usd", "theme_mktcap_usd"])
    df = df[df["free_float_mktcap_usd"] >= float(thr)]
    after = len(df)
    if after < before:
        warnings.append(f"Filtered out {before-after} names below large-cap threshold (${thr/1e9:.0f}B).")

    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    if len(df) < int(inp.top_n):
        warnings.append(f"Only {len(df)} names survived the market-cap filter (requested Top {int(inp.top_n)}).")

    df = df.head(int(inp.top_n)).copy()
    return df, warnings


def build_definitions_table() -> pd.DataFrame:
    items = [
        {"Field": "company_name", "Definition": "銘柄の一次上場の発行体名（AI提案）"},
        {"Field": "listed_country", "Definition": "一次上場国（Regionフィルタ）"},
        {"Field": "ticker", "Definition": "yfinance互換ティッカー（国/市場サフィックスを含む例: 0700.HK, 9984.T, NESN.SW）"},
        {"Field": "theme_business_summary", "Definition": "テーマ該当事業の要約（日本語）"},
        {"Field": "theme_revenue_ratio (TRR)", "Definition": "TRR = テーマ関連売上 ÷ 総売上（0〜1）"},
        {"Field": "free_float_mktcap_usd", "Definition": "free-float時価総額の近似（原則: floatShares×月末Close。欠損時は sharesOutstanding×Close / marketCap を使用）"},
        {"Field": "theme_mktcap_usd", "Definition": "Theme Market Cap = free_float_mktcap_usd × TRR（降順でランキング）"},
        {"Field": "non_theme_business_summary", "Definition": "テーマ以外の主要事業の要約（日本語）"},
        {"Field": "theme_revenue_ratio_method", "Definition": "disclosed=一次情報で明確 / proxy=代理指標 / estimated=推計（推計は明示）"},
        {"Field": "theme_revenue_ratio_confidence", "Definition": "High/Med/Low（AIが根拠の強さに応じて付与。Lowは推計寄り）"},
        {"Field": "trr_source_*", "Definition": "TRR根拠メタデータ（title/publisher/year/url/locator/excerpt）。url不明なら null、excerpt空文字で運用"},
        {"Field": "mktcap_asof_date", "Definition": "時価総額基準日（原則：直近月末）"},
        {"Field": "mktcap_quality", "Definition": "時価総額算出の品質タグ（free_float/total/field 等の近似情報）"},
    ]
    return pd.DataFrame(items)


# =============================================================================
# UI
# =============================================================================
def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(ALPHALENS_CSS, unsafe_allow_html=True)

    # Minimal header (AlphaLens-like)
    st.markdown('<div class="themelens-title">THEMELENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="themelens-sub">AI Theme Portfolio Builder</div>', unsafe_allow_html=True)

    # Controls (top, no sidebar)
    st.markdown('<div class="tl-panel">', unsafe_allow_html=True)
    with st.form("tl_controls", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([2.2, 1.0, 1.0, 0.9])
        with c1:
            theme_text = st.text_input("Theme", value=st.session_state.get("tl_theme_text", "半導体"))
        with c2:
            region_mode = st.selectbox("Region", ["Global","Japan","US","Europe","China"],
                                      index=["Global","Japan","US","Europe","China"].index(st.session_state.get("tl_region_mode","Global")))
        with c3:
            top_n = st.slider("Top N", 1, 30, int(st.session_state.get("tl_top_n", 10)), 1)
        with c4:
            fast_mode = st.toggle("⚡ Fast", value=bool(st.session_state.get("tl_fast_mode", True)),
                                  help="Fast mode = smaller candidate pool & shorter model output (usually much faster).")
        run = st.form_submit_button("BUILD", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Persist UI state
    st.session_state["tl_theme_text"] = theme_text
    st.session_state["tl_region_mode"] = region_mode
    st.session_state["tl_top_n"] = int(top_n)
    st.session_state["tl_fast_mode"] = bool(fast_mode)

    # Job state
    if "tl_job" not in st.session_state:
        st.session_state["tl_job"] = None
    if "tl_future" not in st.session_state:
        st.session_state["tl_future"] = None
    if "tl_cancel_event" not in st.session_state:
        st.session_state["tl_cancel_event"] = None
    if "tl_ai" not in st.session_state:
        st.session_state["tl_ai"] = None
    if "tl_ranked_df" not in st.session_state:
        st.session_state["tl_ranked_df"] = None
    if "tl_warnings" not in st.session_state:
        st.session_state["tl_warnings"] = []

    today = date.today()
    asof = most_recent_month_end(today)

    # Start job on BUILD
    if run:
        # reset outputs
        st.session_state["tl_ai"] = None
        st.session_state["tl_ranked_df"] = None
        st.session_state["tl_warnings"] = []

        inp = ThemeInput(theme_text=theme_text.strip(), region_mode=region_mode, top_n=int(top_n), fast_mode=bool(fast_mode))
        # Try cache immediately before spinning up background job
        model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
        pool = _candidate_pool(inp.top_n, inp.fast_mode)
        key = _ai_cache_key(inp.theme_text, inp.region_mode, pool, model, inp.fast_mode)
        cached = _ai_cache_load(data_dir, key)

        if cached:
            st.session_state["tl_ai"] = cached
            st.session_state["tl_job"] = {"status": "done", "cached": True, "cache_key": key, "started_at": time.time()}
        else:
            # background job
            cancel_event = threading.Event()
            st.session_state["tl_cancel_event"] = cancel_event
            job = {
                "status": "running",
                "theme": inp.theme_text,
                "region": inp.region_mode,
                "top_n": inp.top_n,
                "fast": inp.fast_mode,
                "started_at": time.time(),
            }
            st.session_state["tl_job"] = job
            st.session_state["tl_future"] = _executor().submit(_ai_worker, inp=inp, asof=asof, data_dir=data_dir, cancel_event=cancel_event)

        st.experimental_rerun()

    # If running, show status + Cancel
    job = st.session_state.get("tl_job")
    fut = st.session_state.get("tl_future")

    if job and job.get("status") == "running" and fut is not None:
        elapsed = time.time() - float(job.get("started_at", time.time()))

        # Crisp English status (requested)
        st.info(
            "Generating theme universe with Gemini…  "
            "Then re-ranking by month-end Theme Market Cap = Free-float Market Cap × TRR.  "
            "TRR is source-linked when available; otherwise explicitly marked as an estimate."
        )

        cA, cB, cC = st.columns([1.0, 1.0, 2.2])
        with cA:
            if st.button("Cancel", use_container_width=True):
                ev = st.session_state.get("tl_cancel_event")
                if ev:
                    try:
                        ev.set()
                    except Exception:
                        pass
                try:
                    fut.cancel()
                except Exception:
                    pass
                st.session_state["tl_job"] = None
                st.session_state["tl_future"] = None
                st.warning("Cancelled. You can retry (Fast mode recommended).")
                st.stop()
        with cB:
            st.metric("Elapsed", f"{elapsed:.1f}s")
        with cC:
            st.caption("Tip: Fast mode usually completes in a few seconds. If your theme is vague, Gemini may take longer or return 'ambiguous'.")

        # Poll for completion (lightweight)
        if fut.done():
            try:
                res = fut.result()
            except Exception as e:
                st.session_state["tl_job"] = None
                st.session_state["tl_future"] = None
                st.error(f"AI generation failed: {e}")
                st.stop()

            st.session_state["tl_job"] = {"status": "done", **res, "finished_at": time.time()}
            st.session_state["tl_future"] = None
            if res.get("status") == "ok":
                st.session_state["tl_ai"] = res.get("ai")
            st.experimental_rerun()

        # Time budget: if too long, suggest cancel
        budget = 22 if bool(job.get("fast")) else 50
        if elapsed > budget:
            st.warning("This is taking longer than expected. Consider pressing Cancel and enabling ⚡ Fast, or making the theme more specific.")
            # stop auto-polling to avoid hammering reruns
            return

        time.sleep(0.7)
        st.experimental_rerun()

    # If we have AI output (cached or finished), build ranked list
    ai = st.session_state.get("tl_ai")
    if ai is None:
        st.caption(f"As-of (market cap): {asof.isoformat()}  •  Ready when you are.")
        return

    # Handle ambiguous theme
    if isinstance(ai, dict) and ai.get("status") == "ambiguous":
        st.error("Theme is ambiguous / not investable (boundary unclear).")
        st.write(ai.get("theme_definition", "") or "")
        ref_etfs = ai.get("reference_etfs", []) or []
        if ref_etfs:
            st.write("Reference ETFs (closest):")
            st.write(", ".join([str(x) for x in ref_etfs]))
        notes = ai.get("notes", "") or ""
        if notes:
            st.caption(notes)
        return

    if not isinstance(ai, dict) or ai.get("status") != "ok":
        st.error("AI did not return a usable candidate list.")
        st.json(ai)
        return

    # Build ranked list (local)
    with st.spinner("Re-ranking by month-end Theme Market Cap…"):
        df, warnings = build_ranked_list_from_ai(ai, ThemeInput(theme_text=theme_text.strip(), region_mode=region_mode, top_n=int(top_n), fast_mode=bool(fast_mode)), asof)
    st.session_state["tl_ranked_df"] = df
    st.session_state["tl_warnings"] = warnings

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("As-of (MktCap)", asof.isoformat())
    with k2:
        st.metric("Region", region_mode)
    with k3:
        st.metric("Top N", int(top_n))
    with k4:
        st.metric("Candidates", len(ai.get("companies", []) or []))

    if warnings:
        st.warning("\n".join([f"- {w}" for w in warnings[:6]]))

    if df is None or df.empty:
        st.error("No deliverable rows (after market-cap filter / missing data). Try Fast off, adjust theme, or broaden region.")
        return

    # Deliverable table (required fields)
    st.subheader("Deliverable: Ranked List (Theme Market Cap, desc)")

    # Display-friendly columns
    view = df.copy()
    view["free_float_mktcap_usd_fmt"] = view["free_float_mktcap_usd"].apply(fmt_money)
    view["theme_mktcap_usd_fmt"] = view["theme_mktcap_usd"].apply(fmt_money)
    view["theme_revenue_ratio_pct"] = (view["theme_revenue_ratio"] * 100).round(1).astype(str) + "%"

    # Keep table readable on mobile: summaries are short by prompt design
    show_cols = [
        "rank",
        "company_name",
        "ticker",
        "listed_country",
        "theme_business_summary",
        "theme_revenue_ratio_pct",
        "free_float_mktcap_usd_fmt",
        "theme_mktcap_usd_fmt",
        "non_theme_business_summary",
        "theme_revenue_ratio_method",
        "theme_revenue_ratio_confidence",
        "trr_source_title",
        "trr_source_publisher",
        "trr_source_year",
        "trr_source_locator",
    ]
    existing = [c for c in show_cols if c in view.columns]
    st.dataframe(view[existing], use_container_width=True, height=520)

    # Download (full fields)
    st.download_button(
        "Download CSV (full deliverable)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"themelens_ranked_list_{_sha12(theme_text+region_mode+asof.isoformat())}.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # Evidence detail (per name)
    with st.expander("Evidence details (TRR) — per company", expanded=False):
        for _, r in df.sort_values("rank").iterrows():
            title = f"{int(r['rank'])}. {r['company_name']} ({r['ticker']}) — TRR {r['theme_revenue_ratio']*100:.1f}% [{r.get('theme_revenue_ratio_confidence','')}]"
            with st.expander(title, expanded=False):
                st.write("**Theme business**")
                st.write(r.get("theme_business_summary", "") or "-")
                st.write("**Non-theme business**")
                st.write(r.get("non_theme_business_summary", "") or "-")
                st.divider()
                st.write("**TRR evidence**")
                st.json({
                    "method": r.get("theme_revenue_ratio_method"),
                    "confidence": r.get("theme_revenue_ratio_confidence"),
                    "source_title": r.get("trr_source_title"),
                    "publisher": r.get("trr_source_publisher"),
                    "year": r.get("trr_source_year"),
                    "url": r.get("trr_source_url"),
                    "locator": r.get("trr_source_locator"),
                    "excerpt": r.get("trr_source_excerpt"),
                })

    # Definitions
    with st.expander("Definitions / Methodology", expanded=False):
        st.caption("Ranking is strictly by Theme Market Cap (month-end): Theme Market Cap = Free-float Market Cap (proxy) × TRR.")
        st.dataframe(build_definitions_table(), use_container_width=True, height=420)
