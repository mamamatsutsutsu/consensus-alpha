
# NEXT GEN APP — Theme Portfolio Builder (AI-first, Mobile-first v7 / Gemini)
# -----------------------------------------------------------------------------
# Goals (requested):
# - No sidebar settings (mobile-friendly). Controls live at the top of the page.
# - Stock list creation relies primarily on AI (LLM) via a well-structured prompt.
# - Deliverable is the ranked list (必須項目 + 定義 + ソース情報), sorted by Theme Market Cap.
#
# How to call from app.py:
#   from next_gen_app_tab import render_next_gen_tab
#   render_next_gen_tab(data_dir="data")
#
# Secrets / env (Gemini):
#   GEMINI_API_KEY or GOOGLE_API_KEY (Gemini Developer API).
#   GEMINI_MODEL (optional, e.g. "gemini-2.5-flash").
#   If using Vertex AI instead of API key, set:
#     GOOGLE_GENAI_USE_VERTEXAI=true
#     GOOGLE_CLOUD_PROJECT=...  and GOOGLE_CLOUD_LOCATION=...
#   (The google-genai SDK can pick these up automatically.)
#
# Dependencies:
#   streamlit, pandas, numpy, yfinance, requests, reportlab
# Optional:
#   plotly (recommended)

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple
from pathlib import Path
import hashlib
import io
import json
import math
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# Optional plotting
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False


# =============================================================================
# UI (AlphaLens-ish styling, no sidebar usage)
# =============================================================================
ALPHALENS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');
:root{
  --al-cyan: #00f2fe;
  --al-bg: rgba(0,0,0,0.58);
  --al-card: rgba(255,255,255,0.045);
  --al-border: rgba(255,255,255,0.10);
  --al-text: rgba(255,255,255,0.88);
  --al-muted: rgba(255,255,255,0.70);
}
div[data-testid="stAppViewContainer"] .block-container{ padding-bottom: 2.0rem; max-width: 1400px; }
/* NOTE: do NOT override padding-top here (keeps Streamlit default so top tabs remain clickable on mobile) */
.al-hero{
  margin: 4px auto 12px auto;
  padding: 18px 18px;
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: radial-gradient(1000px 520px at 0% 0%, rgba(0,242,254,0.14), transparent 55%),
              radial-gradient(1000px 520px at 100% 0%, rgba(123,97,255,0.12), transparent 60%),
              linear-gradient(180deg, rgba(0,0,0,0.65), rgba(0,0,0,0.35));
  backdrop-filter: blur(10px);
  box-shadow: 0 12px 36px rgba(0,0,0,0.35);
}
.al-title{
  font-family: Orbitron, ui-sans-serif, system-ui;
  font-size: 24px;
  letter-spacing: 0.12em;
  color: var(--al-cyan);
  margin: 0;
}
.al-sub{
  margin: 8px 0 0 0;
  color: var(--al-muted);
  font-size: 12.5px;
  line-height: 1.45;
}
.al-panel{
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: var(--al-bg);
  backdrop-filter: blur(10px);
  padding: 12px 12px;
}
.al-card{
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.035);
  padding: 10px 10px;
}
.al-kpi-lab{
  color: var(--al-muted);
  font-size: 12px;
  margin-bottom: 6px;
}
.al-kpi-val{
  color: var(--al-text);
  font-size: 19px;
  font-weight: 900;
}
.al-pill{
  display:inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 12px;
  color: rgba(255,255,255,0.84);
  margin-right: 6px;
  margin-top: 6px;
}
[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.07);
}
.stButton>button{ border-radius: 14px; font-weight: 800; letter-spacing: 0.03em; }
h2, h3{ font-family: Orbitron, ui-sans-serif, system-ui; letter-spacing: 0.06em; }

/* Tabs safe area (prevents browser chrome overlap on some mobiles) */
:root{ --safe-top: env(safe-area-inset-top, 0px); }
@media (max-width: 768px){
  div[data-testid="stTabs"]{ scroll-margin-top: calc(4.5rem + var(--safe-top)); }
}
</style>
"""


# =============================================================================
# Types / Models
# =============================================================================
RegionMode = Literal["Global", "Japan", "US", "Europe", "China"]
DataRigorMode = Literal["Strict", "Balanced", "Expand"]
Confidence = Literal["High", "Med", "Low"]
Method = Literal["disclosed", "estimated", "proxy"]

@dataclass(frozen=True)
class ThemeInput:
    theme_text: str
    region_mode: RegionMode
    top_n: int

    data_rigor: DataRigorMode
    min_confidence: Confidence

    # AI config
    use_ai: bool = True
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    candidate_pool: int = 60

    # Market cap config
    min_free_float_mktcap_usd: float = 10e9  # default: $10B (large-cap-ish)
    asof_date_policy: str = "most_recent_month_end"  # fixed policy per requirements

    # Optional extra checks
    verify_source_urls: bool = False
    show_prompt: bool = False


# =============================================================================
# Utilities
# =============================================================================
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

def confidence_factor(conf: str) -> float:
    return {"High": 1.0, "Med": 0.7, "Low": 0.4}.get(str(conf), 0.4)

def safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def fmt_pct(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

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

def snapshot_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


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

@st.cache_data(ttl=60 * 60, show_spinner=False)
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
                df = _yf_download_cached((sym,), start=(asof - timedelta(days=14)).isoformat(), end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False, group_by="column")
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].dropna()
                if close.empty:
                    continue
                rate = float(close.iloc[-1])
                if rate <= 0 or np.isnan(rate):
                    continue

                usd_per_ccy = rate
                if sym.startswith("USD"):
                    usd_per_ccy = 1.0 / rate
                if 0 < usd_per_ccy < 100:
                    return usd_per_ccy, f"best_effort({sym})"
            except Exception:
                continue
        return None, "unavailable"

    def free_float_mktcap_asof(self, ticker: str, asof: date) -> Tuple[Optional[float], str, Optional[float], str]:
        """Returns (mcap_local, currency, mcap_usd, quality_tag).

        Policy:
        - Prefer floatShares × close_asof (best effort).
        - Else sharesOutstanding × close_asof.
        - Else use yfinance 'marketCap' field (current, not asof).
        """
        info = self.info(ticker)
        currency = str(info.get("currency") or "USD").upper()

        px = None
        try:
            df = _yf_download_cached((ticker,), start=(asof - timedelta(days=14)).isoformat(), end=(asof + timedelta(days=1)).isoformat(), auto_adjust=False, group_by="column")
            if df is not None and not df.empty and "Close" in df.columns:
                close = df["Close"].dropna()
                if not close.empty:
                    px = float(close.iloc[-1])
        except Exception:
            px = None

        if px is None or px <= 0:
            return None, currency, None, "missing_price"

        float_shares = info.get("floatShares")
        shares_out = info.get("sharesOutstanding")

        if float_shares and float_shares > 0:
            mcap_local = float(float_shares) * px
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_free_float(floatShares×Close_asof),{q}"

        if shares_out and shares_out > 0:
            mcap_local = float(shares_out) * px
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_total(sharesOutstanding×Close_asof),{q}"

        mcap = info.get("marketCap")
        if mcap and mcap > 0:
            mcap_local = float(mcap)
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_field(marketCap_current),{q}"

        return None, currency, None, "missing_shares"


# =============================================================================
# AI Provider (OpenAI-compatible chat completions via HTTP)
# =============================================================================
def _get_secret(key: str) -> Optional[str]:
    try:
        v = st.secrets.get(key)
        if v:
            return str(v)
    except Exception:
        pass
    import os
    return os.getenv(key)

def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from an LLM response.
    - Accepts raw JSON or JSON wrapped in code fences.
    - Tries best-effort extraction of the outermost {...}.
    """
    import re
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")
    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        text = re.sub(r"\n```$", "", text.strip())
    # First try: direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # Sometimes the model returns a list; wrap as dict for safety.
        return {"_": obj}
    except Exception:
        pass
    # Second try: extract the largest {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
            return {"_": obj}
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from LLM output")

def _get_google_genai_client(api_key: Optional[str]):
    """
    Prefer the Google GenAI SDK (google-genai). Falls back elsewhere if not installed.

    We cache the client in st.session_state to avoid re-creating HTTP sessions on every rerun.
    """
    cache_key = "__ng_google_genai_client__" + ("key" if api_key else "env")
    try:
        if cache_key in st.session_state:
            return st.session_state[cache_key]
    except Exception:
        pass

    from google import genai  # type: ignore
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    try:
        st.session_state[cache_key] = client
    except Exception:
        pass
    return client

def gemini_generate_json(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """
    Gemini JSON generation with best-effort compatibility.

    Preference order:
      1) Google GenAI SDK (pip install google-genai) — recommended by Google.
      2) Legacy google.generativeai (if present).
      3) REST call to generativelanguage.googleapis.com (Gemini Developer API) if API key available.

    Notes:
      - Uses JSON mode (response MIME type: application/json).
      - Expects messages = [{"role":"system","content":...}, {"role":"user","content":...}, ...]
    """
    # Extract system + user content (single-turn)
    system_txt = ""
    user_txt_parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").lower().strip()
        content = str(m.get("content") or "")
        if role == "system" and not system_txt:
            system_txt = content
        elif role == "user":
            user_txt_parts.append(content)
        # ignore assistant/model history for now (single-turn JSON generation)

    user_txt = "\n\n".join([p for p in user_txt_parts if p]).strip()
    if not user_txt:
        raise RuntimeError("Gemini call: missing user prompt text.")

    api_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")

    last_err: Optional[Exception] = None

    # 1) google-genai (preferred)
    try:
        from google.genai import types  # type: ignore

        client = _get_google_genai_client(api_key)

        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
            system_instruction=system_txt or None,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(model=model, contents=user_txt, config=cfg)
        txt = getattr(resp, "text", "")  # google-genai exposes .text
        if callable(txt):
            txt = txt()
        return _parse_json_from_text(str(txt))
    except Exception as e:
        last_err = e

    # 2) legacy google.generativeai (may exist in some stacks)
    try:
        import google.generativeai as genai_old  # type: ignore

        if api_key:
            genai_old.configure(api_key=api_key)

        generation_config = {
            "temperature": float(temperature),
            "response_mime_type": "application/json",
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

    # 3) REST fallback (requires API key)
    if not api_key:
        raise RuntimeError(
            "Gemini is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY (Gemini Developer API), "
            "or install/configure google-genai for Vertex AI. "
            f"Last error: {last_err}"
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_txt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }
    if system_txt:
        payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_txt}]}

    headers = {"Content-Type": "application/json"}

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            # candidates[0].content.parts[0].text
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
        except Exception as e:
            last_err = e
            time.sleep(1.25 * (2 ** attempt))
            continue

    raise RuntimeError(f"Gemini REST call failed after retries: {last_err}")

def _ai_system_prompt() -> str:
    return (
        "You are a senior global equity portfolio manager and sector analyst. "
        "You build investable theme portfolios for institutional clients. "
        "Be conservative: never invent URLs or quotes. "
        "If you cannot provide a verified URL and exact excerpt, set url=null and excerpt='' and use method='estimated' with confidence='Low'. "
        "Return ONLY valid JSON. No markdown."
    )

def _region_definition(region: RegionMode) -> str:
    if region == "Global":
        return "Developed markets (North America, Western Europe incl. UK & Switzerland, Japan, Australia) PLUS China (Mainland + HK). Exclude frontier markets."
    if region == "Japan":
        return "Japan listed equities (TSE etc)."
    if region == "US":
        return "United States listed equities (NYSE/NASDAQ etc)."
    if region == "Europe":
        return "Developed Europe: EU + UK + Switzerland (exclude Russia)."
    if region == "China":
        return "China exposure universe: Mainland A-shares (Shanghai/Shenzhen) + Hong Kong listed China large-caps."
    return "Global"

def build_ai_messages(inp: ThemeInput, asof_mktcap_date: date) -> List[Dict[str, str]]:
    region_def = _region_definition(inp.region_mode)

    schema = {
        "status": "ok | ambiguous | error",
        "theme_definition": "string",
        "reference_etfs": ["string"],
        "notes": "string",
        "companies": [
            {
                "company_name": "string",
                "ticker": "string (yfinance-compatible)",
                "listed_country": "string",
                "primary_exchange": "string",
                "theme_business_summary": "string (Japanese)",
                "non_theme_business_summary": "string (Japanese)",
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
                        "excerpt": "string (<=25 words, exact quote if url present)",
                    }
                ],
                "theme_profit_ratio": "number (0-1) or null",
                "theme_profit_ratio_year": "number or null",
            }
        ],
    }

    user_prompt = f"""
# Task
Build an investable theme portfolio candidate list.

## Theme
- Theme text: "{inp.theme_text}"
- Region filter: {inp.region_mode}
- Region definition: {region_def}
- Target count (Top N): {inp.top_n}
- Candidate pool to return (>= Top N): {inp.candidate_pool}
- Data rigor: {inp.data_rigor}
- Minimum confidence: {inp.min_confidence}

## Universe / size constraints
- Only large, liquid listed companies that are typical constituents of major indices in the region.
- Avoid small caps and obscure listings.

## Ranking concept (IMPORTANT)
- We will rank by Theme Market Cap (descending), defined as:
  Theme Market Cap = Free-float Market Cap × TRR
- You MUST provide TRR for each company (0-1). If no reliable disclosure exists, you may estimate, but label method="estimated" and confidence="Low".

## TRR definition
- TRR (Theme Revenue Ratio) = (revenue from theme-related business) / (total company revenue).

## Profit-based ratio
- Provide theme_profit_ratio if it can be sourced (segment profit / operating profit share). If not available, return null.

## Rigor policy (IMPORTANT)
- Strict: Only include companies where TRR is directly disclosed in primary sources (filings/official IR). Use confidence=High.
- Balanced: Prefer disclosed. If missing, allow estimates (method=estimated, confidence=Low) but keep sources used for estimation.
- Expand: Allow broader proxies/estimates if needed.
- Minimum confidence: Try to keep company-level confidence >= {inp.min_confidence}. If not feasible, mention it in notes.

## Evidence rules (commercial-grade)
- Prefer sources in this priority order:
  1) Regulatory filings / annual report / 20-F / 10-K
  2) Official earnings materials / investor presentation / segment tables
  3) Company website product/segment revenue mix
  4) Reputable third-party industry reports (only if official not available)
  5) Estimation (allowed only if above unavailable; must be clearly labeled)
- Do NOT invent URLs or excerpts.
  - If you cannot provide a verified URL + exact excerpt, set url=null and excerpt="".
  - In that case, method should usually be "estimated" and confidence="Low".

## Output language
- Japanese for summaries and notes.

## Output format (STRICT)
Return ONLY JSON with this exact top-level schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

If the theme is ambiguous/un-investable (too vague, boundary unclear), return:
- status="ambiguous"
- theme_definition: explain why ambiguous
- reference_etfs: suggest 3-8 closest ETFs (tickers)
- companies: []
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

    norm_companies = []
    for c in companies:
        if not isinstance(c, dict):
            continue
        trr = clamp01(c.get("theme_revenue_ratio", 0.0))
        conf = c.get("theme_revenue_ratio_confidence", "Low")
        if conf not in ("High", "Med", "Low"):
            conf = "Low"
        method = c.get("theme_revenue_ratio_method", "estimated")
        if method not in ("disclosed", "estimated", "proxy"):
            method = "estimated"

        sources = c.get("theme_revenue_sources", [])
        if not isinstance(sources, list):
            sources = []
        sources2 = []
        for s in sources[:2]:
            if not isinstance(s, dict):
                continue
            sources2.append({
                "source_title": safe_str(s.get("source_title")),
                "publisher": safe_str(s.get("publisher")),
                "year": s.get("year", None),
                "url": s.get("url", None),
                "locator": safe_str(s.get("locator")),
                "excerpt": safe_str(s.get("excerpt")),
            })

        norm_companies.append({
            "company_name": safe_str(c.get("company_name")),
            "ticker": safe_str(c.get("ticker")).upper().strip(),
            "listed_country": safe_str(c.get("listed_country")),
            "primary_exchange": safe_str(c.get("primary_exchange")),
            "theme_business_summary": safe_str(c.get("theme_business_summary")),
            "non_theme_business_summary": safe_str(c.get("non_theme_business_summary")),
            "theme_revenue_ratio": float(trr),
            "theme_revenue_ratio_year": c.get("theme_revenue_ratio_year", None),
            "theme_revenue_ratio_method": method,
            "theme_revenue_ratio_confidence": conf,
            "theme_revenue_sources": sources2,
            "theme_profit_ratio": c.get("theme_profit_ratio", None),
            "theme_profit_ratio_year": c.get("theme_profit_ratio_year", None),
        })

    return {
        "status": status,
        "theme_definition": safe_str(obj.get("theme_definition", "")),
        "reference_etfs": obj.get("reference_etfs", []) if isinstance(obj.get("reference_etfs", []), list) else [],
        "notes": safe_str(obj.get("notes", "")),
        "companies": norm_companies,
    }

def _verify_urls(companies: List[Dict[str, Any]], timeout_s: int = 10) -> List[str]:
    warnings: List[str] = []
    checked = 0
    for c in companies:
        srcs = c.get("theme_revenue_sources", []) or []
        for s in srcs:
            url = s.get("url")
            if not url or not isinstance(url, str):
                continue
            if checked >= 25:
                warnings.append("URL verification capped at 25 sources for speed.")
                return warnings
            checked += 1
            try:
                r = requests.head(url, timeout=timeout_s, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code >= 400:
                    warnings.append(f"Source URL not reachable ({r.status_code}): {url}")
            except Exception:
                warnings.append(f"Source URL not reachable (exception): {url}")
    return warnings

def _repair_bad_tickers_with_ai(
    *,
    bad: List[Tuple[str, str]],
    model: str,
    temperature: float,
) -> Dict[str, str]:
    if not bad:
        return {}

    schema = {"repairs": [{"company_name": "string", "old_ticker": "string", "new_ticker": "string"}]}
    prompt = f"""
Fix yfinance tickers. Return ONLY JSON.
Input pairs:
{json.dumps([{"company_name": n, "old_ticker": t} for n,t in bad], ensure_ascii=False, indent=2)}

Rules:
- Provide correct yfinance-compatible ticker for the primary listing.
- If already correct, return same.

Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    messages = [{"role":"system","content":_ai_system_prompt()},
                {"role":"user","content":prompt.strip()}]
    obj = gemini_generate_json(messages=messages, model=model, temperature=temperature, timeout_s=90)
    out: Dict[str, str] = {}
    try:
        repairs = obj.get("repairs", [])
        if isinstance(repairs, list):
            for r in repairs:
                if not isinstance(r, dict):
                    continue
                old = safe_str(r.get("old_ticker")).upper().strip()
                new = safe_str(r.get("new_ticker")).upper().strip()
                if old and new:
                    out[old] = new
    except Exception:
        pass
    return out


# =============================================================================
# Core: Build deliverable ranked list
# =============================================================================
def build_ranked_list_ai(inp: ThemeInput, today: Optional[date] = None) -> Dict[str, Any]:
    today = today or date.today()
    asof = most_recent_month_end(today)

    if not inp.use_ai:
        return {
            "status": "error",
            "asof_mktcap_date": asof.isoformat(),
            "ai": {"status": "error", "notes": "AI is disabled. This version is AI-first; enable AI to generate the list.", "companies": []},
            "rows": [],
            "warnings": ["AI is disabled. Turn on 'Use AI for stock list' in Advanced settings."],
        }

    messages = build_ai_messages(inp, asof)
    ai_raw = gemini_generate_json(messages=messages, model=inp.model, temperature=inp.temperature, timeout_s=180)
    ai = _normalize_ai_output(ai_raw)

    if inp.show_prompt:
        ai["__prompt_messages__"] = messages

    if ai.get("status") != "ok":
        return {
            "status": ai.get("status", "error"),
            "asof_mktcap_date": asof.isoformat(),
            "ai": ai,
            "rows": [],
            "warnings": [],
        }

    companies = ai.get("companies", []) or []
    if not companies:
        return {
            "status": "error",
            "asof_mktcap_date": asof.isoformat(),
            "ai": ai,
            "rows": [],
            "warnings": ["AI returned zero companies."],
        }

    market = MarketProviderYF()
    rows: List[Dict[str, Any]] = []
    bad_tickers: List[Tuple[str, str]] = []

    for c in companies:
        ticker = safe_str(c.get("ticker")).upper().strip()
        if not ticker:
            continue

        mcap_local, ccy, mcap_usd, mcap_q = market.free_float_mktcap_asof(ticker, asof)
        if mcap_usd is None:
            bad_tickers.append((safe_str(c.get("company_name")), ticker))

        trr = clamp01(c.get("theme_revenue_ratio", 0.0))
        theme_mcap = (mcap_usd * trr) if (mcap_usd is not None) else None

        srcs = c.get("theme_revenue_sources", []) or []
        src0 = srcs[0] if srcs else {}

        rows.append({
            "company_name": safe_str(c.get("company_name")),
            "ticker": ticker,
            "listed_country": safe_str(c.get("listed_country")),
            "primary_exchange": safe_str(c.get("primary_exchange")),
            "theme_business_summary": safe_str(c.get("theme_business_summary")),
            "non_theme_business_summary": safe_str(c.get("non_theme_business_summary")),

            "theme_revenue_ratio": trr,
            "theme_revenue_ratio_year": c.get("theme_revenue_ratio_year", None),
            "theme_revenue_ratio_method": safe_str(c.get("theme_revenue_ratio_method")),
            "theme_revenue_ratio_confidence": safe_str(c.get("theme_revenue_ratio_confidence")),

            "trr_source_title": safe_str(src0.get("source_title")),
            "trr_source_publisher": safe_str(src0.get("publisher")),
            "trr_source_year": src0.get("year", None),
            "trr_source_url": src0.get("url", None),
            "trr_source_locator": safe_str(src0.get("locator")),
            "trr_source_excerpt": safe_str(src0.get("excerpt")),

            "mktcap_asof_date": asof.isoformat(),
            "free_float_mktcap_usd": mcap_usd,
            "free_float_mktcap_ccy": ccy,
            "free_float_mktcap_local": mcap_local,
            "mktcap_quality": mcap_q,
            "theme_mktcap_usd": theme_mcap,

            "theme_profit_ratio": c.get("theme_profit_ratio", None),
            "theme_profit_ratio_year": c.get("theme_profit_ratio_year", None),
        })

    warnings: List[str] = []
    if bad_tickers and inp.use_ai and len(bad_tickers) <= 20:
        repairs = _repair_bad_tickers_with_ai(bad=bad_tickers[:20], model=inp.model, temperature=min(inp.temperature, 0.3))
        if repairs:
            market2 = MarketProviderYF()
            for r in rows:
                old = safe_str(r.get("ticker")).upper().strip()
                if old in repairs:
                    new = repairs[old]
                    r["ticker"] = new
                    mcap_local, ccy, mcap_usd, mcap_q = market2.free_float_mktcap_asof(new, asof)
                    r["free_float_mktcap_usd"] = mcap_usd
                    r["free_float_mktcap_ccy"] = ccy
                    r["free_float_mktcap_local"] = mcap_local
                    r["mktcap_quality"] = mcap_q
                    trr = float(r.get("theme_revenue_ratio", 0.0) or 0.0)
                    r["theme_mktcap_usd"] = (mcap_usd * trr) if mcap_usd is not None else None
            warnings.append(f"Repaired {len(repairs)} tickers via AI (yfinance compatibility).")

    df = pd.DataFrame(rows)

    if not df.empty:
        df["free_float_mktcap_usd"] = pd.to_numeric(df["free_float_mktcap_usd"], errors="coerce")
        df["theme_mktcap_usd"] = pd.to_numeric(df["theme_mktcap_usd"], errors="coerce")
        df = df.dropna(subset=["free_float_mktcap_usd", "theme_mktcap_usd"])
        df = df[df["free_float_mktcap_usd"] >= float(inp.min_free_float_mktcap_usd)]

    # Confidence / rigor filters (so UI controls actually matter)
    if not df.empty:
        def _conf_rank(x: Any) -> int:
            return {"Low": 1, "Med": 2, "High": 3}.get(str(x), 1)
        min_rank = _conf_rank(inp.min_confidence)
        df["__conf_rank__"] = df["theme_revenue_ratio_confidence"].map(_conf_rank)
        df = df[df["__conf_rank__"] >= min_rank]

        if inp.data_rigor == "Strict":
            df = df[df["theme_revenue_ratio_method"] == "disclosed"]
            df = df[df["theme_revenue_ratio_confidence"] == "High"]
            # require at least a named source (URL may be null if not confident)
            df = df[df["trr_source_title"].astype(str).str.len() > 0]

        df = df.drop(columns=["__conf_rank__"], errors="ignore")

    if df.empty:
        return {
            "status": "error",
            "asof_mktcap_date": asof.isoformat(),
            "ai": ai,
            "rows": [],
            "warnings": warnings + ["All candidates filtered out (market cap missing or below threshold). Try lowering min mktcap or using Expand mode."],
        }

    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    n_after = len(df)
    if n_after < int(inp.top_n):
        warnings.append(f"Only {n_after} names meet the confidence/rigor filters (requested {int(inp.top_n)}).")
    df = df.head(int(inp.top_n)).copy()

    if inp.verify_source_urls:
        warnings += _verify_urls(companies=ai.get("companies", []), timeout_s=10)

    return {
        "status": "ok",
        "asof_mktcap_date": asof.isoformat(),
        "ai": ai,
        "rows": df.to_dict(orient="records"),
        "warnings": warnings,
    }


# =============================================================================
# Definitions (requested)
# =============================================================================
def build_definitions_table() -> pd.DataFrame:
    items = [
        {"項目": "企業名 (company_name)", "定義": "銘柄の一次上場の発行体名（AIが候補を提案）", "単位/補足": "文字列"},
        {"項目": "上場国名 (listed_country)", "定義": "一次上場国（地域フィルタ適用に使用）", "単位/補足": "文字列"},
        {"項目": "テーマ関連事業の概要 (theme_business_summary)", "定義": "テーマに該当する事業・製品・サービスの要約（日本語）", "単位/補足": "文字列"},
        {"項目": "テーマ関連売上比率 TRR (theme_revenue_ratio)", "定義": "TRR = テーマ関連事業売上 ÷ 総売上（0〜1）", "単位/補足": "比率（0-1）"},
        {"項目": "free-float時価総額（近似）(free_float_mktcap_usd)", "定義": "原則 floatShares×Close(as-of)（無い場合 sharesOutstanding×Close、さらに無ければmarketCap）", "単位/補足": "USD（換算）。qualityタグに算出根拠"},
        {"項目": "テーマ関連時価総額 (theme_mktcap_usd)", "定義": "Theme Market Cap = free_float_mktcap_usd × TRR", "単位/補足": "USD"},
        {"項目": "非テーマ事業概要 (non_theme_business_summary)", "定義": "テーマ以外の主要事業の要約（日本語）", "単位/補足": "文字列"},
        {"項目": "TRRソース (trr_source_*)", "定義": "title/publisher/year/url/locator/excerpt を保持。確証が無い場合 url=null で推計扱い", "単位/補足": "監査用メタデータ"},
        {"項目": "Data rigor", "定義": "Strict=一次情報でTRRが直接開示されたもの中心 / Balanced=推計も許容 / Expand=proxy含め広く探索", "単位/補足": "UIで選択"},
        {"項目": "Confidence", "定義": "High=一次情報で明確 / Med=限定的開示＋推定 / Low=推計依存（method=estimated）", "単位/補足": "UIで選択"},
        {"項目": "時価総額基準日 (mktcap_asof_date)", "定義": "原則：直近月末（取得できない場合は直近に近い営業日価格で代替）", "単位/補足": "YYYY-MM-DD"},
    ]
    return pd.DataFrame(items)


# =============================================================================
# PDF export (log)
# =============================================================================
def build_pdf_report(payload: Dict[str, Any]) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
        base_font = "HeiseiKakuGo-W5"
    except Exception:
        base_font = "Helvetica"

    styles = getSampleStyleSheet()
    styleN = ParagraphStyle("N", parent=styles["Normal"], fontName=base_font, fontSize=9, leading=12)
    styleH = ParagraphStyle("H", parent=styles["Heading1"], fontName=base_font, fontSize=14, leading=18, spaceAfter=8, textColor=colors.HexColor("#00AEB7"))
    styleS = ParagraphStyle("S", parent=styles["Normal"], fontName=base_font, fontSize=8, leading=11, textColor=colors.grey)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14*mm, rightMargin=14*mm, topMargin=14*mm, bottomMargin=14*mm)

    meta = payload.get("meta", {}) or {}
    rows = payload.get("rows", []) or []
    warnings = payload.get("warnings", []) or []

    elements = []
    elements.append(Paragraph("ALPHALENS — NEXT GEN APP Snapshot", styleH))
    elements.append(Paragraph(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styleS))
    elements.append(Paragraph(f"Theme: {meta.get('theme_text','')}", styleN))
    elements.append(Paragraph(f"Region: {meta.get('region_mode','')}", styleN))
    elements.append(Paragraph(f"As-of (MktCap): {meta.get('asof_mktcap_date','')}", styleN))
    elements.append(Spacer(1, 6))

    if warnings:
        elements.append(Paragraph("Warnings", styleN))
        for w in warnings[:10]:
            elements.append(Paragraph(f"- {w}", styleS))
        elements.append(Spacer(1, 6))

    header = ["Rank","Ticker","Company","TRR","FF MktCap(USD)","Theme MktCap(USD)","TRR Source"]
    data = [header]
    for r in rows:
        data.append([
            str(r.get("rank","")),
            safe_str(r.get("ticker","")),
            safe_str(r.get("company_name",""))[:22],
            f"{float(r.get('theme_revenue_ratio',0))*100:.1f}%",
            fmt_money(r.get("free_float_mktcap_usd")),
            fmt_money(r.get("theme_mktcap_usd")),
            (safe_str(r.get("trr_source_title",""))[:28] or "-"),
        ])

    table = Table(data, colWidths=[10*mm, 18*mm, 42*mm, 14*mm, 30*mm, 30*mm, 30*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), base_font),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    elements.append(table)
    doc.build(elements)
    return buf.getvalue()


# =============================================================================
# Streamlit entrypoint (no sidebar)
# =============================================================================
def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(ALPHALENS_CSS, unsafe_allow_html=True)

    st.markdown(
        """
<div class="al-hero">
  <p class="al-title">NEXT GEN APP</p>
  <p class="al-sub">
    AI-first Theme Portfolio Builder. Controls are at the top (no sidebar). Deliverable is the ranked list by Theme Market Cap.
    <br/>TRR (theme revenue ratio) is evidence-backed when possible; otherwise explicitly labeled as estimates.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="al-panel">', unsafe_allow_html=True)
    with st.form("ng_controls_v5", clear_on_submit=False):
        c1, c2 = st.columns([2.2, 1.0])
        with c1:
            theme_text = st.text_input("Theme（自由入力）", value=st.session_state.get("ng_theme_text", "半導体"))
        with c2:
            region_mode = st.selectbox("Region", ["Global","Japan","US","Europe","China"], index=["Global","Japan","US","Europe","China"].index(st.session_state.get("ng_region_mode","Global")))

        c3, c4, c5 = st.columns([1.1, 1.1, 1.0])
        with c3:
            top_n = st.slider("銘柄数 (1-30)", 1, 30, int(st.session_state.get("ng_top_n", 10)), 1)
        with c4:
            data_rigor = st.selectbox("Data rigor", ["Balanced","Strict","Expand"], index=["Balanced","Strict","Expand"].index(st.session_state.get("ng_data_rigor","Balanced")))
        with c5:
            run = st.form_submit_button("RUN", type="primary", use_container_width=True)

        if hasattr(st, "popover"):
            adv = st.popover("Advanced / AI settings")
        else:
            adv = st.expander("Advanced / AI settings", expanded=False)

        with adv:
            use_ai = st.toggle("Use AI for stock list (recommended)", value=bool(st.session_state.get("ng_use_ai", True)))
            model = st.text_input("Model", value=st.session_state.get("ng_model", _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"))
            temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.get("ng_temp", 0.2)), 0.05)
            candidate_pool = st.slider("AI candidate pool size", 20, 120, int(st.session_state.get("ng_pool", 60)), 5)
            min_conf = st.selectbox("Min confidence", ["Low","Med","High"], index=["Low","Med","High"].index(st.session_state.get("ng_min_conf","Low")))
            min_mcap_b = st.slider("Min free-float mktcap (USD, billions)", 1, 200, int(st.session_state.get("ng_min_mcap_b", 10)), 1)
            verify_urls = st.toggle("Verify source URLs (fast HEAD check)", value=bool(st.session_state.get("ng_verify_urls", False)))
            show_prompt = st.toggle("Show prompt (debug)", value=bool(st.session_state.get("ng_show_prompt", False)))

    st.markdown('</div>', unsafe_allow_html=True)

    st.session_state["ng_theme_text"] = theme_text
    st.session_state["ng_region_mode"] = region_mode
    st.session_state["ng_top_n"] = int(top_n)
    st.session_state["ng_data_rigor"] = data_rigor
    st.session_state["ng_use_ai"] = bool(locals().get("use_ai", True))
    st.session_state["ng_model"] = locals().get("model", "gemini-2.5-flash")
    st.session_state["ng_temp"] = float(locals().get("temperature", 0.2))
    st.session_state["ng_pool"] = int(locals().get("candidate_pool", 60))
    st.session_state["ng_min_conf"] = locals().get("min_conf", "Low")
    st.session_state["ng_min_mcap_b"] = int(locals().get("min_mcap_b", 10))
    st.session_state["ng_verify_urls"] = bool(locals().get("verify_urls", False))
    st.session_state["ng_show_prompt"] = bool(locals().get("show_prompt", False))

    with st.expander("表示項目の定義 / Methodology（必読）", expanded=False):
        st.markdown(
            """
**このアプリの“納品物”は Ranked List（テーマ関連時価総額の大きい順）です。**

- **TRR (Theme Revenue Ratio)**: テーマ関連事業売上 ÷ 総売上（0〜1）
- **Theme Market Cap**: **Free-float時価総額（近似）× TRR**
- **Free-float時価総額（近似）**: 原則 `floatShares × Close(as-of)`（取得不可の場合は `sharesOutstanding × Close(as-of)`、さらに不可なら `marketCap` を使用）
- **情報ソース**: 可能な限り一次情報（年次報告書/10-K/20-F/決算資料/公式IR）を優先。無い場合は推計し、**method=estimated** と **confidence=Low** を明示。
- **Data rigor**: Strict=一次情報中心 / Balanced=推計も許容 / Expand=proxy含め探索
- **Confidence**: High=一次情報で明確 / Med=限定的開示＋推定 / Low=推計依存
            """
        )
        st.dataframe(build_definitions_table(), use_container_width=True)

    if not run:
        st.caption("RUNで実行。フォーム方式で、入力中に重い処理が走らない（スマホ向け）。")
        return

    inp = ThemeInput(
        theme_text=theme_text,
        region_mode=region_mode,  # type: ignore
        top_n=int(top_n),
        data_rigor=data_rigor,    # type: ignore
        min_confidence=locals().get("min_conf", "Low"),  # type: ignore
        use_ai=bool(locals().get("use_ai", True)),
        model=str(locals().get("model", "gemini-2.5-flash") or "gemini-2.5-flash"),
        temperature=float(locals().get("temperature", 0.2)),
        candidate_pool=int(locals().get("candidate_pool", 60)),
        min_free_float_mktcap_usd=float(locals().get("min_mcap_b", 10)) * 1e9,
        verify_source_urls=bool(locals().get("verify_urls", False)),
        show_prompt=bool(locals().get("show_prompt", False)),
    )

    with st.spinner("AIがテーマ銘柄リストを生成し、直近月末の時価総額で再ランキング中…"):
        try:
            res = build_ranked_list_ai(inp)
        except Exception as e:
            st.error("NEXT GEN APP failed. Check GEMINI_API_KEY or GOOGLE_API_KEY / model / network.")
            st.exception(e)
            return

    sid = snapshot_id({"input": asdict(inp), "asof": res.get("asof_mktcap_date","")})
    meta = {
        "snapshot_id": sid,
        "theme_text": inp.theme_text,
        "region_mode": inp.region_mode,
        "top_n": inp.top_n,
        "asof_mktcap_date": res.get("asof_mktcap_date"),
        "ai_model": inp.model if inp.use_ai else "N/A",
        "candidate_pool": inp.candidate_pool,
        "min_free_float_mktcap_usd": inp.min_free_float_mktcap_usd,
    }

    if res.get("status") != "ok":
        ai = res.get("ai", {}) or {}
        if res.get("warnings"):
            st.warning("\n".join([f"- {w}" for w in res["warnings"]]))
        if ai.get("status") == "ambiguous":
            st.error("テーマが曖昧で、テーマ運用にはそぐわないため、銘柄選定は行いません。")
            st.write("**AIによる解釈 / なぜ曖昧か**")
            st.write(ai.get("theme_definition", "") or ai.get("notes",""))
            etfs = ai.get("reference_etfs", []) or []
            if etfs:
                st.write("**参考ETF（テーマ定義のたたき台）**")
                st.write(", ".join([str(x) for x in etfs[:10]]))
            if inp.show_prompt and ai.get("__prompt_messages__"):
                with st.expander("Prompt (debug)", expanded=False):
                    st.json(ai["__prompt_messages__"])
            st.caption(f"Snapshot ID: {sid}")
            return

        st.error("候補銘柄が生成できませんでした。条件を調整して再実行してください。")
        st.write(ai.get("notes",""))
        if inp.show_prompt and ai.get("__prompt_messages__"):
            with st.expander("Prompt (debug)", expanded=False):
                st.json(ai["__prompt_messages__"])
        st.caption(f"Snapshot ID: {sid}")
        return

    rows = res.get("rows", []) or []
    df = pd.DataFrame(rows)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="al-card"><div class="al-kpi-lab">As-of (MktCap)</div><div class="al-kpi-val">{meta["asof_mktcap_date"]}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="al-card"><div class="al-kpi-lab">Region</div><div class="al-kpi-val">{meta["region_mode"]}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="al-card"><div class="al-kpi-lab">AI model</div><div class="al-kpi-val">{meta["ai_model"]}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="al-card"><div class="al-kpi-lab">Snapshot</div><div class="al-kpi-val">{sid}</div></div>', unsafe_allow_html=True)

    pills = [
        f"Theme: {inp.theme_text}",
        f"Top N: {inp.top_n}",
        f"Min FF MktCap: ${meta['min_free_float_mktcap_usd']/1e9:.0f}B",
        f"Rigor: {inp.data_rigor}",
        f"MinConf: {inp.min_confidence}",
        f"Pool: {inp.candidate_pool}",
    ]
    st.markdown("".join([f'<span class="al-pill">{p}</span>' for p in pills]), unsafe_allow_html=True)

    if res.get("warnings"):
        st.warning("\n".join([f"- {w}" for w in res["warnings"]]))

    tab_list, tab_detail, tab_export = st.tabs(["✅ Deliverable: Ranked List", "🔎 Detail / Evidence", "⬇️ Export"])

    with tab_list:
        st.subheader("Ranked List（テーマ関連時価総額の大きい順）")
        st.caption("必須項目をすべて含めています。")

        deliver_cols = [
            "rank",
            "company_name",
            "listed_country",
            "theme_business_summary",
            "theme_revenue_ratio",
            "free_float_mktcap_usd",
            "theme_mktcap_usd",
            "non_theme_business_summary",
            "trr_source_title",
            "trr_source_publisher",
            "trr_source_year",
            "trr_source_url",
            "trr_source_locator",
            "theme_revenue_ratio_method",
            "theme_revenue_ratio_confidence",
        ]
        exist = [c for c in deliver_cols if c in df.columns]
        view = df[exist].copy()

        if "theme_revenue_ratio" in view.columns:
            view["theme_revenue_ratio"] = view["theme_revenue_ratio"].map(lambda x: fmt_pct(x))
        if "free_float_mktcap_usd" in view.columns:
            view["free_float_mktcap_usd"] = view["free_float_mktcap_usd"].map(fmt_money)
        if "theme_mktcap_usd" in view.columns:
            view["theme_mktcap_usd"] = view["theme_mktcap_usd"].map(fmt_money)

        st.dataframe(view, use_container_width=True, height=560)
        st.caption("※ URL/引用は確証がある場合のみ提示。無い場合は推計扱いとして明示（method=estimated）。")

    with tab_detail:
        st.subheader("銘柄別 詳細（概要/非テーマ事業/ソース）")
        if df.empty:
            st.info("No rows.")
        else:
            pick = st.selectbox("Pick a ticker", df["ticker"].tolist())
            r = df[df["ticker"] == pick].iloc[0].to_dict()

            st.markdown("### Core")
            st.write({
                "rank": int(r.get("rank", 0)),
                "company_name": r.get("company_name"),
                "ticker": r.get("ticker"),
                "listed_country": r.get("listed_country"),
                "primary_exchange": r.get("primary_exchange"),
                "asof_mktcap_date": r.get("mktcap_asof_date"),
                "free_float_mktcap_usd": r.get("free_float_mktcap_usd"),
                "theme_mktcap_usd": r.get("theme_mktcap_usd"),
                "mktcap_quality": r.get("mktcap_quality"),
            })

            st.markdown("### Theme business summary")
            st.write(r.get("theme_business_summary") or "-")

            st.markdown("### Non-theme business summary")
            st.write(r.get("non_theme_business_summary") or "-")

            st.markdown("### TRR (Theme Revenue Ratio)")
            st.write({
                "TRR": r.get("theme_revenue_ratio"),
                "year": r.get("theme_revenue_ratio_year"),
                "method": r.get("theme_revenue_ratio_method"),
                "confidence": r.get("theme_revenue_ratio_confidence"),
            })

            st.markdown("### TRR source (primary)")
            st.write({
                "title": r.get("trr_source_title"),
                "publisher": r.get("trr_source_publisher"),
                "year": r.get("trr_source_year"),
                "url": r.get("trr_source_url"),
                "locator": r.get("trr_source_locator"),
                "excerpt": r.get("trr_source_excerpt"),
            })

            if inp.show_prompt and res.get("ai", {}).get("__prompt_messages__"):
                with st.expander("Prompt (debug)", expanded=False):
                    st.json(res["ai"]["__prompt_messages__"])

    with tab_export:
        st.subheader("Export")
        st.code(f"Snapshot ID: {sid}")

        df_export = df.copy()
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download Deliverable CSV", data=csv_bytes, file_name=f"theme_ranked_list_{sid}.csv", mime="text/csv")

        payload = {"meta": meta, "rows": rows, "ai": res.get("ai", {}), "warnings": res.get("warnings", [])}
        st.download_button("Download Full Snapshot (JSON)", data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"theme_snapshot_{sid}.json", mime="application/json")

        st.divider()
        st.subheader("PDF report (log)")
        if st.button("Generate PDF", type="primary"):
            try:
                pdf = build_pdf_report({"meta": meta, "rows": rows, "warnings": res.get("warnings", [])})
                st.session_state["ng_pdf_bytes_v5"] = pdf
                st.success("PDF ready.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        pdfb = st.session_state.get("ng_pdf_bytes_v5")
        if pdfb:
            st.download_button("Download PDF", data=pdfb, file_name=f"theme_report_{sid}.pdf", mime="application/pdf")

    st.caption(f"Snapshot ID: {sid}")