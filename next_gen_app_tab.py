# THEMELENS — AI-first Theme Portfolio Builder (Gemini) v14
# -----------------------------------------------------------------------------
# Fixes:
# - ALWAYS SHOW something when AI returns (even if JSON is broken).
#   * We keep & display the raw model output (in an expander)
#   * We try JSON parse; if it fails, we extract tickers from raw text and still show a list.
# - Make draft parsing resilient:
#   * approx_market_cap_usd may come as "2.1T", "$450B", "1,200,000,000,000" -> robust parser
#   * TRR may come as "60%" -> robust parser
# - UI: If draft_rows exists but is empty, still display raw output + a placeholder.
#
# Workflow (stepwise & bounded):
#  1) Draft Universe (N+20): AI returns rows with TRR + approx mktcap (USD) -> immediate list
#  2) Validate & Rank (optional, bounded): yfinance month-end proxy mktcap replaces AI estimates if available
#  3) Refine (optional): add TRR evidence sources
#
# Deliverable: Ranked list by Theme Market Cap = Market Cap × TRR

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
# Styling — AlphaLens-like + high-visibility controls
# =============================================================================
THEMELENS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap');

:root{
  --al-cyan:#00f2fe;
  --al-muted:rgba(255,255,255,0.72);
  --al-border:rgba(255,255,255,0.10);
  --al-bg:rgba(0,0,0,0.58);
}

div[data-testid="stAppViewContainer"] .block-container{
  padding-bottom: 2.0rem;
  max-width: 1400px;
}

.tl-title{
  font-family:Orbitron, ui-sans-serif, system-ui;
  letter-spacing:0.14em;
  color:var(--al-cyan);
  margin: 0 0 12px 0;
  font-size: 28px;
}

.tl-panel{
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: var(--al-bg);
  backdrop-filter: blur(10px);
  padding: 14px 14px;
}

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

label{ font-weight: 900 !important; letter-spacing: 0.02em; }

.stButton>button{
  border-radius: 14px;
  font-weight: 900;
  letter-spacing: 0.04em;
  padding: 0.74rem 1.0rem;
  font-size: 15px;
}
.stButton>button[kind="primary"]{
  box-shadow: 0 0 18px rgba(0,242,254,0.18);
  border: 1px solid rgba(0,242,254,0.40);
}

[data-testid="stDataFrame"]{
  border-radius:14px;
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.07);
}

.tl-chip{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.05);
  color: rgba(255,255,255,0.80);
  font-size: 12px;
  margin-right: 6px;
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
        if abs(v) >= 1e12: return f"{v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"{v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"{v/1e6:.2f}M"
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
    if region in ("US", "Global"):
        return 10e9
    if region == "Japan":
        return 5e9
    if region in ("Europe", "China"):
        return 8e9
    return 8e9


def draft_pool_size(top_n: int) -> int:
    # User-requested: N + 20 (capped)
    n = max(1, min(int(top_n), 30))
    return int(min(80, n + 20))  # allow a bit more buffer; still bounded


# =============================================================================
# Robust parsers (prevent "list not displayed" due to ValueError)
# =============================================================================
_NUM_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

def parse_usd_number(v: Any) -> float:
    """
    Accepts: 1200000000, "1,200,000,000", "$450B", "2.1T", "USD 15.3B"
    Returns float USD value, or 0.0 if unknown.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        try:
            return float(v)
        except Exception:
            return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    # remove currency tokens
    s = s.replace("USD", "").replace("US$", "").replace("$", "").replace(" ", "")
    # commas
    s = s.replace(",", "")
    # suffix
    m = re.match(r"^([0-9]*\.?[0-9]+)([KMBT])?$", s, flags=re.IGNORECASE)
    if m:
        num = float(m.group(1))
        suf = (m.group(2) or "").upper()
        return num * _NUM_SUFFIX.get(suf, 1.0)
    # try extract first numeric with suffix
    m2 = re.search(r"([0-9]*\.?[0-9]+)\s*([KMBT])", s, flags=re.IGNORECASE)
    if m2:
        num = float(m2.group(1))
        suf = m2.group(2).upper()
        return num * _NUM_SUFFIX.get(suf, 1.0)
    # last resort: extract float
    m3 = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            return 0.0
    return 0.0


def parse_trr(v: Any) -> float:
    """
    Accepts: 0.6, "0.6", "60%", "60 %"
    Returns 0-1.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
        return clamp01(v)
    s = str(v).strip()
    if not s:
        return 0.0
    s = s.replace(" ", "")
    if s.endswith("%"):
        try:
            return clamp01(float(s[:-1]) / 100.0)
        except Exception:
            return 0.0
    # extract number
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not m:
        return 0.0
    try:
        x = float(m.group(1))
        # if "60" without %, treat as percent if > 1.5
        if x > 1.5:
            x = x / 100.0
        return clamp01(x)
    except Exception:
        return 0.0


def extract_tickers_from_text(raw: str, max_n: int = 80) -> List[str]:
    """
    Best-effort ticker extraction from messy model output.
    Priority: dot-tickers (0700.HK) and (NVDA) style.
    """
    raw = raw or ""
    # capture with positions to preserve order
    candidates: List[Tuple[int, str]] = []

    # dot tickers
    for m in re.finditer(r"\b[0-9A-Z]{1,10}\.[A-Z]{1,6}\b", raw):
        candidates.append((m.start(), m.group(0).upper()))

    # (NVDA) style
    for m in re.finditer(r"\(([A-Z]{1,5})\)", raw):
        candidates.append((m.start(), m.group(1).upper()))

    # bullet lines: "- NVDA"
    for m in re.finditer(r"(?m)^\s*(?:-|\*|\d+\.)\s*([A-Z]{2,5})\b", raw):
        candidates.append((m.start(), m.group(1).upper()))

    # de-dup in order, filter obvious stopwords
    stop = {"THE","AND","FOR","WITH","FROM","THIS","THAT","WILL","ARE","INC","LTD","PLC","CO","ETF","USD","TRR","TOP","RANK"}
    out: List[str] = []
    seen = set()
    for _, t in sorted(candidates, key=lambda x: x[0]):
        if t in stop:
            continue
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= max_n:
            break
    return out


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
# Gemini — get RAW text (then we parse; if parsing fails we still display raw)
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
    try:
        return json.loads(txt)
    except Exception:
        pass
    lo, hi = txt.find("{"), txt.rfind("}")
    if lo != -1 and hi != -1 and hi > lo:
        frag = txt[lo:hi+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    lo, hi = txt.find("["), txt.rfind("]")
    if lo != -1 and hi != -1 and hi > lo:
        frag = txt[lo:hi+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    raise ValueError("No JSON object/array found.")


def _compose_prompt(messages: List[Dict[str, str]]) -> Tuple[str, str]:
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
    return system_txt, user_txt


def gemini_generate_raw_text(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> str:
    system_txt, user_txt = _compose_prompt(messages)
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
        return str(txt or "")

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
        return str(txt or "")
    except Exception as e:
        raise RuntimeError(f"Gemini call failed. Configure GEMINI_API_KEY/GOOGLE_API_KEY. Error: {e}")


def _ai_system_prompt() -> str:
    return (
        "You are a senior global equity portfolio manager and sector analyst. "
        "Return ONLY valid JSON. No markdown. "
        "Never invent URLs or quotes. "
        "If uncertain, set method='estimated' confidence='Low'."
    )


# =============================================================================
# Prompts
# =============================================================================
def build_messages_draft_universe(theme: str, region: RegionMode, asof: date, pool: int) -> List[Dict[str, str]]:
    region_def = _region_definition(region)

    schema = {
        "status": "ok|ambiguous|error",
        "theme_definition": "string",
        "notes": "string",
        "reference_etfs": ["string"],
        "rows": [
            {
                "rank_hint": 1,
                "company_name": "string",
                "ticker": "string",
                "listed_country": "string",
                "primary_exchange": "string",
                "subtheme_bucket": "string",
                "approx_market_cap_usd": 0.0,
                "theme_revenue_ratio": 0.0,
                "method": "disclosed|proxy|estimated",
                "confidence": "High|Med|Low",
            }
        ],
    }

    user_prompt = f"""
Task: Draft a candidate universe for a theme portfolio.

Inputs:
- Theme: "{theme}"
- Region: {region}
- Region definition: {region_def}
- Candidate count to return: {pool}
- As-of date for market cap approximation: {asof.isoformat()} (month-end)

Requirements (VERY IMPORTANT):
- Focus on LARGE-CAP, highly liquid public equities typical of major indices.
- The list must be complete for likely top "Theme Market Cap" leaders.
- Provide yfinance-compatible tickers WITH suffixes when needed:
  examples: 0700.HK, 005930.KS, 2330.TW, 8035.T, NESN.SW, ASML.AS, HSBA.L
- approx_market_cap_usd: rough month-end market cap estimate in USD (number). Do NOT leave null.
- theme_revenue_ratio (TRR): estimate 0-1 (theme revenue / total revenue). You may output as a number or percent string.
- subtheme_bucket: e.g. "Foundry", "IDM", "Equipment", "EDA", "Memory", "Analog", "Materials", "Packaging", etc.
- If theme is ambiguous / not investable, return status="ambiguous" with theme_definition and 3-8 reference_etfs; rows=[].

Output:
Return ONLY JSON in this schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()

    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


def build_messages_summaries(theme: str, tickers: List[str]) -> List[Dict[str, str]]:
    schema = {"status": "ok|error", "rows": [{"ticker": "string", "theme_ja": "string", "non_theme_ja": "string"}]}
    user_prompt = f"""
Task: Write short Japanese summaries (institutional tone). No disclaimers.
Theme: "{theme}"
Tickers: {json.dumps(tickers, ensure_ascii=False)}

Constraints:
- theme_ja: 1 sentence, <= 120 Japanese characters
- non_theme_ja: 1 sentence, <= 90 Japanese characters

Return ONLY JSON:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": _ai_system_prompt()},
            {"role": "user", "content": user_prompt}]


def build_messages_refine_evidence(theme: str, region: RegionMode, asof: date, payload: List[Dict[str, Any]]) -> List[Dict[str, str]]:
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
{json.dumps(payload, ensure_ascii=False)}

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
# Normalizers (never raise)
# =============================================================================
def normalize_draft_from_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "rows": [], "notes": "LLM output not a JSON object."}
    status = str(obj.get("status","error"))
    if status not in ("ok","ambiguous","error"):
        status = "error"
    rows = obj.get("rows", []) or []
    if not isinstance(rows, list):
        rows = []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker","") or "").strip().upper()
        if not t:
            continue
        out.append({
            "rank_hint": int(r.get("rank_hint", len(out)+1) or (len(out)+1)),
            "company_name": str(r.get("company_name","") or "").strip(),
            "ticker": t,
            "listed_country": str(r.get("listed_country","") or "").strip(),
            "primary_exchange": str(r.get("primary_exchange","") or "").strip(),
            "subtheme_bucket": str(r.get("subtheme_bucket","") or "").strip(),
            "approx_market_cap_usd": parse_usd_number(r.get("approx_market_cap_usd", 0.0)),
            "theme_revenue_ratio": parse_trr(r.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(r.get("method","estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("confidence","Low") or "Low"),
        })
    # de-dup by ticker
    seen = set()
    dedup = []
    for r in out:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        dedup.append(r)
    return {
        "status": status,
        "theme_definition": str(obj.get("theme_definition","") or ""),
        "notes": str(obj.get("notes","") or ""),
        "reference_etfs": obj.get("reference_etfs", []) if isinstance(obj.get("reference_etfs", []), list) else [],
        "rows": dedup,
    }


def normalize_summaries(obj: Any) -> Dict[str, Dict[str, str]]:
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


def normalize_refine(obj: Any) -> Dict[str, Any]:
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
            "theme_revenue_ratio": parse_trr(r.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(r.get("method","estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("confidence","Low") or "Low"),
            "theme_revenue_sources": srcs[:2],
            "theme_business_summary_ja": str(r.get("theme_ja","") or "").strip(),
            "non_theme_business_summary_ja": str(r.get("non_theme_ja","") or "").strip(),
        })
    return {"status": "ok", "rows": out_rows, "notes": str(obj.get("notes","") or "")}


# =============================================================================
# yfinance validation (bounded)
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


def _month_end_close_chunked(tickers: List[str], asof: date, chunk: int = 15) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {t: None for t in tickers}
    start = (asof - timedelta(days=10)).isoformat()
    end = (asof + timedelta(days=1)).isoformat()
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i+chunk]
        try:
            df = _yf_download_cached(tuple(part), start=start, end=end)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for t in part:
                col = (t, "Close")
                if col in df.columns:
                    s = df[col].dropna()
                    out[t] = float(s.iloc[-1]) if not s.empty else None
        else:
            if "Close" in df.columns and len(part) == 1:
                s = df["Close"].dropna()
                out[part[0]] = float(s.iloc[-1]) if not s.empty else None
    return out


def compute_mktcap_asof_usd_bounded(tickers: List[str], asof: date, timeout_total_s: int = 12) -> pd.DataFrame:
    """
    Try to compute month-end market cap proxy for tickers, but never block longer than timeout_total_s.
    If it times out, return an empty DataFrame.
    """
    tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker","mcap_usd","mcap_quality"])

    def _work() -> pd.DataFrame:
        close_map = _month_end_close_chunked(tickers, asof, chunk=15)
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

            rows.append({"ticker": t, "mcap_usd": mcap_usd, "mcap_quality": quality + ("" if fx is not None else "|fx_unavailable")})

        df = pd.DataFrame(rows)
        df["mcap_usd"] = pd.to_numeric(df["mcap_usd"], errors="coerce")
        return df

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_work)
        try:
            return fut.result(timeout=float(timeout_total_s))
        except Exception:
            return pd.DataFrame(columns=["ticker","mcap_usd","mcap_quality"])


# =============================================================================
# Executor
# =============================================================================
@st.cache_resource
def _executor() -> cf.ThreadPoolExecutor:
    return cf.ThreadPoolExecutor(max_workers=1)


# =============================================================================
# Jobs
# =============================================================================
def job_draft_universe(*, inp: ThemeInput, data_dir: str, cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    pool = draft_pool_size(top_n)
    asof = most_recent_month_end(date.today())

    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    cache_key = _cache_key("draft_v14", theme, region, asof.isoformat(), str(pool), model)

    # Try cache first (parsed JSON only)
    cached = _cache_load(data_dir, cache_key)
    raw_text: str = ""
    if cached:
        obj = cached
    else:
        msgs = build_messages_draft_universe(theme, region, asof, pool)
        raw_text = gemini_generate_raw_text(
            messages=msgs,
            model=model,
            temperature=0.05,
            max_output_tokens=1900,
            timeout_s=14,
        )
        # Try parse
        obj = None
        try:
            parsed = _parse_json_any(raw_text)
            if isinstance(parsed, dict):
                obj = parsed
            else:
                obj = {"status": "ok", "rows": parsed}
        except Exception:
            obj = None

        # Cache only if parsed as dict
        if isinstance(obj, dict):
            _cache_save(data_dir, cache_key, obj)

    # Normalize if possible
    notes = ""
    theme_def = ""
    ref_etfs: List[str] = []
    rows: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        norm = normalize_draft_from_obj(obj)
        status = norm.get("status","error")
        notes = str(norm.get("notes","") or "")
        theme_def = str(norm.get("theme_definition","") or "")
        ref_etfs = norm.get("reference_etfs", []) or []
        rows = norm.get("rows", []) or []
        if status == "ambiguous":
            return {
                "status": "ambiguous",
                "asof": asof.isoformat(),
                "theme_definition": theme_def,
                "reference_etfs": ref_etfs,
                "notes": notes,
                "ai_raw_text": raw_text,
            }
    else:
        status = "error"

    # If no structured rows, fallback to ticker extraction from raw text
    if not rows:
        fallback_tickers = extract_tickers_from_text(raw_text, max_n=pool)
        if fallback_tickers:
            notes = (notes + " " if notes else "") + "Fallback: extracted tickers from raw AI output (non-JSON)."
        else:
            notes = (notes + " " if notes else "") + "Fallback: could not parse JSON; showing raw AI output."
        rows = []
        for i, t in enumerate(fallback_tickers):
            rows.append({
                "rank_hint": i + 1,
                "company_name": "",
                "ticker": t,
                "listed_country": "",
                "primary_exchange": "",
                "subtheme_bucket": "",
                "approx_market_cap_usd": 0.0,
                "theme_revenue_ratio": 0.0,
                "theme_revenue_ratio_method": "estimated",
                "theme_revenue_ratio_confidence": "Low",
            })

    # Build DataFrame (even if empty) — DO NOT FAIL
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
        "rank_hint","company_name","ticker","listed_country","primary_exchange","subtheme_bucket",
        "approx_market_cap_usd","theme_revenue_ratio","theme_revenue_ratio_method","theme_revenue_ratio_confidence"
    ])

    # Compute draft ranking fields robustly
    if not df.empty:
        df["approx_market_cap_usd"] = df["approx_market_cap_usd"].apply(parse_usd_number)
        df["theme_revenue_ratio"] = df["theme_revenue_ratio"].apply(parse_trr)
        df["theme_mktcap_usd"] = pd.to_numeric(df["approx_market_cap_usd"], errors="coerce").fillna(0.0) * pd.to_numeric(df["theme_revenue_ratio"], errors="coerce").fillna(0.0)

        df = df.sort_values(["theme_mktcap_usd","approx_market_cap_usd","rank_hint"], ascending=[False, False, True]).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df["mcap_usd"] = df["approx_market_cap_usd"].astype(float)
        df["mcap_quality"] = np.where(df["mcap_usd"] > 0, "ai_estimated_mktcap_usd", "unknown_mktcap")
    else:
        df["theme_mktcap_usd"] = []
        df["rank"] = []
        df["mcap_usd"] = []
        df["mcap_quality"] = []

    df["mktcap_asof_date"] = asof.isoformat()

    # placeholders for summaries & evidence (ranked list uses Validate step)
    if "theme_business_summary" not in df.columns:
        df["theme_business_summary"] = ""
    if "non_theme_business_summary" not in df.columns:
        df["non_theme_business_summary"] = ""
    for col in ["trr_source_title","trr_source_publisher","trr_source_year","trr_source_url","trr_source_locator","trr_source_excerpt","trr_sources_full"]:
        if col not in df.columns:
            df[col] = "" if col != "trr_source_year" else None

    return {
        "status": "ok",
        "mode": "draft",
        "asof": asof.isoformat(),
        "draft_pool": int(pool),
        "notes": notes,
        "theme_definition": theme_def,
        "reference_etfs": ref_etfs,
        "model": model,
        "draft_rows": df.to_dict(orient="records"),
        "ai_raw_text": raw_text,
        "cache_keys": {"draft": cache_key},
    }


def job_validate_and_rank(*, inp: ThemeInput, data_dir: str, draft_rows: List[Dict[str, Any]], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())

    df = pd.DataFrame(draft_rows)
    if df.empty or "ticker" not in df.columns:
        return {"status": "error", "stage": "validate", "notes": "Draft rows missing."}

    tickers = df["ticker"].astype(str).str.upper().tolist()
    tickers = list(dict.fromkeys([t.strip() for t in tickers if t.strip()]))

    # bounded month-end mktcap proxy
    mkt = compute_mktcap_asof_usd_bounded(tickers, asof, timeout_total_s=12)

    validated_count = 0
    if not mkt.empty:
        mkt = mkt.dropna(subset=["mcap_usd"])
        validated_count = int(mkt["mcap_usd"].notna().sum())
        df = df.merge(mkt, on="ticker", how="left", suffixes=("", "_val"))
        df["mcap_usd"] = np.where(df["mcap_usd_val"].notna(), df["mcap_usd_val"], df.get("mcap_usd", 0.0))
        df["mcap_quality"] = np.where(df["mcap_usd_val"].notna(), df["mcap_quality_val"], df.get("mcap_quality","ai_estimated_mktcap_usd"))
        df = df.drop(columns=[c for c in ["mcap_usd_val","mcap_quality_val"] if c in df.columns])

    # compute theme mktcap and rank
    df["theme_revenue_ratio"] = df.get("theme_revenue_ratio", 0.0).apply(parse_trr)
    df["mcap_usd"] = pd.to_numeric(df.get("mcap_usd", 0.0), errors="coerce").fillna(0.0)
    df["theme_mktcap_usd"] = df["mcap_usd"] * df["theme_revenue_ratio"]

    # soft large-cap filter only if we have validated mktcap
    thr = _large_cap_threshold_usd(region)
    df2 = df.copy()
    if validated_count > 0:
        df2 = df2[df2["mcap_usd"] >= thr]

    df2 = df2.sort_values(["theme_mktcap_usd","mcap_usd","rank_hint"], ascending=[False, False, True]).reset_index(drop=True)
    df2["rank"] = np.arange(1, len(df2) + 1)
    df2 = df2.head(top_n).copy()
    df2["mktcap_asof_date"] = asof.isoformat()

    # Summaries for Top-N only (best-effort)
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    top_tickers = df2["ticker"].astype(str).tolist()

    raw_sum = ""
    try:
        msgs_sum = build_messages_summaries(theme, top_tickers)
        raw_sum = gemini_generate_raw_text(
            messages=msgs_sum,
            model=model,
            temperature=0.15,
            max_output_tokens=900,
            timeout_s=10,
        )
        try:
            obj = _parse_json_any(raw_sum)
        except Exception:
            obj = {}
        smap = normalize_summaries(obj)
    except Exception:
        smap = {}

    df2["theme_business_summary"] = df2["ticker"].apply(lambda t: smap.get(t, {}).get("theme_ja","").strip())
    df2["non_theme_business_summary"] = df2["ticker"].apply(lambda t: smap.get(t, {}).get("non_theme_ja","").strip())
    df2["theme_business_summary"] = df2["theme_business_summary"].replace("", "テーマ関連は推計（Draft）。")
    df2["non_theme_business_summary"] = df2["non_theme_business_summary"].replace("", "非テーマ事業: 多角化（要確認）。")

    # evidence placeholders
    for col in ["trr_source_title","trr_source_publisher","trr_source_year","trr_source_url","trr_source_locator","trr_source_excerpt","trr_sources_full"]:
        if col not in df2.columns:
            df2[col] = "" if col != "trr_source_year" else None

    return {
        "status": "ok",
        "mode": "ranked",
        "asof": asof.isoformat(),
        "validated_mktcap_count": validated_count,
        "validated_mktcap_total": len(tickers),
        "ranked": df2.to_dict(orient="records"),
        "notes": "",
        "model": model,
        "ai_raw_text_summaries": raw_sum,
    }


def job_refine(*, inp: ThemeInput, data_dir: str, current_rows: List[Dict[str, Any]], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"

    df = pd.DataFrame(current_rows)
    if df.empty:
        return {"status": "error", "stage": "refine", "notes": "No ranked rows."}

    payload = []
    for _, r in df.head(top_n).iterrows():
        payload.append({
            "ticker": r.get("ticker"),
            "company_name": r.get("company_name"),
            "listed_country": r.get("listed_country"),
            "current_trr": r.get("theme_revenue_ratio"),
            "mcap_usd": r.get("mcap_usd"),
            "theme_mktcap_usd": r.get("theme_mktcap_usd"),
        })

    msgs = build_messages_refine_evidence(theme, region, asof, payload)
    raw = gemini_generate_raw_text(
        messages=msgs,
        model=model,
        temperature=0.10,
        max_output_tokens=2400,
        timeout_s=16,
    )
    obj = {}
    try:
        obj = _parse_json_any(raw)
    except Exception:
        obj = {"status":"error","notes":"Refine output was not JSON."}

    norm = normalize_refine(obj)
    if norm.get("status") != "ok":
        return {"status": "error", "stage": "refine", "notes": norm.get("notes","")}

    upd = {r["ticker"]: r for r in (norm.get("rows") or []) if isinstance(r, dict) and r.get("ticker")}

    def u(t: str, k: str, default: Any) -> Any:
        return upd.get(t, {}).get(k, default)

    df["theme_revenue_ratio"] = df["ticker"].apply(lambda t: parse_trr(u(t, "theme_revenue_ratio", df.loc[df["ticker"]==t, "theme_revenue_ratio"].values[0])))
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

    df["theme_mktcap_usd"] = pd.to_numeric(df.get("mcap_usd"), errors="coerce").fillna(0.0) * pd.to_numeric(df.get("theme_revenue_ratio"), errors="coerce").fillna(0.0)
    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return {
        "status": "ok",
        "mode": "refined",
        "asof": asof.isoformat(),
        "ranked": df.to_dict(orient="records"),
        "notes": str(norm.get("notes","") or ""),
        "model": model,
        "ai_raw_text_refine": raw,
    }


# =============================================================================
# Definitions (always displayed)
# =============================================================================
def build_definitions_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"Field": "Theme Market Cap", "Definition": "Theme Market Cap = Market Cap (USD, month-end target; validated if possible) × TRR"},
        {"Field": "TRR (Theme Revenue Ratio)", "Definition": "TRR = テーマ関連売上 ÷ 総売上（0〜1）"},
        {"Field": "Market Cap (validated)", "Definition": "Validate step uses month-end proxy: floatShares×Close (fallbacks: sharesOutstanding×Close / derived / marketCap field)"},
        {"Field": "Market Cap (draft)", "Definition": "Draft step uses AI estimated month-end market cap in USD (or unknown if model output is unstructured)"},
        {"Field": "As-of date", "Definition": "時価総額基準日（原則：直近月末）"},
        {"Field": "Method / Confidence", "Definition": "disclosed/proxy/estimated と High/Med/Low（推計は明示）"},
        {"Field": "Sources (Refine)", "Definition": "title/publisher/year/url/locator/excerpt。url無しの場合 excerpt は空文字"},
    ])


# =============================================================================
# UI
# =============================================================================
def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(THEMELENS_CSS, unsafe_allow_html=True)
    st.markdown('<div class="tl-title">THEMELENS</div>', unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("tl_draft_rows", None)   # draft list
    ss.setdefault("tl_rows", None)         # ranked top-N
    ss.setdefault("tl_meta", {})
    ss.setdefault("tl_job", None)
    ss.setdefault("tl_future", None)
    ss.setdefault("tl_cancel_event", None)

    # Controls
    st.markdown('<div class="tl-panel">', unsafe_allow_html=True)
    with st.form("tl_controls", clear_on_submit=False):
        theme_text = st.text_input("Theme", value=ss.get("tl_theme_text", "半導体"), placeholder="例: 半導体 / ゲーム / サイバー / 生成AI / 防衛 ...")
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.0, 1.2, 1.2, 1.2])
        with c1:
            region_mode = st.radio("Region", ["Global","Japan","US","Europe","China"], horizontal=True,
                                   index=["Global","Japan","US","Europe","China"].index(ss.get("tl_region_mode","Global")))
        with c2:
            top_n = st.number_input("Top N", min_value=1, max_value=30, value=int(ss.get("tl_top_n", 30)), step=1)
        pool = draft_pool_size(int(top_n))
        with c3:
            draft_btn = st.form_submit_button(f"Draft {pool}", type="primary", use_container_width=True)
        with c4:
            validate_btn = st.form_submit_button("Validate & Rank", type="secondary", use_container_width=True)
        with c5:
            refine_btn = st.form_submit_button("Refine", type="secondary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ss["tl_theme_text"] = theme_text
    ss["tl_region_mode"] = region_mode
    ss["tl_top_n"] = int(top_n)

    inp = ThemeInput(theme_text=str(theme_text or "").strip(), region_mode=region_mode, top_n=int(top_n))
    asof = most_recent_month_end(date.today())

    # Button validation
    if validate_btn and ss.get("tl_draft_rows") is None:
        st.warning("Run Draft first.")
        validate_btn = False
    if refine_btn and ss.get("tl_rows") is None:
        st.warning("Run Validate & Rank first (or at least have a ranked list).")
        refine_btn = False

    # Start jobs
    if draft_btn:
        ss["tl_draft_rows"] = []  # set to empty list immediately so UI can show "raw" later
        ss["tl_rows"] = None
        ss["tl_meta"] = {}
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "draft", "status": "running", "started_at": time.time()}
        ss["tl_future"] = _executor().submit(job_draft_universe, inp=inp, data_dir=data_dir, cancel_event=cancel_event)
        _rerun()

    if validate_btn:
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "validate", "status": "running", "started_at": time.time()}
        draft_rows = ss.get("tl_draft_rows") or []
        ss["tl_future"] = _executor().submit(job_validate_and_rank, inp=inp, data_dir=data_dir, draft_rows=draft_rows, cancel_event=cancel_event)
        _rerun()

    if refine_btn:
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "refine", "status": "running", "started_at": time.time()}
        current_rows = ss.get("tl_rows") or []
        ss["tl_future"] = _executor().submit(job_refine, inp=inp, data_dir=data_dir, current_rows=current_rows, cancel_event=cancel_event)
        _rerun()

    # Running UI
    job = ss.get("tl_job")
    fut = ss.get("tl_future")
    if job and job.get("status") == "running" and fut is not None:
        elapsed = time.time() - float(job.get("started_at", time.time()))
        mode = job.get("mode", "")

        if mode == "draft":
            st.info("Draft: Gemini returns a large-cap universe (N+20) with TRR + market-cap estimates. If JSON breaks, we still show the raw output and extract tickers.")
        elif mode == "validate":
            st.info("Validate & Rank: try to replace AI market caps with month-end proxy. Hard timeout is enforced.")
        else:
            st.info("Refine: add TRR sources/locators and improve summaries.")

        cA, cB, cC = st.columns([1.0, 1.0, 3.0])
        with cA:
            if st.button("Cancel", use_container_width=True):
                ev = ss.get("tl_cancel_event")
                if ev:
                    try: ev.set()
                    except Exception: pass
                try: fut.cancel()
                except Exception: pass
                ss["tl_job"] = None
                ss["tl_future"] = None
                st.warning("Cancelled.")
                st.stop()
        with cB:
            st.metric("Elapsed", f"{elapsed:.1f}s")
        with cC:
            st.caption("If it takes too long: cancel → at least the Draft stage can still show raw output/tickers.")

        budget = 16 if mode == "draft" else (18 if mode == "validate" else 22)
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
                # ALWAYS store draft_rows if present, even if empty
                if res.get("mode") == "draft":
                    ss["tl_draft_rows"] = res.get("draft_rows", ss.get("tl_draft_rows"))
                if res.get("mode") in ("ranked","refined"):
                    ss["tl_rows"] = res.get("ranked")
            else:
                ss["tl_meta"] = {"status": "error", "notes": "Unexpected result type."}

            _rerun()

        time.sleep(0.45)
        _rerun()

    # Output
    meta = ss.get("tl_meta") or {}
    draft_rows = ss.get("tl_draft_rows")   # may be None, [], or list
    rows = ss.get("tl_rows")
    ai_raw = str(meta.get("ai_raw_text","") or "").strip()

    # Always offer raw model output when present
    if ai_raw:
        with st.expander("AI raw output (debug)", expanded=False):
            st.text(ai_raw[:12000])  # cap for UI safety

    # Ambiguous
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

    # Draft list display — show even if empty list
    if draft_rows is not None:
        ddf = pd.DataFrame(draft_rows) if isinstance(draft_rows, list) else pd.DataFrame()
        st.markdown("#### Draft Universe")
        chips = []
        chips.append(f'<span class="tl-chip">As-of target: {asof.isoformat()}</span>')
        if not ddf.empty:
            chips.append(f'<span class="tl-chip">Pool: {len(ddf)}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)

        if ddf.empty:
            st.warning("No structured rows to display yet. If AI returned something, see 'AI raw output (debug)' above.")
        else:
            dview = ddf.copy()
            dview["TRR"] = (pd.to_numeric(dview.get("theme_revenue_ratio", 0.0), errors="coerce").fillna(0.0) * 100).round(1).astype(str) + "%"
            dview["MktCap (USD)"] = pd.to_numeric(dview.get("mcap_usd", 0.0), errors="coerce").apply(fmt_money)
            dview["Theme MktCap (USD)"] = pd.to_numeric(dview.get("theme_mktcap_usd", 0.0), errors="coerce").apply(fmt_money)
            cols = ["rank","company_name","ticker","listed_country","subtheme_bucket","TRR","MktCap (USD)","Theme MktCap (USD)","theme_revenue_ratio_method","theme_revenue_ratio_confidence","mcap_quality"]
            cols = [c for c in cols if c in dview.columns]
            st.dataframe(dview[cols], use_container_width=True, height=420)

    # Ranked deliverable
    if rows:
        df = pd.DataFrame(rows)
        if not df.empty:
            st.subheader("Deliverable: Ranked List (Theme Market Cap, desc)")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("As-of (MktCap)", str(df.get("mktcap_asof_date", pd.Series([asof.isoformat()])).iloc[0]))
            with k2:
                st.metric("Region", region_mode)
            with k3:
                st.metric("Top N", int(top_n))
            with k4:
                st.metric("Mode", str(meta.get("mode","")))

            if meta.get("mode") == "ranked":
                vc = meta.get("validated_mktcap_count", 0)
                vt = meta.get("validated_mktcap_total", 0)
                st.caption(f"Validated market cap: {vc}/{vt} tickers. (Unvalidated use Draft estimates.)")

            view = df.copy()
            view["TRR"] = (pd.to_numeric(view.get("theme_revenue_ratio", 0.0), errors="coerce").fillna(0.0) * 100).round(1).astype(str) + "%"
            view["MktCap (USD)"] = pd.to_numeric(view.get("mcap_usd", 0.0), errors="coerce").apply(fmt_money)
            view["Theme MktCap (USD)"] = pd.to_numeric(view.get("theme_mktcap_usd", 0.0), errors="coerce").apply(fmt_money)

            show_cols = [
                "rank","company_name","ticker","listed_country",
                "theme_business_summary","TRR","MktCap (USD)","Theme MktCap (USD)",
                "non_theme_business_summary",
                "trr_source_title","trr_source_publisher","trr_source_year","trr_source_locator",
                "mcap_quality"
            ]
            show_cols = [c for c in show_cols if c in view.columns]
            st.dataframe(view[show_cols], use_container_width=True, height=560)

            sid = _sha12(f"{theme_text}|{region_mode}|{asof.isoformat()}|{top_n}")
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"themelens_ranked_list_{sid}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download JSON (snapshot)",
                data=json.dumps({"meta": meta, "draft_rows": draft_rows, "rows": rows}, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"themelens_snapshot_{sid}.json",
                mime="application/json",
            )

    # Definitions (always visible)
    st.markdown("### Definitions")
    st.markdown(
        "- **Theme Market Cap** = Market Cap (USD, month-end target; validated if possible) × **TRR**\n"
        "- **TRR (Theme Revenue Ratio)** = テーマ関連売上 ÷ 総売上（0〜1）\n"
        "- **Draft** = AI推計（market cap / TRR）。JSONが壊れても raw を表示し、ticker抽出で最低限のリストを作ります\n"
        "- **Validate & Rank** = 月末時価総額proxyを取得できた銘柄だけ置換し、再ランキング"
    )
    st.dataframe(build_definitions_table(), use_container_width=True, height=330)
