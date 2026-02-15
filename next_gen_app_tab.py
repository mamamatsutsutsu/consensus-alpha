# THEMELENS — AI-first Theme Portfolio Builder (Gemini) v17
# -----------------------------------------------------------------------------
# Requested UX + correctness improvements:
# - China region includes Taiwan (display: "China (incl. Taiwan)")
# - Theme input: stylish English helper text above the field
# - Top N max = 20
# - Do NOT show Draft Universe (N+20) table
# - After 60s: show "too slow, you may cancel" message
# - Definitions: remove definitions table; show a stylish definition block ABOVE the ranked list area
# - TRR must exist: strengthen prompts + add automatic TRR repair pass for missing/zero TRR
# - Deliverable table should be filled:
#   * Fill company/exchange/listed-country from yfinance + ticker suffix inference if AI missing
#   * Ensure TRR exists for ranking (best-effort estimate if no disclosure)
#
# Deliverable: ranked list by Theme Market Cap = Market Cap (USD, month-end target; validated if possible) × TRR.
# TRR is evidence-backed when possible; otherwise explicitly marked as estimated.

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
# Styling
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
  margin: 0 0 8px 0;
  font-size: 28px;
}

.tl-subtitle{
  opacity:0.78;
  letter-spacing:0.03em;
  margin: 0 0 16px 0;
  font-size: 13px;
}

.tl-panel{
  border-radius: 18px;
  border: 1px solid var(--al-border);
  background: var(--al-bg);
  backdrop-filter: blur(10px);
  padding: 14px 14px;
}

.tl-hero{
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(0,0,0,0.42);
  padding: 14px 14px;
}

.tl-hero .kicker{
  font-family:Orbitron, ui-sans-serif, system-ui;
  letter-spacing:0.10em;
  color: var(--al-cyan);
  font-weight: 800;
  font-size: 12px;
}
.tl-hero .body{
  opacity:0.86;
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.45;
}
.tl-hero .fine{
  opacity:0.65;
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.40;
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
    # Global must include KR/TW to avoid missing industry leaders.
    if region == "Global":
        return (
            "Global large-caps: US, Japan, Developed Europe (EU+UK+Switzerland), "
            "plus major Asia large-cap markets that dominate global industries (Korea, Taiwan), "
            "and China (Mainland + HK). Avoid frontier/illiquid markets."
        )
    if region == "Japan":
        return "Japan listed equities (TSE Prime etc)."
    if region == "US":
        return "United States listed equities (NYSE/NASDAQ etc)."
    if region == "Europe":
        return "Developed Europe: EU + UK + Switzerland (exclude Russia)."
    if region == "China":
        # User request: include Taiwan in China mode.
        return "Greater China large-caps: Mainland A-shares + Hong Kong + Taiwan (TW)."
    return "Global"


def _large_cap_threshold_usd(region: RegionMode) -> float:
    if region in ("US", "Global"):
        return 10e9
    if region == "Japan":
        return 5e9
    if region in ("Europe", "China"):
        return 8e9
    return 8e9


def clamp_top_n(n: int) -> int:
    return max(1, min(int(n), 20))


def draft_pool_size(top_n: int) -> int:
    n = clamp_top_n(top_n)
    # N+20 (strictly)
    return int(n + 20)


# =============================================================================
# Robust parsers
# =============================================================================
_NUM_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

def parse_usd_number(v: Any) -> float:
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
    s = s.replace("USD", "").replace("US$", "").replace("$", "").replace(" ", "")
    s = s.replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)([KMBT])?$", s, flags=re.IGNORECASE)
    if m:
        num = float(m.group(1))
        suf = (m.group(2) or "").upper()
        return num * _NUM_SUFFIX.get(suf, 1.0)
    m2 = re.search(r"([0-9]*\.?[0-9]+)\s*([KMBT])", s, flags=re.IGNORECASE)
    if m2:
        num = float(m2.group(1))
        suf = m2.group(2).upper()
        return num * _NUM_SUFFIX.get(suf, 1.0)
    m3 = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m3:
        try:
            return float(m3.group(1))
        except Exception:
            return 0.0
    return 0.0


def parse_trr(v: Any) -> float:
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
    m = re.search(r"([0-9]*\.?[0-9]+)", s)
    if not m:
        return 0.0
    try:
        x = float(m.group(1))
        if x > 1.5:
            x = x / 100.0
        return clamp01(x)
    except Exception:
        return 0.0


# =============================================================================
# Country inference (listed country)
# =============================================================================
_SUFFIX_COUNTRY = {
    "T": "Japan",
    "HK": "Hong Kong",
    "SS": "China",
    "SZ": "China",
    "KS": "South Korea",
    "KQ": "South Korea",
    "TW": "Taiwan",
    "SW": "Switzerland",
    "L": "United Kingdom",
    "AS": "Netherlands",
    "PA": "France",
    "DE": "Germany",
    "MI": "Italy",
    "MC": "Spain",
    "ST": "Sweden",
    "HE": "Finland",
    "CO": "Denmark",
    "OL": "Norway",
    "BR": "Belgium",
    "VI": "Austria",
    "IR": "Ireland",
    "TO": "Canada",
    "AX": "Australia",
    "SG": "Singapore",
}

def infer_listed_country(ticker: str, info: Dict[str, Any]) -> str:
    t = (ticker or "").upper().strip()
    if "." in t:
        suf = t.split(".")[-1]
        if suf in _SUFFIX_COUNTRY:
            return _SUFFIX_COUNTRY[suf]
    exch = str(info.get("exchange") or info.get("fullExchangeName") or "").upper()
    # crude US detection
    if exch in ("NYQ","NMS","NAS","NGM","ASE","BATS","CBOE"):
        return "United States"
    if "NYSE" in exch or "NASDAQ" in exch:
        return "United States"
    return ""


# =============================================================================
# Disk cache
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
# Gemini (REST-first)
# =============================================================================
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
    return system_txt, "\n\n".join([p for p in user_parts if p]).strip()


def gemini_generate_raw_text(
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
    response_mime_type: str,
) -> Tuple[str, Dict[str, Any]]:
    system_txt, user_txt = _compose_prompt(messages)
    if not user_txt:
        raise RuntimeError("Gemini call: missing user prompt.")
    api_key = _get_secret("GEMINI_API_KEY") or _get_secret("GOOGLE_API_KEY")
    debug: Dict[str, Any] = {}

    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": user_txt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens),
                "responseMimeType": response_mime_type,
            },
        }
        if system_txt:
            payload["systemInstruction"] = {"role": "system", "parts": [{"text": system_txt}]}

        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=int(timeout_s))
        r.raise_for_status()
        data = r.json()
        debug["promptFeedback"] = data.get("promptFeedback")
        candidates = data.get("candidates") or []
        if not candidates:
            debug["no_candidates"] = True
            return "", debug
        cand0 = candidates[0]
        debug["finishReason"] = cand0.get("finishReason")
        content = (cand0.get("content") or {})
        parts = (content.get("parts") or [])
        chunks: List[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if "text" in p:
                chunks.append(str(p.get("text","") or ""))
            elif "inlineData" in p:
                chunks.append(json.dumps(p.get("inlineData"), ensure_ascii=False))
            elif "functionCall" in p:
                chunks.append(json.dumps(p.get("functionCall"), ensure_ascii=False))
        return "\n".join(chunks).strip(), debug

    # SDK fallback (Vertex/ADC)
    try:
        from google.genai import types  # type: ignore
        from google import genai  # type: ignore
        client = genai.Client()
        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
            system_instruction=system_txt or None,
            response_mime_type=response_mime_type,
            max_output_tokens=int(max_output_tokens),
        )
        resp = client.models.generate_content(model=model, contents=user_txt, config=cfg)
        txt = getattr(resp, "text", "") or ""
        if callable(txt):
            txt = txt()
        return str(txt or "").strip(), debug
    except Exception as e:
        raise RuntimeError(f"Gemini call failed. Configure GEMINI_API_KEY/GOOGLE_API_KEY. Error: {e}")


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


# =============================================================================
# Ticker parsing (strict lines)
# =============================================================================
_TICKER_RE = re.compile(r"^[0-9A-Z]{1,10}(?:[-][0-9A-Z]{1,4})?(?:\.[A-Z]{1,6})?$")

def _clean_ticker_line(line: str) -> str:
    s = (line or "").strip().upper()
    s = re.sub(r"^\s*(?:-|\*|•|\d+[\).\]]\s*)", "", s).strip()
    s = re.split(r"\s+", s)[0].strip()
    s = s.replace(",", "")
    return s


def parse_tickers_lines(txt: str, max_n: int) -> List[str]:
    lines = [ln for ln in (txt or "").splitlines() if ln.strip()]
    out: List[str] = []
    seen = set()
    for ln in lines:
        t = _clean_ticker_line(ln)
        if not t:
            continue
        if not _TICKER_RE.match(t):
            continue
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= max_n:
            break
    return out


# =============================================================================
# Prompts
# =============================================================================
def _ai_system() -> str:
    return "You are a senior global equity PM and sector analyst. Follow instructions exactly."

def build_messages_tickers_only_strict(theme: str, region: RegionMode, pool: int) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    user_prompt = f"""
Return ONLY tickers, one ticker per line. No numbering, no bullets, no extra text.

Theme: "{theme}"
Region: {region} (definition: {region_def})

HARD RULES (MUST, otherwise invalid):
- Output EXACTLY {pool} tickers (exactly {pool} non-empty lines).
- LARGE-CAP, index-sized, liquid public equities only.
- yfinance-compatible tickers WITH suffixes when needed (0700.HK, 2330.TW, 005930.KS, 8035.T, ASML.AS, NESN.SW, HSBA.L).
- Do NOT include ETFs, funds, private companies.
- Sort by estimated Theme Market Cap (Market Cap × TRR) descending.

If you are tempted to stop early, continue until you reach EXACTLY {pool} lines.
""".strip()
    return [{"role": "system", "content": "Return only plain text tickers. Follow the exact line-count."},
            {"role": "user", "content": user_prompt}]


def build_messages_tickers_topup(theme: str, region: RegionMode, need: int, exclude: List[str]) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    user_prompt = f"""
Return ONLY tickers, one ticker per line. No numbering, no bullets, no extra text.

Theme: "{theme}"
Region: {region} (definition: {region_def})

Need EXACTLY {need} additional tickers (exactly {need} non-empty lines).
MUST NOT include any of these tickers:
{json.dumps(exclude, ensure_ascii=False)}

Constraints:
- LARGE-CAP only, yfinance tickers with suffixes when needed.
- No ETFs/funds.
- Sort by estimated Theme Market Cap (desc) among the additional tickers.
""".strip()
    return [{"role": "system", "content": "Return only plain text tickers. Follow the exact line-count."},
            {"role": "user", "content": user_prompt}]


def build_messages_trr_only(theme: str, region: RegionMode, tickers: List[str]) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    schema = {
        "status": "ok|error",
        "rows": [
            {
                "ticker": "string",
                "subtheme_bucket": "string",
                "theme_revenue_ratio": 0.0,
                "method": "disclosed|proxy|estimated",
                "confidence": "High|Med|Low",
            }
        ]
    }
    example = {
        "status": "ok",
        "rows": [
            {"ticker":"NVDA","subtheme_bucket":"GPU/AI","theme_revenue_ratio":0.95,"method":"estimated","confidence":"Med"}
        ]
    }
    user_prompt = f"""
Return ONLY JSON. No markdown.

Task: Estimate TRR (Theme Revenue Ratio) for each ticker.

Theme: "{theme}"
Region: {region} (definition: {region_def})

Tickers (MUST cover ALL, no omissions, no extras):
{json.dumps(tickers, ensure_ascii=False)}

Critical requirements:
- TRR MUST be present for EVERY ticker: a number in [0,1].
- Do NOT return 0.0 unless theme exposure is genuinely negligible (<1% of revenue). If uncertain, provide a defensible estimate and set method="estimated" confidence="Low".
- Prefer disclosed segment revenue if you know it; otherwise use best proxy/estimate.
- subtheme_bucket: short label (<=18 chars).

Schema:
{json.dumps(schema, ensure_ascii=False)}
Example (shape only):
{json.dumps(example, ensure_ascii=False)}
""".strip()
    return [{"role":"system","content":_ai_system()},
            {"role":"user","content":user_prompt}]


def build_messages_summaries(theme: str, tickers: List[str]) -> List[Dict[str, str]]:
    schema = {"status": "ok|error", "rows": [{"ticker": "string", "theme_ja": "string", "non_theme_ja": "string"}]}
    user_prompt = f"""
Return ONLY JSON.

Task: Write short Japanese summaries (institutional tone).
Theme: "{theme}"
Tickers: {json.dumps(tickers, ensure_ascii=False)}

Constraints:
- theme_ja: 1 sentence, <= 120 Japanese characters
- non_theme_ja: 1 sentence, <= 90 Japanese characters
Schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": _ai_system()},
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
Return ONLY JSON.

Task: REFINE Top-N. Add TRR evidence metadata and improve summaries.
Theme: "{theme}"
Region: {region} (definition: {region_def})
As-of: {asof.isoformat()}

Top-N payload:
{json.dumps(payload, ensure_ascii=False)}

Rules (critical):
- NEVER invent URLs or quotes.
- If you cannot provide a verified URL + exact excerpt, set url=null and excerpt="" and keep method="estimated" confidence="Low".
- Prefer filings/annual report/IR decks. Up to 2 sources per company.

Schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    return [{"role": "system", "content": _ai_system()},
            {"role": "user", "content": user_prompt}]


# =============================================================================
# Normalizers (key drift tolerant)
# =============================================================================
def _get_list_any(obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in obj:
            return obj.get(k)
    return None


def normalize_trr(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status":"error","rows":[]}
    status = str(obj.get("status","ok"))
    if status not in ("ok","error","ambiguous"):
        status = "ok"
    raw_rows = _get_list_any(obj, ["rows","companies","items","data","results"])
    if raw_rows is None:
        raw_rows = _get_list_any(obj, ["tickers","symbols"])
    rows: List[Dict[str, Any]] = []
    if not isinstance(raw_rows, list):
        raw_rows = []
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker", r.get("symbol","")) or "").strip().upper()
        if not t:
            continue
        rows.append({
            "ticker": t,
            "subtheme_bucket": str(r.get("subtheme_bucket", r.get("subtheme","")) or "").strip(),
            "theme_revenue_ratio": parse_trr(r.get("theme_revenue_ratio", r.get("trr", 0.0))),
            "theme_revenue_ratio_method": str(r.get("method","estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("confidence","Low") or "Low"),
        })
    return {"status": status, "rows": rows}


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
        out[t] = {"theme_ja": str(r.get("theme_ja","") or "").strip(),
                  "non_theme_ja": str(r.get("non_theme_ja","") or "").strip()}
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
# yfinance helpers (cached)
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
    tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker","mcap_usd","mcap_quality","company_name_yf","exchange_yf","listed_country_yf"])

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

            company = str(info.get("shortName") or info.get("longName") or "")[:80]
            exch = str(info.get("fullExchangeName") or info.get("exchange") or "")
            listed = infer_listed_country(t, info)

            rows.append({
                "ticker": t,
                "mcap_usd": mcap_usd,
                "mcap_quality": quality + ("" if fx is not None else "|fx_unavailable"),
                "company_name_yf": company,
                "exchange_yf": exch,
                "listed_country_yf": listed,
            })

        df = pd.DataFrame(rows)
        df["mcap_usd"] = pd.to_numeric(df["mcap_usd"], errors="coerce")
        return df

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_work)
        try:
            return fut.result(timeout=float(timeout_total_s))
        except Exception:
            return pd.DataFrame(columns=["ticker","mcap_usd","mcap_quality","company_name_yf","exchange_yf","listed_country_yf"])


# =============================================================================
# Executor (single worker)
# =============================================================================
@st.cache_resource
def _executor() -> cf.ThreadPoolExecutor:
    return cf.ThreadPoolExecutor(max_workers=1)


# =============================================================================
# Jobs (Auto pipeline)
# =============================================================================
def job_auto_tickers(*, inp: ThemeInput, data_dir: str, cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = clamp_top_n(inp.top_n)
    pool = draft_pool_size(top_n)
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    key = _cache_key("tickers_v17", theme, region, str(pool), model)

    cached = _cache_load(data_dir, key)
    if isinstance(cached, dict) and isinstance(cached.get("tickers"), list) and len(cached.get("tickers")) >= pool:
        tickers = [str(t).strip().upper() for t in cached.get("tickers")][:pool]
        return {"status":"ok","tickers":tickers,"pool":pool,"model":model,"notes":"(cache)","ai_raw_text":str(cached.get("ai_raw_text","") or "")}

    msgs = build_messages_tickers_only_strict(theme, region, pool)
    raw, dbg = gemini_generate_raw_text(
        messages=msgs, model=model, temperature=0.05, max_output_tokens=900, timeout_s=10, response_mime_type="text/plain"
    )
    tickers = parse_tickers_lines(raw, max_n=pool)

    # Top-up if short (strictly additional tickers)
    attempts = 0
    while len(tickers) < pool and attempts < 2 and not cancel_event.is_set():
        need = pool - len(tickers)
        msgs2 = build_messages_tickers_topup(theme, region, need=need, exclude=tickers)
        raw2, dbg2 = gemini_generate_raw_text(
            messages=msgs2, model=model, temperature=0.05, max_output_tokens=700, timeout_s=10, response_mime_type="text/plain"
        )
        add = parse_tickers_lines(raw2, max_n=need)
        for t in add:
            if t not in tickers:
                tickers.append(t)
        raw = (raw + "\n" + raw2).strip()
        dbg = {"first": dbg, "topup": dbg2}
        attempts += 1

    tickers = tickers[:pool]
    notes = ""
    if len(tickers) < pool:
        notes = f"WARNING: Only {len(tickers)}/{pool} tickers obtained. (Proceeding.)"

    payload = {"tickers": tickers, "ai_raw_text": raw}
    if len(tickers) >= pool:
        _cache_save(data_dir, key, payload)

    return {"status":"ok","tickers":tickers,"pool":pool,"model":model,"notes":notes,"ai_raw_text":raw,"debug_meta":dbg}


def job_auto_trr(*, inp: ThemeInput, data_dir: str, tickers: List[str], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = clamp_top_n(inp.top_n)
    pool = draft_pool_size(top_n)
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    asof = most_recent_month_end(date.today())

    tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))[:pool]

    # Chunk (reduce JSON failures)
    chunk_size = 25 if len(tickers) > 25 else len(tickers)
    chunks: List[List[str]] = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

    rows_map: Dict[str, Dict[str, Any]] = {}
    raw_all: List[str] = []

    for ch in chunks:
        if cancel_event.is_set():
            break
        key = _cache_key("trr_v17", theme, region, asof.isoformat(), _sha12(",".join(ch)), model)
        cached = _cache_load(data_dir, key)
        obj = cached if isinstance(cached, dict) else None

        if obj is None:
            msgs = build_messages_trr_only(theme, region, ch)
            raw, dbg = gemini_generate_raw_text(
                messages=msgs, model=model, temperature=0.10, max_output_tokens=1800, timeout_s=14, response_mime_type="application/json"
            )
            raw_all.append(raw)
            try:
                parsed = _parse_json_any(raw)
                obj = parsed if isinstance(parsed, dict) else {"status":"ok","rows":parsed}
            except Exception:
                obj = {"status":"error","rows":[]}
            # cache if rows exist
            if isinstance(obj, dict) and normalize_trr(obj).get("rows"):
                _cache_save(data_dir, key, obj)

        norm = normalize_trr(obj)
        for r in (norm.get("rows") or []):
            t = str(r.get("ticker","") or "").strip().upper()
            if not t:
                continue
            rows_map[t] = r

    # Repair pass: if TRR missing/zero for many, re-ask only those tickers once.
    missing = []
    for t in tickers:
        trr = parse_trr(rows_map.get(t, {}).get("theme_revenue_ratio", 0.0))
        if trr <= 0.0:
            missing.append(t)
    if missing and not cancel_event.is_set():
        # One repair call (chunked)
        miss_chunks = [missing[i:i+20] for i in range(0, len(missing), 20)]
        for mch in miss_chunks:
            if cancel_event.is_set():
                break
            msgs = build_messages_trr_only(theme, region, mch)
            raw, _dbg = gemini_generate_raw_text(
                messages=msgs, model=model, temperature=0.10, max_output_tokens=1500, timeout_s=12, response_mime_type="application/json"
            )
            raw_all.append(raw)
            try:
                obj2 = _parse_json_any(raw)
            except Exception:
                obj2 = {"status":"error","rows":[]}
            norm2 = normalize_trr(obj2)
            for r in (norm2.get("rows") or []):
                t = str(r.get("ticker","") or "").strip().upper()
                if t:
                    rows_map[t] = r

    # Final rows for every ticker (fallback TRR if still missing)
    out_rows: List[Dict[str, Any]] = []
    for t in tickers:
        r = rows_map.get(t, {})
        trr = parse_trr(r.get("theme_revenue_ratio", 0.0))
        if trr <= 0.0:
            # conservative fallback to keep ranking operational
            trr = 0.05
            method = "estimated"
            conf = "Low"
        else:
            method = str(r.get("theme_revenue_ratio_method", r.get("method","estimated")) or "estimated")
            conf = str(r.get("theme_revenue_ratio_confidence", r.get("confidence","Low")) or "Low")

        out_rows.append({
            "ticker": t,
            "subtheme_bucket": str(r.get("subtheme_bucket","") or "")[:18],
            "theme_revenue_ratio": float(trr),
            "theme_revenue_ratio_method": method,
            "theme_revenue_ratio_confidence": conf,
        })

    df = pd.DataFrame(out_rows)
    df["theme_revenue_ratio"] = df["theme_revenue_ratio"].apply(parse_trr)
    return {"status":"ok","asof":asof.isoformat(),"trr_rows":df.to_dict(orient="records"),"ai_raw_text":"\n\n---\n\n".join(raw_all)}


def job_validate_rank_and_summarize(*, inp: ThemeInput, data_dir: str, tickers: List[str], trr_rows: List[Dict[str, Any]], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = clamp_top_n(inp.top_n)
    asof = most_recent_month_end(date.today())

    tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return {"status":"error","notes":"No tickers."}

    trr_df = pd.DataFrame(trr_rows or [])
    if trr_df.empty:
        trr_df = pd.DataFrame({"ticker": tickers, "theme_revenue_ratio": [0.05]*len(tickers),
                               "theme_revenue_ratio_method": ["estimated"]*len(tickers),
                               "theme_revenue_ratio_confidence": ["Low"]*len(tickers),
                               "subtheme_bucket": [""]*len(tickers)})

    trr_df["ticker"] = trr_df["ticker"].astype(str).str.upper()
    trr_df = trr_df.drop_duplicates(subset=["ticker"], keep="first")

    # Market cap + yfinance metadata (bounded)
    mkt = compute_mktcap_asof_usd_bounded(tickers, asof, timeout_total_s=12)

    df = pd.DataFrame({"ticker": tickers})
    df = df.merge(trr_df, on="ticker", how="left")

    # Attach market cap if available
    if not mkt.empty:
        df = df.merge(mkt, on="ticker", how="left")
    else:
        df["mcap_usd"] = np.nan
        df["mcap_quality"] = "missing"

    # Fill company_name / exchange / listed_country from yfinance (best effort)
    df["company_name"] = df.get("company_name_yf", "")
    df["primary_exchange"] = df.get("exchange_yf", "")
    df["listed_country"] = df.get("listed_country_yf", "")

    for col in ["company_name","primary_exchange","listed_country"]:
        if col not in df.columns:
            df[col] = ""

    # Some may still be empty; last-resort country inference from ticker suffix
    df["listed_country"] = df.apply(lambda r: r["listed_country"] if str(r.get("listed_country","")).strip() else infer_listed_country(str(r["ticker"]), {}), axis=1)

    # Compute theme market cap
    df["mcap_usd"] = pd.to_numeric(df.get("mcap_usd"), errors="coerce")
    # Fallback if mcap missing: use yfinance current marketCap field quickly (cached info)
    missing_mcap = df["mcap_usd"].isna() | (df["mcap_usd"] <= 0)
    if missing_mcap.any():
        info_map = _batch_infos(df.loc[missing_mcap, "ticker"].tolist(), max_workers=14)
        mcap_now = []
        for t in df.loc[missing_mcap, "ticker"].tolist():
            info = info_map.get(t, {}) or {}
            mcap_now.append(float(info.get("marketCap") or 0.0))
        df.loc[missing_mcap, "mcap_usd"] = pd.to_numeric(mcap_now, errors="coerce")
        df.loc[missing_mcap, "mcap_quality"] = df.loc[missing_mcap, "mcap_quality"].astype(str) + "|fallback_current_marketCap"

        # Fill company/exchange if missing
        def _fill_name(t: str) -> str:
            info = info_map.get(t, {}) or {}
            return str(info.get("shortName") or info.get("longName") or "")
        def _fill_ex(t: str) -> str:
            info = info_map.get(t, {}) or {}
            return str(info.get("fullExchangeName") or info.get("exchange") or "")
        df.loc[missing_mcap, "company_name"] = df.loc[missing_mcap, "company_name"].astype(str).replace("", np.nan)
        df.loc[missing_mcap, "company_name"] = df.loc[missing_mcap, "company_name"].fillna(df.loc[missing_mcap, "ticker"].apply(_fill_name))
        df.loc[missing_mcap, "primary_exchange"] = df.loc[missing_mcap, "primary_exchange"].astype(str).replace("", np.nan)
        df.loc[missing_mcap, "primary_exchange"] = df.loc[missing_mcap, "primary_exchange"].fillna(df.loc[missing_mcap, "ticker"].apply(_fill_ex))
        df.loc[missing_mcap, "listed_country"] = df.loc[missing_mcap, "listed_country"].astype(str).replace("", np.nan)
        df.loc[missing_mcap, "listed_country"] = df.loc[missing_mcap, "listed_country"].fillna(df.loc[missing_mcap, "ticker"].apply(lambda t: infer_listed_country(t, info_map.get(t, {}) or {})))

    df["theme_revenue_ratio"] = df.get("theme_revenue_ratio", 0.05).apply(parse_trr)
    df["theme_mktcap_usd"] = pd.to_numeric(df.get("mcap_usd"), errors="coerce").fillna(0.0) * pd.to_numeric(df.get("theme_revenue_ratio"), errors="coerce").fillna(0.0)

    # Large-cap filter only if mcap present
    thr = _large_cap_threshold_usd(region)
    df2 = df.copy()
    df2 = df2[df2["mcap_usd"].fillna(0.0) >= thr] if df2["mcap_usd"].notna().any() else df2

    df2 = df2.sort_values(["theme_mktcap_usd","mcap_usd"], ascending=[False, False]).reset_index(drop=True)
    df2["rank"] = np.arange(1, len(df2) + 1)
    df2 = df2.head(top_n).copy()
    df2["mktcap_asof_date"] = asof.isoformat()

    # Summaries (best-effort)
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    top_tickers = df2["ticker"].astype(str).tolist()

    smap: Dict[str, Dict[str, str]] = {}
    raw_sum = ""
    try:
        msgs = build_messages_summaries(theme, top_tickers)
        raw_sum, _dbg = gemini_generate_raw_text(
            messages=msgs, model=model, temperature=0.15, max_output_tokens=900, timeout_s=10, response_mime_type="application/json"
        )
        try:
            obj = _parse_json_any(raw_sum)
        except Exception:
            obj = {}
        smap = normalize_summaries(obj)
    except Exception:
        smap = {}

    df2["theme_business_summary"] = df2["ticker"].apply(lambda t: smap.get(t, {}).get("theme_ja","").strip() or "テーマ関連は推計。")
    df2["non_theme_business_summary"] = df2["ticker"].apply(lambda t: smap.get(t, {}).get("non_theme_ja","").strip() or "非テーマ事業: 多角化（要確認）。")

    # Source placeholders (refine stage fills)
    for col in ["trr_source_title","trr_source_publisher","trr_source_year","trr_source_url","trr_source_locator","trr_source_excerpt","trr_sources_full"]:
        df2[col] = "" if col != "trr_source_year" else None

    return {"status":"ok","mode":"ranked","asof":asof.isoformat(),"ranked":df2.to_dict(orient="records"),"ai_raw_text_summaries":raw_sum}


def job_refine(*, inp: ThemeInput, data_dir: str, current_rows: List[Dict[str, Any]], cancel_event: threading.Event) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = clamp_top_n(inp.top_n)
    asof = most_recent_month_end(date.today())
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"

    df = pd.DataFrame(current_rows or [])
    if df.empty:
        return {"status":"error","notes":"No ranked rows."}

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
    raw, _dbg = gemini_generate_raw_text(
        messages=msgs, model=model, temperature=0.10, max_output_tokens=2400, timeout_s=16, response_mime_type="application/json"
    )
    try:
        obj = _parse_json_any(raw)
    except Exception:
        obj = {"status":"error","notes":"Refine output was not JSON."}

    norm = normalize_refine(obj)
    if norm.get("status") != "ok":
        return {"status":"error","notes": norm.get("notes","Refine failed")}

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

    return {"status":"ok","mode":"refined","asof":asof.isoformat(),"ranked":df.to_dict(orient="records"),"ai_raw_text_refine":raw}


# =============================================================================
# Display helper: start from Rank (hide index if possible)
# =============================================================================
def show_table_start_rank(df: pd.DataFrame, cols: List[str], height: int) -> None:
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()
    if "rank" in view.columns:
        ordered = ["rank"] + [c for c in view.columns if c != "rank"]
        view = view[ordered]
    try:
        st.dataframe(view, use_container_width=True, height=height, hide_index=True)
    except TypeError:
        if "rank" in view.columns:
            v2 = view.set_index("rank")
            st.dataframe(v2, use_container_width=True, height=height)
        else:
            st.dataframe(view, use_container_width=True, height=height)


# =============================================================================
# UI
# =============================================================================
def _format_region_label(r: str) -> str:
    if r == "China":
        return "China (incl. Taiwan)"
    return r


def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(THEMELENS_CSS, unsafe_allow_html=True)
    st.markdown('<div class="tl-title">THEMELENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="tl-subtitle">AI-first Theme Portfolio Builder</div>', unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("tl_rows", None)           # ranked topN
    ss.setdefault("tl_meta", {})
    ss.setdefault("tl_job", None)
    ss.setdefault("tl_future", None)
    ss.setdefault("tl_cancel_event", None)
    ss.setdefault("tl_pipeline_start", None)
    ss.setdefault("tl_elapsed_final", None)
    ss.setdefault("tl_stage", "idle")
    ss.setdefault("tl_universe_tickers", None)
    ss.setdefault("tl_trr_rows", None)

    # Controls (top, no sidebar)
    st.markdown('<div class="tl-panel">', unsafe_allow_html=True)
    with st.form("tl_controls", clear_on_submit=False):
        # Stylish helper above Theme
        st.markdown(
            '<div style="opacity:0.72;margin:2px 2px 8px 2px;letter-spacing:0.03em;">'
            'Type any theme you want to turn into an investable portfolio.'
            '</div>',
            unsafe_allow_html=True,
        )
        theme_text = st.text_input("Theme", value=ss.get("tl_theme_text", "半導体"),
                                   placeholder="e.g., Semiconductors / Gaming / Cybersecurity / GenAI ...")
        c1, c2, c3 = st.columns([2.2, 1.0, 1.6])
        region_options = ["Global","Japan","US","Europe","China"]
        with c1:
            region_mode = st.radio("Region", region_options, horizontal=True,
                                   format_func=_format_region_label,
                                   index=region_options.index(ss.get("tl_region_mode","Global")))
        with c2:
            top_n = st.number_input("Top N", min_value=1, max_value=20, value=int(ss.get("tl_top_n", 20)), step=1)
        pool = draft_pool_size(int(top_n))
        with c3:
            build_btn = st.form_submit_button(f"Build — Top {int(top_n)} (Universe {pool})", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ss["tl_theme_text"] = theme_text
    ss["tl_region_mode"] = region_mode
    ss["tl_top_n"] = int(top_n)

    inp = ThemeInput(theme_text=str(theme_text or "").strip(), region_mode=region_mode, top_n=int(top_n))
    asof = most_recent_month_end(date.today())

    # Start pipeline
    if build_btn:
        ss["tl_rows"] = None
        ss["tl_meta"] = {}
        ss["tl_universe_tickers"] = None
        ss["tl_trr_rows"] = None
        ss["tl_stage"] = "tickers"
        ss["tl_pipeline_start"] = time.time()
        ss["tl_elapsed_final"] = None

        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"status":"running", "started_at": ss["tl_pipeline_start"]}
        ss["tl_future"] = _executor().submit(job_auto_tickers, inp=inp, data_dir=data_dir, cancel_event=cancel_event)
        _rerun()

    # Elapsed metric (live or final)
    pipeline_start = ss.get("tl_pipeline_start")
    stage = ss.get("tl_stage","idle")
    if pipeline_start is not None:
        if ss.get("tl_elapsed_final") is None:
            elapsed_live = time.time() - float(pipeline_start)
            st.metric("Elapsed (s)", f"{elapsed_live:.1f}")
            if elapsed_live > 60 and stage not in ("done","idle"):
                st.warning("Over 60 seconds — it may be better to cancel and retry (or choose a narrower theme / region).")
        else:
            st.metric("Elapsed (s)", f"{float(ss['tl_elapsed_final']):.1f}")

    # Running orchestrator
    job = ss.get("tl_job")
    fut = ss.get("tl_future")
    if job and job.get("status") == "running" and fut is not None:
        if stage == "tickers":
            st.info("Stage 1/4: Building investable universe (strict count).")
        elif stage == "trr":
            st.info("Stage 2/4: Estimating TRR for the universe (best-effort, but never blank).")
        elif stage == "rank":
            st.info("Stage 3/4: Validating month-end market cap and ranking Top N.")
        elif stage == "refine":
            st.info("Stage 4/4: Adding evidence + tightening descriptions (Top N).")
        else:
            st.info("Working...")

        cA, cB, cC = st.columns([1.0, 2.0, 3.0])
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
                ss["tl_stage"] = "idle"
                ss["tl_elapsed_final"] = time.time() - float(pipeline_start or time.time())
                st.warning("Cancelled.")
                st.stop()
        with cB:
            # Universe size chip once available
            if ss.get("tl_universe_tickers"):
                st.caption(f"Universe: {len(ss['tl_universe_tickers'])} tickers")
        with cC:
            st.caption("The ranked list appears as soon as Stage 3 completes; refinement will auto-update.")

        if fut.done():
            try:
                res = fut.result()
            except Exception as e:
                ss["tl_job"] = None
                ss["tl_future"] = None
                ss["tl_stage"] = "error"
                ss["tl_elapsed_final"] = time.time() - float(pipeline_start or time.time())
                st.error(f"Job failed: {e}")
                st.stop()

            # Save last stage meta
            ss["tl_meta"] = res if isinstance(res, dict) else {"status":"error","notes":"Unexpected result type."}
            cancel_event = ss.get("tl_cancel_event") or threading.Event()

            if stage == "tickers":
                tickers = (res or {}).get("tickers", []) if isinstance(res, dict) else []
                ss["tl_universe_tickers"] = tickers
                ss["tl_stage"] = "trr"
                ss["tl_job"] = {"status":"running", "started_at": ss["tl_pipeline_start"]}
                ss["tl_future"] = _executor().submit(job_auto_trr, inp=inp, data_dir=data_dir, tickers=tickers, cancel_event=cancel_event)
                _rerun()

            elif stage == "trr":
                trr_rows = (res or {}).get("trr_rows", []) if isinstance(res, dict) else []
                ss["tl_trr_rows"] = trr_rows
                ss["tl_stage"] = "rank"
                ss["tl_job"] = {"status":"running", "started_at": ss["tl_pipeline_start"]}
                ss["tl_future"] = _executor().submit(job_validate_rank_and_summarize, inp=inp, data_dir=data_dir,
                                                     tickers=ss.get("tl_universe_tickers") or [],
                                                     trr_rows=trr_rows, cancel_event=cancel_event)
                _rerun()

            elif stage == "rank":
                if isinstance(res, dict) and res.get("status") == "ok":
                    ss["tl_rows"] = res.get("ranked")
                ss["tl_stage"] = "refine"
                ss["tl_job"] = {"status":"running", "started_at": ss["tl_pipeline_start"]}
                ss["tl_future"] = _executor().submit(job_refine, inp=inp, data_dir=data_dir, current_rows=ss.get("tl_rows") or [], cancel_event=cancel_event)
                _rerun()

            elif stage == "refine":
                if isinstance(res, dict) and res.get("status") == "ok":
                    ss["tl_rows"] = res.get("ranked", ss.get("tl_rows"))
                ss["tl_job"] = None
                ss["tl_future"] = None
                ss["tl_stage"] = "done"
                ss["tl_elapsed_final"] = time.time() - float(pipeline_start or time.time())
                _rerun()

        time.sleep(0.35)
        _rerun()

    # Deliverable (no Draft Universe display)
    rows = ss.get("tl_rows")
    meta = ss.get("tl_meta") or {}

    # Stylish definition block (always shown where the list will be)
    st.markdown(
        """
        <div class="tl-hero">
          <div class="kicker">DELIVERABLE</div>
          <div class="body">
            Ranked by <b>Theme Market Cap</b> = <b>Market Cap</b> (month-end, USD; validated if possible)
            × <b>TRR</b> (Theme Revenue Ratio).
          </div>
          <div class="fine">
            TRR is evidence-backed when possible; otherwise explicitly marked as <b>estimated</b>.
            Market cap aims for free-float adjustments where available.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rows:
        df = pd.DataFrame(rows)
        if not df.empty:
            st.subheader("Ranked List")
            chips = []
            chips.append(f'<span class="tl-chip">As-of: {str(df.get("mktcap_asof_date", pd.Series([asof.isoformat()])).iloc[0])}</span>')
            chips.append(f'<span class="tl-chip">Region: {_format_region_label(region_mode)}</span>')
            chips.append(f'<span class="tl-chip">Top N: {clamp_top_n(int(top_n))}</span>')
            st.markdown("".join(chips), unsafe_allow_html=True)

            view = df.copy()
            view["TRR"] = (pd.to_numeric(view.get("theme_revenue_ratio", 0.0), errors="coerce").fillna(0.0) * 100).round(1).astype(str) + "%"
            view["MktCap (USD)"] = pd.to_numeric(view.get("mcap_usd", 0.0), errors="coerce").apply(fmt_money)
            view["Theme MktCap (USD)"] = pd.to_numeric(view.get("theme_mktcap_usd", 0.0), errors="coerce").apply(fmt_money)

            show_cols = [
                "rank","company_name","ticker","listed_country",
                "theme_business_summary","TRR","MktCap (USD)","Theme MktCap (USD)",
                "non_theme_business_summary",
                "theme_revenue_ratio_method","theme_revenue_ratio_confidence",
                "trr_source_title","trr_source_publisher","trr_source_year","trr_source_locator",
                "mcap_quality"
            ]
            show_table_start_rank(view, show_cols, height=620)

            sid = _sha12(f"{theme_text}|{region_mode}|{asof.isoformat()}|{top_n}")
            st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name=f"themelens_ranked_list_{sid}.csv", mime="text/csv")
    else:
        st.caption("Run Build to generate the ranked list.")

    # Optional raw debug (kept but hidden)
    if isinstance(meta, dict) and meta.get("ai_raw_text"):
        with st.expander("AI raw output (debug)", expanded=False):
            st.text(str(meta.get("ai_raw_text"))[:12000])
