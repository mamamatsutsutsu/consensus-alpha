# THEMELENS — AI-first Theme Portfolio Builder (Gemini) v10
# -----------------------------------------------------------------------------
# UX goals:
# - ALPHALENS-like minimal header (no long description on the page).
# - Controls at the top (no sidebar).
# - "Quick Draft" -> show deliverable list fast (TRR is mostly estimated; sources optional).
# - "Refine" -> add evidence (sources/locators) for TRR and improve summaries; re-rank.
# - Deliverable: Ranked list by Theme Market Cap (month-end) = Free-float MktCap (proxy) × TRR.
# - Speed: disk cache + aggressive timeouts + cancel button + reduced token budgets.
#
# Notes:
# - This is a commercial/pro prototype. For production accuracy, replace yfinance with licensed data
#   (free-float market cap snapshots, segment revenue exposures, ETF holdings feeds).

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import concurrent.futures as cf
import hashlib
import json
import math
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
# Styling (do not touch padding-top; app.py owns safe top padding)
# =============================================================================
THEMELENS_CSS = """
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
  padding: 12px 12px;
}

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
TRRMethod = Literal["disclosed", "proxy", "estimated"]
TRRConfidence = Literal["High", "Med", "Low"]


@dataclass(frozen=True)
class ThemeInput:
    theme_text: str
    region_mode: RegionMode
    top_n: int


# =============================================================================
# Secrets helpers
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


# =============================================================================
# ETF seeds (behind-the-scenes; improves completeness of large-cap candidates)
# =============================================================================
_BROAD_ETF_BY_REGION: Dict[str, str] = {
    "Global": "VT",
    "US": "SPY",
    "Japan": "EWJ",
    "Europe": "VGK",
    "China": "MCHI",
}

_THEME_ETF_LIBRARY: Dict[str, List[str]] = {
    "半導体": ["SMH", "SOXX", "SOXQ"],
    "半導体装置": ["SOXX", "SMH"],
    "AI": ["AIQ", "BOTZ", "ROBO"],
    "生成AI": ["AIQ", "BOTZ"],
    "クラウド": ["SKYY", "CLOU"],
    "サイバー": ["HACK", "CIBR"],
    "サイバーセキュリティ": ["HACK", "CIBR"],
    "ロボ": ["ROBO", "BOTZ"],
    "ロボティクス": ["ROBO", "BOTZ"],
    "ゲーム": ["HERO", "ESPO"],
    "防衛": ["ITA", "XAR"],
    "宇宙": ["ARKX"],
    "EV": ["DRIV", "IDRV"],
    "クリーンエネルギー": ["ICLN", "TAN", "QCLN"],
    "フィンテック": ["FINX"],
    "バイオ": ["XBI", "IBB"],
    "ヘルスケア": ["XLV", "VHT"],
}


def suggest_theme_etfs(theme_text: str, region: RegionMode) -> List[str]:
    t = (theme_text or "").strip()
    etfs: List[str] = []
    for k, v in _THEME_ETF_LIBRARY.items():
        if k and k in t:
            etfs.extend(v)

    # region broad ETF always included
    broad = _BROAD_ETF_BY_REGION.get(region, "")
    if broad:
        etfs.insert(0, broad)

    # de-dup
    out: List[str] = []
    seen = set()
    for e in etfs:
        e2 = str(e).strip().upper()
        if e2 and e2 not in seen:
            out.append(e2); seen.add(e2)
    return out[:8]


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def _etf_holdings_cached(etf_ticker: str, top_n: int) -> List[str]:
    try:
        t = yf.Ticker(etf_ticker)
        h = getattr(t, "fund_holdings", None)
        if h is not None and hasattr(h, "head") and "symbol" in h.columns:
            syms = h["symbol"].dropna().astype(str).tolist()
            out = [s.strip().upper() for s in syms if s and s.strip()]
            return out[: int(top_n)]
    except Exception:
        pass
    return []


def build_seed_tickers(theme_text: str, region: RegionMode, max_total: int = 380) -> Tuple[List[str], List[str], List[str]]:
    etfs = suggest_theme_etfs(theme_text, region)
    holdings: List[str] = []
    used: List[str] = []
    missing: List[str] = []

    for e in etfs:
        h = _etf_holdings_cached(e, top_n=180)
        if h:
            used.append(e)
            holdings.extend(h)
        else:
            missing.append(e)

    # de-dup & cap
    out: List[str] = []
    seen = set()
    for x in holdings:
        x2 = str(x).strip().upper()
        if not x2:
            continue
        if x2 not in seen:
            out.append(x2); seen.add(x2)
        if len(out) >= max_total:
            break

    return out, used, missing


# =============================================================================
# Batch market-cap as-of month-end (proxy free-float)
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
    # Return USD per 1 unit of currency
    ccy = (ccy or "USD").upper()
    if ccy == "USD":
        return 1.0
    asof = date.fromisoformat(asof_iso)
    # Try both directions
    candidates = [f"{ccy}USD=X", f"USD{ccy}=X"]
    for sym in candidates:
        try:
            df = _yf_download_cached((sym,), start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat())
            if df is None or df.empty:
                continue
            # yfinance download for single ticker without multiindex
            if isinstance(df.columns, pd.MultiIndex):
                close = df[(sym, "Close")].dropna() if (sym, "Close") in df.columns else pd.Series(dtype=float)
            else:
                close = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
            if close.empty:
                continue
            rate = float(close.iloc[-1])
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


def _batch_infos(tickers: List[str], max_workers: int = 10) -> Dict[str, Dict[str, Any]]:
    # Parallel info fetch for speed
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
    # fetch around month-end so we can take last available close
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
        # single ticker
        if "Close" in df.columns and len(tickers) == 1:
            s = df["Close"].dropna()
            out[tickers[0]] = float(s.iloc[-1]) if not s.empty else None

    return out


def batch_mktcap_asof_usd(tickers: List[str], asof: date) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      ticker, currency, close_asof, mcap_local, mcap_usd, mcap_quality
    """
    tickers = [str(t).strip().upper() for t in tickers if t and str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame(columns=["ticker","currency","close_asof","mcap_local","mcap_usd","mcap_quality"])

    close_map = month_end_close_batch(tickers, asof)
    info_map = _batch_infos(tickers, max_workers=10)

    # FX for currencies encountered
    currencies = set()
    for t in tickers:
        ccy = str((info_map.get(t, {}) or {}).get("currency") or "USD").upper()
        currencies.add(ccy)
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
                # derive shares from marketCap/currentPrice if possible
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
            # no month-end price; fallback to marketCap
            mcap_now = info.get("marketCap")
            if mcap_now and float(mcap_now) > 0:
                mcap_local = float(mcap_now)
                quality = "proxy_field(marketCap_no_asof_price)"

        fx = fx_map.get(ccy)
        mcap_usd = (float(mcap_local) * float(fx)) if (mcap_local is not None and fx is not None) else None

        rows.append({
            "ticker": t,
            "currency": ccy,
            "close_asof": px,
            "mcap_local": mcap_local,
            "mcap_usd": mcap_usd,
            "mcap_quality": quality + ("" if fx is not None else "|fx_unavailable"),
        })

    return pd.DataFrame(rows)


def large_cap_threshold_usd(region: RegionMode) -> float:
    # practical threshold for "major index-sized" companies
    if region in ("US", "Global"):
        return 10e9
    if region == "Japan":
        return 5e9
    if region in ("Europe", "China"):
        return 8e9
    return 8e9


# =============================================================================
# Disk cache for AI outputs (fast re-runs)
# =============================================================================
def _ai_cache_dir(data_dir: str) -> Path:
    p = Path(data_dir) / "ai_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def _cache_key(prefix: str, *parts: str) -> str:
    raw = prefix + "|" + "|".join([p or "" for p in parts])
    return _sha12(raw)


# =============================================================================
# Gemini JSON call (best-effort)
# =============================================================================
def _parse_json_from_text(txt: str) -> Dict[str, Any]:
    if not txt:
        raise ValueError("Empty model response.")
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found.")
    return json.loads(m.group(0))


def _get_google_genai_client(api_key: Optional[str]):
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

    # 1) google-genai SDK
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
            "Gemini is not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY, or install google-genai for Vertex AI."
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
# AI prompts (Quick Draft / Refine)
# =============================================================================
def _ai_system_prompt() -> str:
    return (
        "You are a senior global equity portfolio manager and sector analyst. "
        "Your job is to build an investable theme portfolio for institutional clients. "
        "Be conservative: NEVER invent URLs or quotes. "
        "If you cannot provide a verified URL and exact excerpt, set url=null and excerpt='' and label method='estimated' confidence='Low'. "
        "Return ONLY valid JSON. No markdown."
    )


def build_ai_messages_quick_trr(
    theme: str,
    region: RegionMode,
    asof: date,
    tickers: List[str],
    max_additional: int = 18,
) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    tickers_str = ", ".join(tickers)
    schema = {
        "status": "ok | ambiguous | error",
        "theme_definition": "string",
        "reference_etfs": ["string"],
        "notes": "string",
        "rows": [
            {
                "ticker": "string",
                "theme_revenue_ratio": "number (0-1)",
                "theme_revenue_ratio_method": "disclosed | proxy | estimated",
                "theme_revenue_ratio_confidence": "High | Med | Low",
            }
        ],
        "additional_rows": [
            {
                "ticker": "string",
                "theme_revenue_ratio": "number (0-1)",
                "theme_revenue_ratio_method": "disclosed | proxy | estimated",
                "theme_revenue_ratio_confidence": "High | Med | Low",
            }
        ],
    }

    user_prompt = f"""
# Task (Quick Draft)
Estimate TRR (Theme Revenue Ratio) for a candidate list of large-cap stocks, and optionally add missing mega-cap theme leaders.

## Inputs
- Theme: "{theme}"
- Region: {region}
- Region definition: {region_def}
- Market-cap as-of date (for re-ranking): {asof.isoformat()}
- Candidate tickers (yfinance format): {tickers_str}

## Rules (IMPORTANT)
- TRR = theme-related revenue / total revenue (0-1).
- Use "estimated" + "Low" if you cannot support a disclosed figure.
- Focus on completeness: do NOT miss the likely top Theme Market Cap names (large caps with meaningful TRR).
- If important large-cap theme leaders are missing from the candidate list, add up to {max_additional} additional_rows tickers.

## Output (STRICT JSON ONLY)
Return ONLY JSON in this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    return [{"role": "system", "content": _ai_system_prompt()}, {"role": "user", "content": user_prompt.strip()}]


def build_ai_messages_summaries(
    theme: str,
    region: RegionMode,
    tickers: List[str],
) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    tickers_str = ", ".join(tickers)
    schema = {
        "status": "ok | error",
        "rows": [
            {
                "ticker": "string",
                "theme_business_summary_ja": "string (1-2 sentences)",
                "non_theme_business_summary_ja": "string (1 sentence)",
            }
        ],
    }
    user_prompt = f"""
# Task
Write short Japanese summaries for an institutional theme portfolio report.

## Inputs
- Theme: "{theme}"
- Region: {region}
- Region definition: {region_def}
- Tickers: {tickers_str}

## Output constraints
- Keep it concise.
- Do NOT add disclaimers or markdown.

## Output (STRICT JSON ONLY)
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    return [{"role": "system", "content": _ai_system_prompt()}, {"role": "user", "content": user_prompt.strip()}]


def build_ai_messages_refine_evidence(
    theme: str,
    region: RegionMode,
    asof: date,
    rows_payload: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    region_def = _region_definition(region)
    schema = {
        "status": "ok | error",
        "notes": "string",
        "rows": [
            {
                "ticker": "string",
                "theme_revenue_ratio": "number (0-1)",
                "theme_revenue_ratio_method": "disclosed | proxy | estimated",
                "theme_revenue_ratio_confidence": "High | Med | Low",
                "theme_revenue_sources": [
                    {
                        "source_title": "string",
                        "publisher": "string",
                        "year": "number or null",
                        "url": "string or null",
                        "locator": "string",
                        "excerpt": "string (<=25 words; exact quote ONLY if url is present)",
                    }
                ],
                "theme_business_summary_ja": "string (1-2 sentences)",
                "non_theme_business_summary_ja": "string (1 sentence)",
            }
        ],
    }

    user_prompt = f"""
# Task (Refine)
Add evidence metadata for TRR and improve Japanese summaries for the current Top-N list.
We will re-rank locally by month-end Theme Market Cap after applying refined TRR.

## Inputs
- Theme: "{theme}"
- Region: {region}
- Region definition: {region_def}
- Market-cap as-of date: {asof.isoformat()}
- Current Top-N rows (ticker + current TRR + market cap info):
{json.dumps(rows_payload, ensure_ascii=False, indent=2)}

## Evidence rules (IMPORTANT)
- NEVER invent URLs or quotes.
- If you cannot provide a verified URL + exact excerpt, set url=null and excerpt="". Use method="estimated" confidence="Low".
- Prefer annual report / 10-K / 20-F / filings. If not available, IR deck. Third-party only as last resort.
- Provide up to 2 sources per company.

## Output (STRICT JSON ONLY)
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
    return [{"role": "system", "content": _ai_system_prompt()}, {"role": "user", "content": user_prompt.strip()}]


def _normalize_trr_rows(obj: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], List[str], str]:
    status = str(obj.get("status", "error"))
    if status not in ("ok", "ambiguous", "error"):
        status = "error"
    rows = obj.get("rows", []) or []
    add = obj.get("additional_rows", []) or []
    all_rows = []
    if isinstance(rows, list):
        all_rows.extend([r for r in rows if isinstance(r, dict)])
    if isinstance(add, list):
        all_rows.extend([r for r in add if isinstance(r, dict)])

    out_rows = []
    for r in all_rows:
        t = str(r.get("ticker", "") or "").strip().upper()
        if not t:
            continue
        out_rows.append({
            "ticker": t,
            "theme_revenue_ratio": clamp01(r.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(r.get("theme_revenue_ratio_method", "estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("theme_revenue_ratio_confidence", "Low") or "Low"),
        })

    etfs = obj.get("reference_etfs", []) or []
    etfs = [str(x).strip().upper() for x in etfs if x and str(x).strip()]
    notes = str(obj.get("notes", "") or "")

    return status, out_rows, etfs[:8], notes


def _normalize_summaries(obj: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    if not isinstance(obj, dict) or str(obj.get("status")) != "ok":
        return {}
    rows = obj.get("rows", []) or []
    out: Dict[str, Dict[str, str]] = {}
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker", "") or "").strip().upper()
        if not t:
            continue
        out[t] = {
            "theme_business_summary_ja": str(r.get("theme_business_summary_ja", "") or "").strip(),
            "non_theme_business_summary_ja": str(r.get("non_theme_business_summary_ja", "") or "").strip(),
        }
    return out


def _normalize_refine(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"status": "error", "rows": []}
    status = str(obj.get("status", "error"))
    if status != "ok":
        return {"status": status, "rows": [] , "notes": str(obj.get("notes","") or "")}
    rows = obj.get("rows", []) or []
    out_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("ticker", "") or "").strip().upper()
        if not t:
            continue
        srcs = r.get("theme_revenue_sources", []) or []
        if not isinstance(srcs, list):
            srcs = []
        # take first source as table columns; keep full list for expander
        out_rows.append({
            "ticker": t,
            "theme_revenue_ratio": clamp01(r.get("theme_revenue_ratio", 0.0)),
            "theme_revenue_ratio_method": str(r.get("theme_revenue_ratio_method", "estimated") or "estimated"),
            "theme_revenue_ratio_confidence": str(r.get("theme_revenue_ratio_confidence", "Low") or "Low"),
            "theme_revenue_sources": srcs[:2],
            "theme_business_summary_ja": str(r.get("theme_business_summary_ja","") or "").strip(),
            "non_theme_business_summary_ja": str(r.get("non_theme_business_summary_ja","") or "").strip(),
        })
    return {"status": "ok", "rows": out_rows, "notes": str(obj.get("notes","") or "")}


# =============================================================================
# Async execution
# =============================================================================
@st.cache_resource
def _executor() -> cf.ThreadPoolExecutor:
    # One worker is enough; avoids flooding the API
    return cf.ThreadPoolExecutor(max_workers=1)


def _call_ai_with_cache(
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

    payload = gemini_generate_json(
        messages=messages,
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    if cancel_event.is_set():
        return {"status": "cancelled"}

    _cache_save(data_dir, cache_key, payload)
    return {"status": "ok", "cached": False, "payload": payload}


# =============================================================================
# Quick Draft engine (stepwise)
# =============================================================================
def build_deliverable_quick_draft(
    *,
    inp: ThemeInput,
    data_dir: str,
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())

    # 1) Seed tickers (ETF holdings)
    seed_tickers, used_etfs, missing_etfs = build_seed_tickers(theme, region)

    # Fallback if holdings are missing: still proceed with a small placeholder list
    if len(seed_tickers) < 20:
        # At least include mega ETF as a hint; but we still need tickers
        # In this fallback, we ask AI directly for a small candidate list with TRR (single call).
        # (This is slower, but better than returning nothing.)
        seed_tickers = seed_tickers  # keep whatever we have

    # 2) Compute market caps for seed tickers and pick top-K by market cap (to limit AI payload)
    mkt_df = batch_mktcap_asof_usd(seed_tickers, asof)
    mkt_df["mcap_usd"] = pd.to_numeric(mkt_df["mcap_usd"], errors="coerce")
    mkt_df = mkt_df.dropna(subset=["mcap_usd"])
    mkt_df = mkt_df.sort_values("mcap_usd", ascending=False)

    # Cap count for TRR estimation. Larger -> more complete, but slower.
    # We tune this to be "fast but not missing megacaps".
    k = int(min(120, max(60, 4 * top_n)))
    cand_tickers = mkt_df["ticker"].head(k).astype(str).tolist()

    # If still too few, just use whatever
    if len(cand_tickers) < max(25, 2*top_n):
        cand_tickers = list(dict.fromkeys(seed_tickers))[: max(25, 2*top_n)]

    # 3) AI TRR estimate for candidate tickers
    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    key_trr = _cache_key("trr", theme, region, asof.isoformat(), _sha12(",".join(cand_tickers)), model)
    msgs_trr = build_ai_messages_quick_trr(theme, region, asof, cand_tickers, max_additional=18)

    trr_call = _call_ai_with_cache(
        data_dir=data_dir,
        cache_key=key_trr,
        messages=msgs_trr,
        model=model,
        temperature=0.15,
        max_tokens=1800,
        timeout_s=18,
        cancel_event=cancel_event,
    )
    if trr_call.get("status") != "ok":
        return {"status": trr_call.get("status", "error"), "stage": "trr"}

    trr_raw = trr_call.get("payload", {})
    status, trr_rows, ref_etfs, notes = _normalize_trr_rows(trr_raw)
    if status == "ambiguous":
        return {
            "status": "ambiguous",
            "asof": asof.isoformat(),
            "used_etfs": used_etfs,
            "missing_etfs": missing_etfs,
            "reference_etfs": ref_etfs,
            "theme_definition": str(trr_raw.get("theme_definition","") or ""),
            "notes": notes,
        }
    if status != "ok" or not trr_rows:
        return {"status": "error", "stage": "trr", "notes": "No TRR rows returned."}

    # 4) Ensure we have market caps for all tickers (incl AI additional)
    all_tickers = list(dict.fromkeys([r["ticker"] for r in trr_rows]))
    mkt2 = batch_mktcap_asof_usd(all_tickers, asof)
    mkt2["mcap_usd"] = pd.to_numeric(mkt2["mcap_usd"], errors="coerce")

    thr = large_cap_threshold_usd(region)
    mkt2 = mkt2.dropna(subset=["mcap_usd"])
    mkt2 = mkt2[mkt2["mcap_usd"] >= thr]

    # Merge TRR & Market cap
    trr_df = pd.DataFrame(trr_rows)
    df = trr_df.merge(mkt2[["ticker","currency","mcap_usd","mcap_quality"]], on="ticker", how="inner")
    df["theme_mktcap_usd"] = df["mcap_usd"] * df["theme_revenue_ratio"]
    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df = df.head(top_n).copy()

    # 5) Add company metadata via yfinance (parallel)
    info_map = _batch_infos(df["ticker"].tolist(), max_workers=10)
    def _get(info, key, default=""):
        v = info.get(key)
        return default if v is None else str(v)

    df["company_name"] = df["ticker"].apply(lambda t: (_get(info_map.get(t,{}), "shortName", "") or _get(info_map.get(t,{}), "longName", "") or t))
    df["listed_country"] = df["ticker"].apply(lambda t: _get(info_map.get(t,{}), "country", ""))
    df["primary_exchange"] = df["ticker"].apply(lambda t: _get(info_map.get(t,{}), "exchange", ""))

    # 6) AI summaries for Top-N only (fast)
    key_sum = _cache_key("sum", theme, region, _sha12(",".join(df["ticker"].tolist())), model)
    msgs_sum = build_ai_messages_summaries(theme, region, df["ticker"].tolist())
    sum_call = _call_ai_with_cache(
        data_dir=data_dir,
        cache_key=key_sum,
        messages=msgs_sum,
        model=model,
        temperature=0.20,
        max_tokens=900,
        timeout_s=16,
        cancel_event=cancel_event,
    )
    if sum_call.get("status") == "ok":
        sum_raw = sum_call.get("payload", {})
        smap = _normalize_summaries(sum_raw)
    else:
        smap = {}

    df["theme_business_summary"] = df["ticker"].apply(lambda t: smap.get(t, {}).get("theme_business_summary_ja", "") or "")
    df["non_theme_business_summary"] = df["ticker"].apply(lambda t: smap.get(t, {}).get("non_theme_business_summary_ja", "") or "")

    # 7) Evidence columns are blank in Quick Draft (Refine fills)
    df["trr_source_title"] = ""
    df["trr_source_publisher"] = ""
    df["trr_source_year"] = None
    df["trr_source_url"] = None
    df["trr_source_locator"] = ""
    df["trr_source_excerpt"] = ""
    df["trr_sources_full"] = ""  # json string placeholder

    # 8) As-of date
    df["mktcap_asof_date"] = asof.isoformat()

    return {
        "status": "ok",
        "mode": "quick",
        "asof": asof.isoformat(),
        "used_etfs": used_etfs,
        "missing_etfs": missing_etfs,
        "reference_etfs": ref_etfs,
        "notes": notes,
        "ranked": df.to_dict(orient="records"),
        "model": model,
        "cache_keys": {"trr": key_trr, "summaries": key_sum},
        "counts": {"seed_tickers": len(seed_tickers), "cand_tickers": len(cand_tickers), "trr_rows": len(trr_rows)},
        "threshold_usd": thr,
    }


def refine_with_evidence(
    *,
    inp: ThemeInput,
    data_dir: str,
    current_rows: List[Dict[str, Any]],
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    theme = inp.theme_text.strip()
    region = inp.region_mode
    top_n = max(1, min(int(inp.top_n), 30))
    asof = most_recent_month_end(date.today())

    # payload for AI (keep it small, but informative)
    payload = []
    for r in current_rows[:top_n]:
        payload.append({
            "ticker": r.get("ticker"),
            "company_name": r.get("company_name"),
            "listed_country": r.get("listed_country"),
            "current_trr": r.get("theme_revenue_ratio"),
            "mktcap_usd": r.get("mcap_usd"),
            "theme_mktcap_usd": r.get("theme_mktcap_usd"),
        })

    model = _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    key_ref = _cache_key("refine", theme, region, asof.isoformat(), _sha12(json.dumps(payload, ensure_ascii=False)), model)
    msgs = build_ai_messages_refine_evidence(theme, region, asof, payload)

    call = _call_ai_with_cache(
        data_dir=data_dir,
        cache_key=key_ref,
        messages=msgs,
        model=model,
        temperature=0.15,
        max_tokens=2400,
        timeout_s=28,
        cancel_event=cancel_event,
    )
    if call.get("status") != "ok":
        return {"status": call.get("status","error"), "stage": "refine"}

    raw = call.get("payload", {})
    norm = _normalize_refine(raw)
    if norm.get("status") != "ok":
        return {"status": "error", "stage": "refine", "notes": norm.get("notes","")}

    # merge into current rows
    upd = {r["ticker"]: r for r in (norm.get("rows") or []) if isinstance(r, dict) and r.get("ticker")}
    df = pd.DataFrame(current_rows)
    if df.empty:
        return {"status": "error", "stage": "merge", "notes": "Current rows empty."}

    def get_upd(t, k, default=None):
        return upd.get(t, {}).get(k, default)

    df["theme_revenue_ratio"] = df["ticker"].apply(lambda t: clamp01(get_upd(t, "theme_revenue_ratio", df.loc[df["ticker"]==t, "theme_revenue_ratio"].values[0] if (df["ticker"]==t).any() else 0.0)))
    df["theme_revenue_ratio_method"] = df["ticker"].apply(lambda t: str(get_upd(t, "theme_revenue_ratio_method", df.loc[df["ticker"]==t, "theme_revenue_ratio_method"].values[0] if (df["ticker"]==t).any() else "estimated")))
    df["theme_revenue_ratio_confidence"] = df["ticker"].apply(lambda t: str(get_upd(t, "theme_revenue_ratio_confidence", df.loc[df["ticker"]==t, "theme_revenue_ratio_confidence"].values[0] if (df["ticker"]==t).any() else "Low")))

    # summaries
    df["theme_business_summary"] = df["ticker"].apply(lambda t: str(get_upd(t, "theme_business_summary_ja", df.loc[df["ticker"]==t, "theme_business_summary"].values[0] if (df["ticker"]==t).any() else "")) or "")
    df["non_theme_business_summary"] = df["ticker"].apply(lambda t: str(get_upd(t, "non_theme_business_summary_ja", df.loc[df["ticker"]==t, "non_theme_business_summary"].values[0] if (df["ticker"]==t).any() else "")) or "")

    # sources
    def first_source(t):
        srcs = get_upd(t, "theme_revenue_sources", [])
        if isinstance(srcs, list) and srcs:
            s0 = srcs[0] if isinstance(srcs[0], dict) else {}
            return s0
        return {}

    df["trr_source_title"] = df["ticker"].apply(lambda t: str(first_source(t).get("source_title","") or ""))
    df["trr_source_publisher"] = df["ticker"].apply(lambda t: str(first_source(t).get("publisher","") or ""))
    df["trr_source_year"] = df["ticker"].apply(lambda t: first_source(t).get("year", None))
    df["trr_source_url"] = df["ticker"].apply(lambda t: first_source(t).get("url", None))
    df["trr_source_locator"] = df["ticker"].apply(lambda t: str(first_source(t).get("locator","") or ""))
    df["trr_source_excerpt"] = df["ticker"].apply(lambda t: str(first_source(t).get("excerpt","") or ""))

    df["trr_sources_full"] = df["ticker"].apply(lambda t: json.dumps(get_upd(t, "theme_revenue_sources", []), ensure_ascii=False))

    # recompute theme market cap and rerank
    df["theme_mktcap_usd"] = pd.to_numeric(df.get("mcap_usd"), errors="coerce") * pd.to_numeric(df.get("theme_revenue_ratio"), errors="coerce")
    df = df.sort_values("theme_mktcap_usd", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    return {
        "status": "ok",
        "mode": "refine",
        "asof": asof.isoformat(),
        "ranked": df.to_dict(orient="records"),
        "notes": norm.get("notes",""),
        "model": model,
        "cache_keys": {"refine": key_ref},
    }


# =============================================================================
# Definitions table (kept concise)
# =============================================================================
def build_definitions_table() -> pd.DataFrame:
    items = [
        {"Field": "Theme Market Cap", "Definition": "Theme Market Cap = Free-float Market Cap (proxy, USD) × TRR"},
        {"Field": "TRR (theme_revenue_ratio)", "Definition": "TRR = テーマ関連売上 ÷ 総売上（0〜1）"},
        {"Field": "Free-float Market Cap (proxy)", "Definition": "原則: floatShares×月末Close。欠損時: sharesOutstanding×Close / marketCapを近似利用"},
        {"Field": "As-of date", "Definition": "時価総額基準日（原則：直近月末）"},
        {"Field": "Method", "Definition": "disclosed=一次情報で明確 / proxy=代理指標 / estimated=推計（推計は明示）"},
        {"Field": "Confidence", "Definition": "High/Med/Low（根拠の強さ。Lowは推計寄り）"},
        {"Field": "Sources", "Definition": "title/publisher/year/url/locator/excerpt。urlが無い場合は excerpt は空文字"},
    ]
    return pd.DataFrame(items)


# =============================================================================
# UI
# =============================================================================
def render_next_gen_tab(data_dir: str = "data") -> None:
    st.markdown(THEMELENS_CSS, unsafe_allow_html=True)

    # Minimal header (ALPHALENS-like)
    st.markdown('<div class="tl-title">THEMELENS</div>', unsafe_allow_html=True)

    # State
    ss = st.session_state
    ss.setdefault("tl_rows", None)          # current deliverable (list of dict)
    ss.setdefault("tl_meta", {})            # meta dict
    ss.setdefault("tl_job", None)           # running job info
    ss.setdefault("tl_future", None)        # future
    ss.setdefault("tl_cancel_event", None)  # cancel event

    # Controls (top, no sidebar)
    st.markdown('<div class="tl-panel">', unsafe_allow_html=True)
    with st.form("tl_controls", clear_on_submit=False):
        c1, c2, c3, c4, c5 = st.columns([2.2, 1.0, 1.0, 1.0, 1.0])
        with c1:
            theme_text = st.text_input("Theme", value=ss.get("tl_theme_text", "半導体"))
        with c2:
            region_mode = st.selectbox("Region", ["Global","Japan","US","Europe","China"],
                                       index=["Global","Japan","US","Europe","China"].index(ss.get("tl_region_mode","Global")))
        with c3:
            top_n = st.slider("Top N", 1, 30, int(ss.get("tl_top_n", 10)), 1)
        with c4:
            quick = st.form_submit_button("Quick Draft", type="primary", use_container_width=True)
        with c5:
            refine = st.form_submit_button("Refine", type="secondary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ss["tl_theme_text"] = theme_text
    ss["tl_region_mode"] = region_mode
    ss["tl_top_n"] = int(top_n)

    # If user clicked refine without draft
    if refine and not ss.get("tl_rows"):
        st.warning("Run Quick Draft first.")
        refine = False

    inp = ThemeInput(theme_text=theme_text.strip(), region_mode=region_mode, top_n=int(top_n))
    asof = most_recent_month_end(date.today())

    # Kick off jobs
    if quick:
        ss["tl_rows"] = None
        ss["tl_meta"] = {}
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "quick", "status": "running", "started_at": time.time()}
        ss["tl_future"] = _executor().submit(build_deliverable_quick_draft, inp=inp, data_dir=data_dir, cancel_event=cancel_event)
        _rerun()

    if refine:
        cancel_event = threading.Event()
        ss["tl_cancel_event"] = cancel_event
        ss["tl_job"] = {"mode": "refine", "status": "running", "started_at": time.time()}
        current_rows = ss.get("tl_rows") or []
        ss["tl_future"] = _executor().submit(refine_with_evidence, inp=inp, data_dir=data_dir, current_rows=current_rows, cancel_event=cancel_event)
        _rerun()

    # Running job UI
    job = ss.get("tl_job")
    fut = ss.get("tl_future")
    if job and job.get("status") == "running" and fut is not None:
        elapsed = time.time() - float(job.get("started_at", time.time()))
        mode = job.get("mode", "quick")

        # Short English copy (only while running)
        if mode == "quick":
            st.info(
                "Quick Draft: Gemini proposes TRR for large-cap candidates. "
                "We then re-rank by month-end Theme Market Cap = Free-float Market Cap × TRR."
            )
        else:
            st.info(
                "Refine: Adding TRR evidence (sources/locators) and improving summaries. "
                "Then re-ranking by month-end Theme Market Cap."
            )

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
            st.caption("If it takes too long: cancel → Quick Draft again (or make the theme more specific).")

        # Soft time budget warnings
        budget = 24 if mode == "quick" else 35
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

            if res.get("status") == "ok":
                ss["tl_rows"] = res.get("ranked")
                ss["tl_meta"] = res
            elif res.get("status") == "ambiguous":
                ss["tl_rows"] = None
                ss["tl_meta"] = res
            elif res.get("status") == "cancelled":
                ss["tl_rows"] = None
                ss["tl_meta"] = {"status": "cancelled"}
            else:
                ss["tl_rows"] = None
                ss["tl_meta"] = res

            _rerun()

        time.sleep(0.6)
        _rerun()

    # Output area
    meta = ss.get("tl_meta") or {}
    rows = ss.get("tl_rows")

    # Ambiguous theme
    if isinstance(meta, dict) and meta.get("status") == "ambiguous":
        st.error("Theme is ambiguous / not investable (boundary unclear).")
        if meta.get("theme_definition"):
            st.write(str(meta.get("theme_definition")))
        ref = meta.get("reference_etfs", []) or []
        used = meta.get("used_etfs", []) or []
        if used:
            st.caption("Screening ETFs used: " + ", ".join([str(x) for x in used]))
        if ref:
            st.caption("Reference ETFs: " + ", ".join([str(x) for x in ref]))
        if meta.get("notes"):
            st.caption(str(meta.get("notes")))
        return

    # Ready state (no list yet)
    if not rows:
        st.caption(f"As-of (market cap): {asof.isoformat()}  •  Click Quick Draft to generate the ranked list.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No rows returned.")
        return

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("As-of (MktCap)", str(df.get("mktcap_asof_date", pd.Series([asof.isoformat()])).iloc[0]))
    with k2:
        st.metric("Region", region_mode)
    with k3:
        st.metric("Top N", int(top_n))
    with k4:
        st.metric("Mode", str(meta.get("mode","")))

    # ETF info (small)
    used_etfs = meta.get("used_etfs", []) or []
    if used_etfs:
        st.caption("Screening ETFs used: " + ", ".join([str(x) for x in used_etfs[:6]]))

    # Deliverable list (required columns)
    st.subheader("Deliverable: Ranked List (Theme Market Cap, desc)")

    # friendly columns for display
    view = df.copy()
    view["TRR"] = (pd.to_numeric(view["theme_revenue_ratio"], errors="coerce") * 100).round(1).astype(str) + "%"
    view["Free-float MktCap (USD)"] = pd.to_numeric(view["mcap_usd"], errors="coerce").apply(fmt_money)
    view["Theme MktCap (USD)"] = pd.to_numeric(view["theme_mktcap_usd"], errors="coerce").apply(fmt_money)

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
        "theme_revenue_ratio_method",
        "theme_revenue_ratio_confidence",
        "trr_source_title",
        "trr_source_publisher",
        "trr_source_year",
        "trr_source_locator",
    ]
    existing = [c for c in show_cols if c in view.columns]
    st.dataframe(view[existing], use_container_width=True, height=560)

    # Exports
    sid = _sha12(theme_text + region_mode + asof.isoformat() + str(top_n))
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

    # Evidence details
    with st.expander("Evidence details (TRR)", expanded=False):
        for _, r in df.sort_values("rank").iterrows():
            title = f"{int(r['rank'])}. {r.get('company_name','')} ({r.get('ticker','')}) — TRR {float(r.get('theme_revenue_ratio',0))*100:.1f}% [{r.get('theme_revenue_ratio_confidence','')}]"
            with st.expander(title, expanded=False):
                st.write("**Theme business**")
                st.write(r.get("theme_business_summary","") or "-")
                st.write("**Non-theme business**")
                st.write(r.get("non_theme_business_summary","") or "-")
                st.divider()
                st.write("**TRR**")
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

    with st.expander("Definitions / Methodology", expanded=False):
        st.dataframe(build_definitions_table(), use_container_width=True, height=340)
