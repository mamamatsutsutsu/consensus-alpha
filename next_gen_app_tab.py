# NEXT GEN APP — Theme Portfolio Builder (Single-file, Pro Scaffold v2)
# -----------------------------------------------------------------------------
# Drop this file into your repo, then import from app.py:
#
#   from next_gen_app_tab import render_next_gen_tab
#   tab_next = st.tabs([...,"NEXT GEN APP"])[-1]
#   with tab_next:
#       render_next_gen_tab(data_dir="data")
#
# Implements requested pro features:
# 2) Operationalizability score (theme investability) with reasons
#    - penalizes too narrow (not enough material names) and too broad (too many)
#    - targets near user-specified N
# 3) Robustness/Sensitivity
#    - Monte Carlo TRR perturbation over candidate pool to test Top-N stability
# 4) Crowding / ETF overlap as "other theme examples"
#    - for each selected stock, show other theme labels whose ETFs also hold it (best effort)
# 5) PDF report generation (log snapshot)
#    - export a clean PDF that records inputs, meta, ranked list and evidence summary
#
# Dependencies:
#   streamlit, pandas, numpy, yfinance, reportlab
# Optional (recommended for slick charts):
#   plotly
#
# COMMERCIAL NOTES (important):
# - Free-float market cap and ETF holdings are not reliably available from Yahoo/yfinance.
#   For a commercial app, plug in a licensed dataset and keep the Evidence Ledger as your audit trail.
# - This file is structured to make provider replacement easy (Universe CSV snapshots + overrides JSON).

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import hashlib
import json
import math
import io

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting
try:
    import plotly.express as px
    _PLOTLY = True
except Exception:
    _PLOTLY = False

# Dev data provider
import yfinance as yf


# =============================================================================
# Styling (dark, pro)
# =============================================================================
DARK_CSS = """
<style>
/* Scoped styles for NEXT GEN APP — designed to blend with AlphaLens aesthetic */
.ng-hero {
  padding: 18px 22px;
  border-radius: 18px;
  background: rgba(0,0,0,0.55);
  border: 1px solid rgba(0,242,254,0.18);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  backdrop-filter: blur(10px);
}

.ng-title {
  font-family: Orbitron, ui-sans-serif, system-ui, -apple-system, "Segoe UI", "Helvetica Neue", Arial;
  font-size: 26px;
  font-weight: 900;
  letter-spacing: 0.12em;
  margin: 0;
  color: #00f2fe;
}

.ng-subtitle {
  margin: 10px 0 0 0;
  color: rgba(255,255,255,0.78);
  font-size: 13px;
}

.ng-card {
  padding: 14px 14px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(0,0,0,0.35);
}

.ng-kpi {
  font-size: 12px;
  color: rgba(255,255,255,0.70);
  margin-bottom: 6px;
}

.ng-kpi-val {
  font-size: 20px;
  font-weight: 900;
}

.ng-muted { color: rgba(255,255,255,0.68); }

.ng-tag {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 10px;
  border: 1px solid rgba(0,242,254,0.18);
  background: rgba(0,242,254,0.06);
  margin-right: 6px;
  margin-top: 6px;
  font-size: 12px;
  color: rgba(255,255,255,0.86);
}
</style>
"""


# =============================================================================
# Types / Models
# =============================================================================
Confidence = Literal["High", "Med", "Low"]
Method = Literal["disclosed", "estimated", "proxy"]
RegionMode = Literal["Global", "Japan", "US", "Europe", "China"]
DataRigorMode = Literal["Strict", "Balanced", "Expand"]
WeightingScheme = Literal["theme_mktcap", "sqrt_theme_mktcap", "confidence_adj_theme_mktcap", "equal", "inv_vol"]


@dataclass(frozen=True)
class ThemeInput:
    theme_text: str
    region_mode: RegionMode = "Global"
    n_max: int = 10
    data_rigor: DataRigorMode = "Balanced"
    min_confidence: Confidence = "Low"
    weighting: WeightingScheme = "theme_mktcap"
    etf_override: Optional[List[str]] = None


@dataclass(frozen=True)
class UniverseRule:
    name: str
    description: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class UniverseConstituent:
    ticker: str
    exchange: str
    company: str = ""
    listed_country: str = ""
    sector: str = ""
    currency: str = ""

    # optional snapshot data (recommended for speed)
    asof_date: Optional[str] = None
    free_float_mktcap_local: Optional[float] = None
    free_float_mktcap_usd: Optional[float] = None
    liquidity_usd: Optional[float] = None
    index_tags: Optional[str] = None


@dataclass(frozen=True)
class EvidenceItem:
    source_name: str
    source_year: Optional[int]
    locator: str
    excerpt: str
    confidence: Confidence
    method: Method
    notes: str = ""


@dataclass(frozen=True)
class ThemeExposure:
    trr: float
    tpr: Optional[float]
    theme_business_summary: str
    non_theme_business_summary: str
    trr_evidence: EvidenceItem
    tpr_evidence: Optional[EvidenceItem] = None


@dataclass
class SecurityRow:
    rank: int
    ticker: str
    exchange: str
    company: str
    listed_country: str
    sector: str

    currency: str
    mktcap_asof_date: str
    free_float_mktcap_local: float
    free_float_mktcap_usd: Optional[float]
    mktcap_quality: str

    trr: float
    trr_confidence: Confidence
    trr_method: Method
    trr_source: str
    trr_locator: str
    trr_excerpt: str

    tpr: Optional[float]
    tpr_confidence: Optional[Confidence]
    tpr_method: Optional[Method]
    tpr_source: Optional[str]
    tpr_locator: Optional[str]

    theme_mktcap_r_usd: Optional[float]
    theme_mktcap_p_usd: Optional[float]

    theme_business_summary: str
    non_theme_business_summary: str

    # crowding (other theme examples via ETF overlap)
    other_theme_examples: str = ""

    # price analytics
    price_series_info: str = ""
    ret_3y: Optional[float] = None
    cagr_3y: Optional[float] = None
    vol_3y: Optional[float] = None
    maxdd_3y: Optional[float] = None
    beta_3y: Optional[float] = None
    te_3y: Optional[float] = None
    ir_3y: Optional[float] = None
    mom_12m: Optional[float] = None
    mom_3m: Optional[float] = None

    confidence_factor: float = 0.0


# =============================================================================
# Helper analytics
# =============================================================================
def most_recent_month_end(today: date) -> date:
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = next_month - timedelta(days=1)
    if today >= month_end:
        return month_end
    return today.replace(day=1) - timedelta(days=1)


def clamp01(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def confidence_factor(conf: Confidence) -> float:
    return {"High": 1.0, "Med": 0.7, "Low": 0.4}.get(conf, 0.4)


def compute_return_metrics(price: pd.Series) -> Dict[str, Optional[float]]:
    if price is None or len(price) < 50:
        return {"ret": None, "cagr": None, "vol": None, "maxdd": None}
    px = price.dropna()
    if len(px) < 50:
        return {"ret": None, "cagr": None, "vol": None, "maxdd": None}
    start = float(px.iloc[0]); end = float(px.iloc[-1])
    if start <= 0 or end <= 0:
        return {"ret": None, "cagr": None, "vol": None, "maxdd": None}

    ret = end / start - 1.0
    days = (px.index[-1] - px.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (end / start) ** (1.0 / years) - 1.0

    daily = px.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(252.0)) if len(daily) > 10 else None

    cum = (1.0 + daily).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    maxdd = float(dd.min()) if len(dd) else None

    return {"ret": float(ret), "cagr": float(cagr), "vol": vol, "maxdd": maxdd}


def beta_te_ir(port_ret: pd.Series, bench_ret: pd.Series) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    df = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 50:
        return None, None, None
    p = df["p"]; b = df["b"]
    var_b = float(b.var())
    beta = float(p.cov(b) / var_b) if var_b > 0 else None
    active = p - b
    te = float(active.std() * np.sqrt(252.0)) if active.std() > 0 else None
    ir = float(active.mean() * 252.0 / te) if te and te > 0 else None
    return beta, te, ir


def momentum(price: pd.Series, lookback_days: int) -> Optional[float]:
    if price is None or price.dropna().shape[0] < lookback_days + 5:
        return None
    px = price.dropna()
    end = float(px.iloc[-1])
    start = float(px.iloc[-(lookback_days + 1)])
    if start <= 0:
        return None
    return float(end / start - 1.0)


def align_price_matrix(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame({k: v for k, v in prices.items() if v is not None and not v.empty})
    if df.empty:
        return df
    return df.sort_index().ffill().dropna(how="all")


def daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    return price_df.pct_change().dropna(how="all")


def cap_weight(values: pd.Series, power: float = 1.0) -> pd.Series:
    v = values.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return pd.Series(dtype=float)
    v = np.maximum(v, 0.0) ** power
    if v.sum() <= 0:
        return pd.Series(dtype=float)
    return v / v.sum()


def inv_vol_weights(ret_df: pd.DataFrame) -> pd.Series:
    if ret_df is None or ret_df.empty:
        return pd.Series(dtype=float)
    vol = ret_df.std() * np.sqrt(252.0)
    vol = vol.replace(0, np.nan).dropna()
    if vol.empty:
        return pd.Series(dtype=float)
    w = 1.0 / vol
    return w / w.sum()


def portfolio_returns(ret_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if ret_df is None or ret_df.empty or weights is None or weights.empty:
        return pd.Series(dtype=float)
    cols = [c for c in ret_df.columns if c in weights.index]
    if not cols:
        return pd.Series(dtype=float)
    w = weights[cols].fillna(0.0)
    w = w / w.sum() if w.sum() != 0 else w
    return (ret_df[cols] * w).sum(axis=1)


def hhi(weights: pd.Series) -> Optional[float]:
    if weights is None or weights.empty:
        return None
    w = weights.fillna(0.0).values
    return float((w ** 2).sum())


def top_n_concentration(weights: pd.Series, n: int = 5) -> Optional[float]:
    if weights is None or weights.empty:
        return None
    return float(weights.sort_values(ascending=False).head(n).sum())


# =============================================================================
# Theme ambiguity + operationalizability
# =============================================================================
AMBIGUOUS_PATTERNS = ["サステナ", "未来", "次世代", "DX", "ソリューション", "イノベーション", "スマート", "成長", "ESG"]

def is_ambiguous_theme(theme_text: str, etfs: List[str]) -> bool:
    t = (theme_text or "").strip()
    if len(t) <= 2:
        return True
    if any(p in t for p in AMBIGUOUS_PATTERNS):
        return True
    # If no ETF proxy found and theme is very short/broad, reject.
    if not etfs and len(t) <= 4:
        concrete_allow = ["半導体", "ゲーム", "防衛", "宇宙", "水素", "原子力", "銀行", "保険", "自動車", "医薬"]
        if not any(c in t for c in concrete_allow):
            return True
    return False


def effective_name_count(theme_scores: np.ndarray) -> float:
    """Effective number of material names using inverse HHI of theme-score weights."""
    x = np.array(theme_scores, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    w = x / x.sum()
    h = float((w ** 2).sum())
    return float(1.0 / h) if h > 0 else 0.0


def operationalizability_score_v2(
    *,
    target_n: int,
    raw_candidate_count: int,
    theme_scores: np.ndarray,
    avg_conf: float,
    etf_count: int,
) -> Dict[str, Any]:
    """Score 0..100. High when 'effective candidate breadth' is near target N."""
    N = max(1, int(target_n))
    C_raw = max(0, int(raw_candidate_count))
    enc = effective_name_count(theme_scores)

    # Base fit around N using log-ratio gaussian
    # enc ~ N -> ~100, enc << N or enc >> N -> low
    if enc <= 0:
        base = 0.0
    else:
        ratio = enc / N
        sigma = 0.80
        penalty = abs(math.log(ratio))
        # shortage is slightly harsher (can't fill N with material exposure)
        if enc < N:
            penalty *= 1.15
        base = 100.0 * math.exp(-(penalty ** 2) / (2.0 * sigma ** 2))

    # If raw pool can't even supply N names, apply additional penalty
    if C_raw < N:
        base *= max(0.15, C_raw / N)  # quickly pushes down if not enough names at all

    # Evidence adjustment (small, but real)
    avg_conf = float(max(0.0, min(avg_conf, 1.0)))
    score = base * (0.85 + 0.15 * avg_conf)

    # ETF proxy bonus (tiny; avoids punishing legitimate but sparse ETF coverage too much)
    score += min(5.0, float(etf_count))

    score_i = int(max(0, min(100, round(score))))

    # Reasoning
    ratio_msg = "-"
    if enc > 0:
        ratio_msg = f"{enc:.1f} (effective) vs N={N}"
    fit = "balanced"
    if enc < max(2.0, 0.6 * N):
        fit = "too_narrow"
    elif enc > 2.5 * N:
        fit = "too_broad"

    reasons: List[str] = []
    reasons.append(f"Target N = {N}")
    reasons.append(f"Raw candidate pool (post filters) = {C_raw}")
    reasons.append(f"Effective name count (material exposure breadth) = {enc:.1f}")
    reasons.append(f"Avg evidence confidence factor = {avg_conf:.2f} (1.00=High mix, 0.40~Low mix)")
    if etf_count <= 0:
        reasons.append("ETF proxies: none (screening confidence may be lower)")
    else:
        reasons.append(f"ETF proxies used: {etf_count} tickers")

    if fit == "too_narrow":
        headline = "テーマがニッチで、投資対象が狭い（materialな銘柄が少ない）"
        suggestions = [
            "テーマを少し広げる（例：サブテーマを統合）",
            "地域をGlobalにする / data_rigorをBalancedにする",
            "関連バリューチェーン（装置/素材/インフラ等）を含める",
        ]
    elif fit == "too_broad":
        headline = "テーマが広すぎ、投資対象が多すぎる（テーマ境界が広い）"
        suggestions = [
            "テーマを具体化（例：AI→AIインフラ / AIソフト / エッジAIなど）",
            "除外条件を追加（例：純度TRR>=x%）",
            "地域を絞る（Japan/US/Europe/China）",
        ]
    else:
        headline = "候補の広さが指定Nに近く、テーマ運用として扱いやすい"
        suggestions = [
            "このまま運用可能。TRR/TPRの開示ソースを増やすと監査耐性が上がります。",
        ]

    detail = {
        "score": score_i,
        "fit": fit,
        "headline": headline,
        "reasons": reasons,
        "suggestions": suggestions,
        "enc": enc,
        "raw_count": C_raw,
        "target_n": N,
        "base_score": float(base),
    }
    return detail


# =============================================================================
# Providers (dev / scaffold)
# =============================================================================
class UniverseProviderCSV:
    """Fast, reproducible universe source for commercial use."""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _file(self, region: RegionMode) -> Path:
        mapping = {
            "Global": "universe_global.csv",
            "US": "universe_us.csv",
            "Japan": "universe_japan.csv",
            "Europe": "universe_europe.csv",
            "China": "universe_china.csv",
        }
        return self.data_dir / mapping.get(region, "universe_global.csv")

    def get_universe(self, region: RegionMode, asof: date) -> Tuple[List[UniverseConstituent], UniverseRule]:
        path = self._file(region)
        if not path.exists():
            fallback = [
                UniverseConstituent("AAPL","NASDAQ","Apple","US","Technology","USD"),
                UniverseConstituent("MSFT","NASDAQ","Microsoft","US","Technology","USD"),
                UniverseConstituent("NVDA","NASDAQ","NVIDIA","US","Technology","USD"),
                UniverseConstituent("0700.HK","HKEX","Tencent","HK","Communication Services","HKD"),
                UniverseConstituent("ASML","NASDAQ","ASML","NL","Technology","USD"),
            ]
            rule = UniverseRule(
                name=f"FallbackStaticUniverse({region})",
                description="Universe CSV not found; using minimal fallback list. Provide universe_*.csv snapshots for production.",
                params={"expected_path": str(path)},
            )
            return fallback, rule

        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}

        def getv(r, name, default=""):
            c = cols.get(name)
            if c is None:
                return default
            v = r.get(c)
            return default if pd.isna(v) else v

        out: List[UniverseConstituent] = []
        for _, r in df.iterrows():
            out.append(
                UniverseConstituent(
                    ticker=str(getv(r,"ticker")).strip(),
                    exchange=str(getv(r,"exchange")).strip(),
                    company=str(getv(r,"company")).strip(),
                    listed_country=str(getv(r,"listed_country")).strip(),
                    sector=str(getv(r,"sector")).strip() if "sector" in cols else "",
                    currency=str(getv(r,"currency")).strip() if "currency" in cols else "",
                    asof_date=str(getv(r,"asof_date")).strip() if "asof_date" in cols else None,
                    free_float_mktcap_local=float(getv(r,"free_float_mktcap_local", np.nan)) if "free_float_mktcap_local" in cols and pd.notna(getv(r,"free_float_mktcap_local", np.nan)) else None,
                    free_float_mktcap_usd=float(getv(r,"free_float_mktcap_usd", np.nan)) if "free_float_mktcap_usd" in cols and pd.notna(getv(r,"free_float_mktcap_usd", np.nan)) else None,
                    liquidity_usd=float(getv(r,"liquidity_usd", np.nan)) if "liquidity_usd" in cols and pd.notna(getv(r,"liquidity_usd", np.nan)) else None,
                    index_tags=str(getv(r,"index_tags")).strip() if "index_tags" in cols else None,
                )
            )

        rule = UniverseRule(
            name=f"UniverseCSV({region})",
            description="Large-cap universe loaded from CSV snapshot (recommended for commercial speed/reproducibility).",
            params={"path": str(path), "rows": len(out)},
        )
        return out, rule


class ETFProviderHeuristic:
    """ETF ticker suggestions + best-effort holdings (for prototype)."""
    def __init__(self):
        self._holdings_cache: Dict[str, List[str]] = {}

    # Theme library used for "other theme examples" (crowding)
    THEME_LIBRARY: Dict[str, List[str]] = {
        "半導体": ["SMH", "SOXX", "SOXQ"],
        "AI": ["AIQ", "BOTZ", "ROBO"],
        "クラウド": ["SKYY", "CLOU"],
        "サイバーセキュリティ": ["HACK", "CIBR"],
        "ロボティクス": ["ROBO", "BOTZ"],
        "ゲーム": ["HERO", "ESPO"],
        "防衛": ["ITA", "XAR"],
        "宇宙": ["ARKX"],
        "EV": ["DRIV", "IDRV"],
        "クリーンエネルギー": ["ICLN", "TAN", "QCLN"],
        "フィンテック": ["FINX"],
        "ヘルスケア": ["XLV", "VHT"],
        "中国": ["MCHI", "FXI", "ASHR"],
        "欧州": ["VGK", "IEUR"],
        "日本": ["EWJ"],
        "米国": ["SPY", "IVV", "QQQ"],
    }

    def suggest(self, theme: str, region: RegionMode) -> List[str]:
        t = (theme or "").strip()
        etfs: List[str] = []
        for k, v in self.THEME_LIBRARY.items():
            if k in t:
                etfs.extend(v)
        # region additions
        if region == "China":
            etfs.extend(self.THEME_LIBRARY.get("中国", []))
        if region == "Japan":
            etfs.extend(self.THEME_LIBRARY.get("日本", []))
        if region == "US":
            etfs.extend(self.THEME_LIBRARY.get("米国", []))
        if region == "Europe":
            etfs.extend(self.THEME_LIBRARY.get("欧州", []))

        # unique keep order
        seen=set(); out=[]
        for e in etfs:
            if e and e not in seen:
                out.append(e); seen.add(e)
        return out[:8]

    def get_profile(self, etf_ticker: str) -> Dict[str, str]:
        try:
            info = yf.Ticker(etf_ticker).info or {}
            return {
                "name": str(info.get("shortName") or info.get("longName") or etf_ticker),
                "category": str(info.get("category") or ""),
                "family": str(info.get("fundFamily") or ""),
            }
        except Exception:
            return {"name": etf_ticker, "category": "", "family": ""}

    def get_top_holdings(self, etf_ticker: str, top_n: int = 80) -> List[str]:
        # Cache first (important for crowding scan)
        key = f"{etf_ticker}:{top_n}"
        if key in self._holdings_cache:
            return self._holdings_cache[key]

        holdings: List[str] = []
        try:
            t = yf.Ticker(etf_ticker)
            # yfinance holdings are inconsistent, may be absent
            h = getattr(t, "fund_holdings", None)
            if h is not None and hasattr(h, "head") and "symbol" in h.columns:
                syms = h["symbol"].dropna().astype(str).tolist()
                holdings = [s.strip().upper() for s in syms][:top_n]
        except Exception:
            holdings = []

        self._holdings_cache[key] = holdings
        return holdings


class MarketProviderYF:
    """Market data dev provider."""
    def __init__(self):
        self._info_cache: Dict[str, Dict] = {}

    def info(self, ticker: str) -> Dict:
        if ticker in self._info_cache:
            return self._info_cache[ticker]
        try:
            d = yf.Ticker(ticker).info or {}
        except Exception:
            d = {}
        self._info_cache[ticker] = d
        return d

    def fx_usd_per_ccy(self, ccy: str, asof: date) -> Tuple[Optional[float], str]:
        ccy = (ccy or "USD").upper()
        if ccy == "USD":
            return 1.0, "exact"

        candidates = [f"{ccy}USD=X", f"USD{ccy}=X"]
        for sym in candidates:
            try:
                df = yf.download(sym, start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat(),
                                 progress=False, auto_adjust=False)
                if df is None or df.empty:
                    continue
                close = df["Close"].dropna()
                if close.empty:
                    continue
                rate = float(close.iloc[-1])
                if rate <= 0 or np.isnan(rate):
                    continue
                usd_per_ccy = rate
                if sym.startswith("USD") or rate > 20:
                    usd_per_ccy = 1.0 / rate
                if 0 < usd_per_ccy < 100:
                    return usd_per_ccy, f"best_effort({sym})"
            except Exception:
                continue
        return None, "unavailable"

    def free_float_mktcap_asof(self, ticker: str, asof: date) -> Tuple[Optional[float], str, Optional[float], str]:
        info = self.info(ticker)
        currency = str(info.get("currency") or "USD").upper()

        px = None
        try:
            hist = yf.download(ticker, start=(asof - timedelta(days=10)).isoformat(), end=(asof + timedelta(days=1)).isoformat(),
                               progress=False, auto_adjust=False)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = hist["Close"].dropna()
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
            return mcap_local, currency, mcap_usd, f"proxy_free_float(floatShares×Close),{q}"

        if shares_out and shares_out > 0:
            mcap_local = float(shares_out) * px
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_total(sharesOutstanding×Close),{q}"

        mcap = info.get("marketCap")
        if mcap and mcap > 0:
            mcap_local = float(mcap)
            usd_per_ccy, q = self.fx_usd_per_ccy(currency, asof)
            mcap_usd = (mcap_local * usd_per_ccy) if usd_per_ccy else None
            return mcap_local, currency, mcap_usd, f"proxy_field(marketCap),{q}"

        return None, currency, None, "missing_shares"

    def price_batch(self, tickers: List[str], start: date, end: date, auto_adjust: bool = True) -> Dict[str, pd.Series]:
        if not tickers:
            return {}
        data = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            group_by="ticker",
            auto_adjust=auto_adjust,
            progress=False,
            threads=True,
        )
        out: Dict[str, pd.Series] = {}
        if data is None or data.empty:
            return out
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if (t, "Close") in data.columns:
                    s = data[(t, "Close")].dropna()
                    s.index = pd.to_datetime(s.index)
                    out[t] = s
        else:
            if "Close" in data.columns:
                s = data["Close"].dropna()
                s.index = pd.to_datetime(s.index)
                out[tickers[0]] = s
        return out

    def profile(self, ticker: str) -> Dict[str, str]:
        info = self.info(ticker)
        return {
            "company": str(info.get("shortName") or info.get("longName") or ticker),
            "listed_country": str(info.get("country") or ""),
            "sector": str(info.get("sector") or ""),
            "currency": str(info.get("currency") or "USD"),
        }


class ExposureEstimatorOverrides:
    """Evidence-first TRR/TPR using JSON overrides."""
    def __init__(self, data_dir: str):
        self.path = Path(data_dir) / "exposure_overrides.json"
        self._cache: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _default_ev(self, method: Method="estimated") -> EvidenceItem:
        return EvidenceItem(
            source_name="ESTIMATED (no explicit disclosure found)",
            source_year=None,
            locator="N/A",
            excerpt="Estimated exposure. For commercial use: attach filings/IR evidence with locator + short excerpt.",
            confidence="Low",
            method=method,
            notes="Heuristic mode. Replace with evidence overrides.",
        )

    def estimate(self, theme: str, sec: UniverseConstituent) -> ThemeExposure:
        theme = (theme or "").strip()
        t = sec.ticker.upper().strip()
        entry = (self._cache.get(theme, {}) or {}).get(t)

        if entry:
            trr = clamp01(float(entry.get("trr", 0.0)))
            tpr = entry.get("tpr", None)
            tpr = clamp01(float(tpr)) if tpr is not None else None

            def parse_ev(obj) -> EvidenceItem:
                if not obj:
                    return self._default_ev("estimated")
                return EvidenceItem(
                    source_name=str(obj.get("source_name","")),
                    source_year=obj.get("source_year", None),
                    locator=str(obj.get("locator","")),
                    excerpt=str(obj.get("excerpt","")),
                    confidence=obj.get("confidence","Low"),
                    method=obj.get("method","estimated"),
                    notes=str(obj.get("notes","")),
                )

            trr_ev = parse_ev(entry.get("trr_evidence"))
            tpr_ev = parse_ev(entry.get("tpr_evidence")) if entry.get("tpr_evidence") else None

            return ThemeExposure(
                trr=trr,
                tpr=tpr,
                theme_business_summary=str(entry.get("theme_business_summary","")),
                non_theme_business_summary=str(entry.get("non_theme_business_summary","")),
                trr_evidence=trr_ev,
                tpr_evidence=tpr_ev,
            )

        # Heuristic fallback (explicitly Low confidence)
        theme_l = theme.lower()
        sector_l = (sec.sector or "").lower()
        trr_guess = 0.03

        if ("半導体" in theme) or ("semi" in theme_l):
            if ("semiconductor" in sector_l) or ("technology" in sector_l):
                trr_guess = 0.25
        if ("ゲーム" in theme) or ("gaming" in theme_l):
            if ("communication" in sector_l) or ("consumer" in sector_l):
                trr_guess = 0.15
        if ("防衛" in theme) or ("defense" in theme_l):
            trr_guess = 0.12
        if ("クラウド" in theme) or ("cloud" in theme_l):
            if ("technology" in sector_l):
                trr_guess = 0.15

        trr_guess = clamp01(trr_guess)

        return ThemeExposure(
            trr=trr_guess,
            tpr=None,
            theme_business_summary=f"Estimated exposure to '{theme}' (sector-based heuristic). Replace with evidence.",
            non_theme_business_summary="Not available (heuristic). Replace with evidence.",
            trr_evidence=self._default_ev("estimated"),
            tpr_evidence=None,
        )

    def save_override(self, theme: str, ticker: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
        theme = (theme or "").strip()
        ticker = (ticker or "").strip().upper()
        if not theme or not ticker:
            return False, "theme/ticker required"
        obj = self._cache
        obj.setdefault(theme, {})
        obj[theme][ticker] = payload
        try:
            self.path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            self._cache = obj
            return True, f"Saved override: {theme} / {ticker}"
        except Exception as e:
            return False, f"Failed to save: {e}"


# =============================================================================
# Robustness simulation (candidate pool)
# =============================================================================
def robustness_simulation(
    pool: pd.DataFrame,
    top_n: int,
    iters: int = 200,
    seed: int = 7,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Monte Carlo TRR perturbation; returns inclusion frequency and summary."""
    if pool is None or pool.empty:
        return pd.DataFrame(), {"status": "no_data"}

    n = max(1, int(top_n))
    iters = int(max(20, min(iters, 2000)))
    rng = np.random.default_rng(seed)

    # Required columns: ticker, trr, trr_confidence, mcap_base
    df = pool.copy()
    for c in ["trr", "mcap_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["trr", "mcap_base"])
    df["trr"] = df["trr"].clip(0.0, 1.0)
    df["mcap_base"] = df["mcap_base"].clip(lower=0.0)

    if df.empty:
        return pd.DataFrame(), {"status": "no_valid_rows"}

    conf = df["trr_confidence"].astype(str).values
    sigma = np.array([0.05 if c=="High" else 0.10 if c=="Med" else 0.20 for c in conf], dtype=float)

    tickers = df["ticker"].astype(str).values
    base_rank = df["base_rank"].astype(int).values if "base_rank" in df.columns else np.arange(len(df)) + 1
    base_top = set(tickers[np.argsort(base_rank)[:n]])  # baseline topN by base_rank

    counts = {t: 0 for t in tickers}
    for _ in range(iters):
        eps = rng.normal(0.0, sigma)
        trr2 = np.clip(df["trr"].values * (1.0 + eps), 0.0, 1.0)
        score = df["mcap_base"].values * trr2
        idx = np.argsort(score)[::-1][:n]
        for t in tickers[idx]:
            counts[t] += 1

    freq = pd.DataFrame({
        "ticker": list(counts.keys()),
        "topN_inclusion_rate": [counts[t]/iters for t in counts.keys()],
    }).sort_values("topN_inclusion_rate", ascending=False)

    # Summary on baseline topN stability
    base_freq = freq.set_index("ticker").loc[list(base_top), "topN_inclusion_rate"] if base_top else pd.Series(dtype=float)
    summary = {
        "status": "ok",
        "iters": iters,
        "top_n": n,
        "baseline_topN_mean_inclusion": float(base_freq.mean()) if not base_freq.empty else None,
        "baseline_topN_min_inclusion": float(base_freq.min()) if not base_freq.empty else None,
        "unstable_names_in_baseline_topN": int((base_freq < 0.60).sum()) if not base_freq.empty else None,
    }
    return freq, summary


# =============================================================================
# Crowding / ETF overlap as other theme examples
# =============================================================================
def compute_other_theme_examples(
    *,
    tickers: List[str],
    theme_text: str,
    etf_provider: ETFProviderHeuristic,
    max_themes_per_ticker: int = 5,
) -> Dict[str, str]:
    """Return ticker->string of other theme examples based on ETF overlap (best effort)."""
    tset = set([t.upper().strip() for t in tickers if t])
    if not tset:
        return {}

    theme_text = (theme_text or "").strip()
    out: Dict[str, List[str]] = {t: [] for t in tset}

    # Iterate library themes; fetch ETF holdings and check membership
    for theme_label, etfs in etf_provider.THEME_LIBRARY.items():
        # skip themes that look like the input theme (rough heuristic)
        if theme_label and (theme_label in theme_text or theme_text in theme_label):
            continue

        matched_tickers: set = set()
        used_etfs: List[str] = []
        for etf in etfs[:2]:  # keep it cheap
            holdings = etf_provider.get_top_holdings(etf, top_n=120)
            if holdings:
                used_etfs.append(etf)
                matched_tickers |= (tset & set(holdings))

        if not matched_tickers:
            continue

        label_str = theme_label
        # optionally append ETF codes for transparency
        if used_etfs:
            label_str = f"{theme_label}({','.join(used_etfs)})"

        for t in matched_tickers:
            out[t].append(label_str)

    # format
    fmt: Dict[str, str] = {}
    for t, labels in out.items():
        labels = list(dict.fromkeys(labels))  # unique preserve order
        if not labels:
            fmt[t] = ""
        else:
            fmt[t] = ", ".join(labels[:max_themes_per_ticker])
    return fmt


# =============================================================================
# PDF report generation (log snapshot)
# =============================================================================
def build_pdf_report(payload: Dict[str, Any]) -> bytes:
    """Create a PDF report from the snapshot payload. Returns PDF bytes."""
    # Lazy import to keep app startup fast
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # Register a Japanese-capable font if available in ReportLab's CID fonts
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
        base_font = "HeiseiKakuGo-W5"
    except Exception:
        base_font = "Helvetica"

    styles = getSampleStyleSheet()
    styleN = ParagraphStyle("N", parent=styles["Normal"], fontName=base_font, fontSize=10, leading=13)
    styleH = ParagraphStyle("H", parent=styles["Heading1"], fontName=base_font, fontSize=16, leading=20, spaceAfter=8)
    styleH2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=base_font, fontSize=12, leading=16, spaceAfter=6)
    styleS = ParagraphStyle("S", parent=styles["Normal"], fontName=base_font, fontSize=8, leading=11, textColor=colors.grey)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14*mm,
        rightMargin=14*mm,
        topMargin=14*mm,
        bottomMargin=14*mm,
        title="NEXT GEN APP Theme Portfolio Snapshot",
    )

    meta = payload.get("meta", {}) or {}
    uni_rule = payload.get("universe_rule", {}) or {}
    op = meta.get("operationalizability", {}) or payload.get("operationalizability", {}) or {}
    rows = payload.get("rows", []) or []
    portfolio = payload.get("portfolio", {}) or {}
    etfs = payload.get("etfs_used", []) or []
    sid = payload.get("snapshot_id", "")

    elements: List[Any] = []
    elements.append(Paragraph("NEXT GEN APP - Theme Portfolio Snapshot", styleH))
    elements.append(Paragraph(f"Snapshot ID: {sid}", styleS))
    elements.append(Paragraph(f"Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styleS))
    elements.append(Spacer(1, 6))

    # Inputs/meta table
    kv = [
        ["Theme", str(meta.get("theme",""))],
        ["Region", str(meta.get("region_mode",""))],
        ["Top N", str(meta.get("n_max",""))],
        ["MktCap As-of", str(meta.get("mktcap_asof_date",""))],
        ["Benchmark", str(meta.get("benchmark",""))],
        ["Price End", str(meta.get("price_end_date",""))],
        ["Data Rigor", str(meta.get("data_rigor",""))],
        ["Min Confidence", str(meta.get("min_confidence",""))],
        ["Weighting", str(meta.get("weighting",""))],
    ]
    t = Table(kv, colWidths=[30*mm, 140*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), base_font),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))

    # Universe + ETFs
    elements.append(Paragraph("Universe & Screening", styleH2))
    elements.append(Paragraph(f"Universe rule: {uni_rule.get('name','')}", styleN))
    elements.append(Paragraph(str(uni_rule.get("description","")), styleS))
    elements.append(Paragraph(f"ETFs used (screening): {', '.join(etfs)}", styleN))
    elements.append(Spacer(1, 8))

    # Operationalizability
    if op:
        elements.append(Paragraph("Operationalizability (Investability) Score", styleH2))
        elements.append(Paragraph(f"Score: {op.get('score','-')} / 100", styleN))
        elements.append(Paragraph(str(op.get("headline","")), styleN))
        for r in (op.get("reasons", []) or [])[:8]:
            elements.append(Paragraph(f"- {r}", styleN))
        for s in (op.get("suggestions", []) or [])[:5]:
            elements.append(Paragraph(f"* {s}", styleS))
        elements.append(Spacer(1, 8))

    # Portfolio summary
    if portfolio:
        elements.append(Paragraph("Portfolio Summary (3Y)", styleH2))
        ps = [
            ["3Y Return", fmt_pct(portfolio.get("portfolio_return_3y"))],
            ["3Y CAGR", fmt_pct(portfolio.get("portfolio_cagr_3y"))],
            ["Vol", fmt_pct(portfolio.get("portfolio_vol_3y"))],
            ["MaxDD", fmt_pct(portfolio.get("portfolio_maxdd_3y"))],
            ["Beta", fmt_float(portfolio.get("beta_3y"))],
            ["Tracking Error", fmt_pct(portfolio.get("tracking_error_3y"))],
            ["Information Ratio", fmt_float(portfolio.get("information_ratio_3y"))],
            ["Top5 weight", fmt_pct(portfolio.get("top5_weight"))],
            ["HHI", fmt_float(portfolio.get("hhi"))],
            ["Wgt Avg TRR", fmt_pct(portfolio.get("wavg_trr"))],
        ]
        tt = Table(ps, colWidths=[45*mm, 125*mm])
        tt.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), base_font),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        elements.append(tt)
        elements.append(Spacer(1, 8))

    # Ranked table
    elements.append(Paragraph("Ranked List", styleH2))
    header = ["Rank","Ticker","Company","TRR","Conf","FF MktCap(USD)","ThemeMktCap_R(USD)","3Y CAGR","Other themes"]
    data = [header]
    for r in rows:
        data.append([
            str(r.get("rank","")),
            str(r.get("ticker","")),
            str(r.get("company",""))[:28],
            f"{float(r.get('trr',0))*100:.1f}%",
            str(r.get("trr_confidence","")),
            fmt_money(r.get("free_float_mktcap_usd")),
            fmt_money(r.get("theme_mktcap_r_usd")),
            fmt_pct(r.get("cagr_3y")),
            str(r.get("other_theme_examples",""))[:40],
        ])
    table = Table(data, colWidths=[10*mm, 18*mm, 40*mm, 14*mm, 12*mm, 30*mm, 30*mm, 16*mm, 36*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), base_font),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    elements.append(table)
    elements.append(PageBreak())

    # Evidence section
    elements.append(Paragraph("Evidence Ledger (TRR)", styleH))
    elements.append(Paragraph("This section is the audit trail. Replace estimated items with disclosed sources as you mature the dataset.", styleS))
    elements.append(Spacer(1, 6))
    for r in rows:
        elements.append(Paragraph(f"{r.get('rank','')}. {r.get('company','')} ({r.get('ticker','')})", styleH2))
        elements.append(Paragraph(f"TRR: {float(r.get('trr',0))*100:.1f}%  |  Method: {r.get('trr_method','')}  |  Confidence: {r.get('trr_confidence','')}", styleN))
        elements.append(Paragraph(f"Source: {r.get('trr_source','')}", styleN))
        elements.append(Paragraph(f"Locator: {r.get('trr_locator','')}", styleS))
        excerpt = str(r.get("trr_excerpt","") or "").strip()
        if excerpt:
            elements.append(Paragraph(f"Excerpt: {excerpt[:300]}", styleS))
        elements.append(Spacer(1, 8))

    doc.build(elements)
    return buf.getvalue()


def fmt_pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"


def fmt_float(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def fmt_money(x) -> str:
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


# =============================================================================
# Engine (audit-friendly, pro)
# =============================================================================
class ThemeEngine:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.universe = UniverseProviderCSV(data_dir)
        self.etf = ETFProviderHeuristic()
        self.market = MarketProviderYF()
        self.exposure = ExposureEstimatorOverrides(data_dir)
        self.benchmark_map = {
            "Global": "ACWI",
            "US": "SPY",
            "Japan": "EWJ",
            "Europe": "VGK",
            "China": "MCHI",
        }

    def run(self, inp: ThemeInput, today: Optional[date] = None) -> Dict[str, Any]:
        today = today or date.today()
        theme = inp.theme_text.strip()
        region = inp.region_mode
        n_max = max(1, min(int(inp.n_max), 30))
        asof = most_recent_month_end(today)

        suggested_etfs = self.etf.suggest(theme, region)
        etfs_used = inp.etf_override or suggested_etfs

        if is_ambiguous_theme(theme, etfs_used):
            profiles = [{"ticker": e, **self.etf.get_profile(e)} for e in etfs_used[:8]]
            return {
                "status": "ambiguous",
                "message": "そのテーマは曖昧で、テーマ運用にはそぐわない（運用/説明の再現性が低い）ため、銘柄選定は行いません。",
                "meta": {
                    "theme": theme,
                    "region_mode": region,
                    "asof_mktcap_date": asof.isoformat(),
                    "etf_profiles": profiles,
                    "hint": "類似ETF/指数（参考）を提示します。テーマをより具体化（対象事業・除外範囲・収益源）して再入力してください。",
                },
                "etfs_used": etfs_used,
            }

        uni, uni_rule = self.universe.get_universe(region, asof)

        # Candidate discovery using ETF holdings (best effort)
        holdings = []
        for e in etfs_used[:6]:
            holdings.extend(self.etf.get_top_holdings(e, top_n=120))
        holding_set = set([h.upper().strip() for h in holdings if h])

        # Candidate policy
        candidates: List[UniverseConstituent] = []
        if holding_set:
            candidates = [s for s in uni if s.ticker.upper().strip() in holding_set]
        if len(candidates) < min(25, max(10, len(uni)//20)):
            candidates = uni[: min(len(uni), 600)]

        warnings: List[str] = []
        if not holding_set:
            warnings.append("ETF holdings intersection not available (or empty). Candidate discovery falls back to top slice of the large-cap universe.")

        # Build pool rows (pre selection)
        pool_rows: List[SecurityRow] = []

        has_snapshot = any(s.free_float_mktcap_usd is not None for s in candidates)
        if not has_snapshot:
            warnings.append("Universe CSV has no free_float_mktcap_usd coverage. Falling back to yfinance proxy (slower & less accurate).")

        for s in candidates:
            prof = self.market.profile(s.ticker)
            company = s.company or prof.get("company") or s.ticker
            listed_country = s.listed_country or prof.get("listed_country") or ""
            sector = s.sector or prof.get("sector") or ""
            currency = s.currency or prof.get("currency") or "USD"

            mcap_quality = "universe_snapshot"
            ff_local = s.free_float_mktcap_local
            ff_usd = s.free_float_mktcap_usd

            if ff_local is None and ff_usd is None:
                ff_local, currency2, ff_usd2, mcap_quality = self.market.free_float_mktcap_asof(s.ticker, asof)
                if currency2:
                    currency = currency2
                ff_usd = ff_usd2

            if ff_local is None and ff_usd is None:
                continue

            exp = self.exposure.estimate(theme, s)
            trr = clamp01(exp.trr)
            tpr = None if exp.tpr is None else clamp01(exp.tpr)

            # Data rigor filters
            if inp.data_rigor == "Strict" and exp.trr_evidence.method != "disclosed":
                continue
            if inp.min_confidence == "Med" and exp.trr_evidence.confidence == "Low":
                continue
            if inp.min_confidence == "High" and exp.trr_evidence.confidence != "High":
                continue

            conf_fac = confidence_factor(exp.trr_evidence.confidence)

            theme_mcap_r_usd = (ff_usd * trr) if ff_usd is not None else None
            theme_mcap_p_usd = (ff_usd * tpr) if (ff_usd is not None and tpr is not None) else None

            pool_rows.append(SecurityRow(
                rank=0,
                ticker=s.ticker,
                exchange=s.exchange,
                company=company,
                listed_country=listed_country,
                sector=sector,
                currency=currency,
                mktcap_asof_date=asof.isoformat(),
                free_float_mktcap_local=float(ff_local) if ff_local is not None else float("nan"),
                free_float_mktcap_usd=float(ff_usd) if ff_usd is not None else None,
                mktcap_quality=mcap_quality,
                trr=float(trr),
                trr_confidence=exp.trr_evidence.confidence,
                trr_method=exp.trr_evidence.method,
                trr_source=exp.trr_evidence.source_name,
                trr_locator=exp.trr_evidence.locator,
                trr_excerpt=exp.trr_evidence.excerpt,
                tpr=float(tpr) if tpr is not None else None,
                tpr_confidence=exp.tpr_evidence.confidence if exp.tpr_evidence else None,
                tpr_method=exp.tpr_evidence.method if exp.tpr_evidence else None,
                tpr_source=exp.tpr_evidence.source_name if exp.tpr_evidence else None,
                tpr_locator=exp.tpr_evidence.locator if exp.tpr_evidence else None,
                theme_mktcap_r_usd=float(theme_mcap_r_usd) if theme_mcap_r_usd is not None else None,
                theme_mktcap_p_usd=float(theme_mcap_p_usd) if theme_mcap_p_usd is not None else None,
                theme_business_summary=exp.theme_business_summary,
                non_theme_business_summary=exp.non_theme_business_summary,
                confidence_factor=float(conf_fac),
            ))

        if not pool_rows:
            return {
                "status": "error",
                "message": "候補銘柄が見つかりませんでした（大型株制約/データ厳格モード/推計排除が強すぎる可能性）。条件を緩めるかテーマを具体化してください。",
                "meta": {"theme": theme, "region_mode": region, "asof_mktcap_date": asof.isoformat()},
                "etfs_used": etfs_used,
                "universe_rule": asdict(uni_rule),
                "warnings": warnings,
            }

        # Rank pool by theme score
        def theme_score_usd(r: SecurityRow) -> float:
            if r.theme_mktcap_r_usd is not None:
                return float(r.theme_mktcap_r_usd)
            if not math.isnan(r.free_float_mktcap_local):
                return float(r.free_float_mktcap_local * r.trr)
            return 0.0

        pool_rows.sort(key=theme_score_usd, reverse=True)
        for i, r in enumerate(pool_rows, start=1):
            r.rank = i

        # Operationalizability score (v2): uses effective name count of theme-score distribution
        scores = np.array([theme_score_usd(r) for r in pool_rows], dtype=float)
        avg_conf = float(np.mean([confidence_factor(r.trr_confidence) for r in pool_rows])) if pool_rows else 0.0
        op = operationalizability_score_v2(
            target_n=n_max,
            raw_candidate_count=len(pool_rows),
            theme_scores=scores,
            avg_conf=avg_conf,
            etf_count=len(etfs_used),
        )

        # Select top N
        rows = pool_rows[:n_max]

        # Crowding / other theme examples (best effort)
        other_map = compute_other_theme_examples(
            tickers=[r.ticker for r in rows],
            theme_text=theme,
            etf_provider=self.etf,
            max_themes_per_ticker=5,
        )
        for r in rows:
            r.other_theme_examples = other_map.get(r.ticker.upper().strip(), "")

        # Price analytics
        bench = self.benchmark_map.get(region, "ACWI")
        end = today
        start = date(end.year - 3, end.month, min(end.day, 28))

        tickers = [r.ticker for r in rows]
        px_map = self.market.price_batch(tickers + [bench], start=start, end=end, auto_adjust=True)
        bench_px = px_map.get(bench, pd.Series(dtype=float))
        bench_ret = bench_px.pct_change().dropna() if bench_px is not None and not bench_px.empty else pd.Series(dtype=float)
        if bench_px is None or bench_px.empty:
            warnings.append(f"Benchmark price series unavailable: {bench}")

        for r in rows:
            px = px_map.get(r.ticker, pd.Series(dtype=float))
            r.price_series_info = f"AdjClose {start.isoformat()}~{end.isoformat()}"
            m = compute_return_metrics(px)
            r.ret_3y, r.cagr_3y, r.vol_3y, r.maxdd_3y = m["ret"], m["cagr"], m["vol"], m["maxdd"]
            r.mom_12m = momentum(px, 252)
            r.mom_3m = momentum(px, 63)

            if px is not None and not px.empty and not bench_ret.empty:
                pr = px.pct_change().dropna()
                b, te, ir = beta_te_ir(pr, bench_ret)
                r.beta_3y, r.te_3y, r.ir_3y = b, te, ir

        # Portfolio weights + stats
        price_mat = align_price_matrix({r.ticker: px_map.get(r.ticker, pd.Series(dtype=float)) for r in rows})
        ret_mat = daily_returns(price_mat)
        weights = make_weights(rows, ret_mat, inp.weighting)
        port_ret = portfolio_returns(ret_mat, weights)
        port_px = (1.0 + port_ret).cumprod()

        pm = compute_return_metrics(port_px)
        bench_px2 = bench_px.reindex(price_mat.index).ffill().dropna() if (bench_px is not None and not bench_px.empty and not price_mat.empty) else pd.Series(dtype=float)
        bench_ret2 = bench_px2.pct_change().dropna() if not bench_px2.empty else pd.Series(dtype=float)
        beta, te, ir = beta_te_ir(port_ret, bench_ret2) if not bench_ret2.empty else (None, None, None)

        meta = {
            "theme": theme,
            "region_mode": region,
            "n_max": n_max,
            "data_rigor": inp.data_rigor,
            "min_confidence": inp.min_confidence,
            "weighting": inp.weighting,
            "mktcap_asof_date": asof.isoformat(),
            "benchmark": bench,
            "price_end_date": end.isoformat(),
            "price_series": "Adj Close (auto_adjust=True)",
            "operationalizability": op,
        }

        portfolio = {
            "portfolio_return_3y": pm["ret"],
            "portfolio_cagr_3y": pm["cagr"],
            "portfolio_vol_3y": pm["vol"],
            "portfolio_maxdd_3y": pm["maxdd"],
            "beta_3y": beta,
            "tracking_error_3y": te,
            "information_ratio_3y": ir,
            "top5_weight": top_n_concentration(weights, 5),
            "hhi": hhi(weights),
            "wavg_trr": float((weights * pd.Series({r.ticker: r.trr for r in rows})).sum()) if not weights.empty else None,
            "wavg_confidence": float((weights * pd.Series({r.ticker: r.confidence_factor for r in rows})).sum()) if not weights.empty else None,
        }

        # Candidate pool payload for robustness: keep topK for speed
        topK = int(min(len(pool_rows), max(200, 6*n_max)))
        pool_for_rob = pool_rows[:topK]
        pool_payload = []
        for r in pool_for_rob:
            base_mcap = r.free_float_mktcap_usd if r.free_float_mktcap_usd is not None else (0.0 if math.isnan(r.free_float_mktcap_local) else float(r.free_float_mktcap_local))
            pool_payload.append({
                "ticker": r.ticker,
                "company": r.company,
                "trr": r.trr,
                "trr_confidence": r.trr_confidence,
                "mcap_base": base_mcap,
                "base_rank": r.rank,
                "base_score": theme_score_usd(r),
            })

        return {
            "status": "ok",
            "message": "ok",
            "meta": meta,
            "universe_rule": asdict(uni_rule),
            "etfs_used": etfs_used,
            "rows": [asdict(r) for r in rows],
            "weights": weights.to_dict(),
            "portfolio": portfolio,
            "candidate_pool": pool_payload,
            "warnings": warnings,
        }


def make_weights(rows: List[SecurityRow], ret_mat: pd.DataFrame, scheme: WeightingScheme) -> pd.Series:
    tickers = [r.ticker for r in rows]
    if not tickers:
        return pd.Series(dtype=float)

    if scheme == "equal":
        w = pd.Series(1.0, index=tickers)
        return w / w.sum()

    theme_mcap = pd.Series({
        r.ticker: (r.theme_mktcap_r_usd if r.theme_mktcap_r_usd is not None else (r.free_float_mktcap_local * r.trr))
        for r in rows
    })

    if scheme == "theme_mktcap":
        return cap_weight(theme_mcap, power=1.0)
    if scheme == "sqrt_theme_mktcap":
        return cap_weight(theme_mcap, power=0.5)
    if scheme == "confidence_adj_theme_mktcap":
        adj = pd.Series({r.ticker: confidence_factor(r.trr_confidence) for r in rows})
        return cap_weight(theme_mcap * adj, power=1.0)
    if scheme == "inv_vol":
        w = inv_vol_weights(ret_mat)
        if not w.empty:
            return w
        w = pd.Series(1.0, index=tickers)
        return w / w.sum()

    w = pd.Series(1.0, index=tickers)
    return w / w.sum()


def snapshot_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


# =============================================================================
# Streamlit UI entrypoint
# =============================================================================
@st.cache_resource
def _engine(data_dir: str) -> ThemeEngine:
    return ThemeEngine(data_dir=data_dir)


@st.cache_data(ttl=60 * 60)
def _run_engine_cached(inp_dict: Dict[str, Any], data_dir: str) -> Dict[str, Any]:
    eng = _engine(data_dir)
    inp = ThemeInput(**inp_dict)
    return eng.run(inp)


def render_next_gen_tab(data_dir: str = "data"):
    """Call this from app.py inside a Streamlit tab."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    st.markdown(
        """
<div class="ng-hero">
  <p class="ng-title">NEXT GEN APP — Theme Portfolio Builder</p>
  <p class="ng-subtitle">
    Evidence-first / Pro analytics / Reproducible snapshots.  テーマ関連売上比率(TRR)×free-float時価総額で「テーマ関連時価総額」を算出し、大型株だけで上位Nを抽出します。
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    with st.sidebar:
        st.header("Inputs")
        theme_text = st.text_input("Theme (自由入力)", value="半導体")
        region_mode = st.selectbox("Region", ["Global", "Japan", "US", "Europe", "China"], index=0)
        n_max = st.slider("Top N", 1, 30, 10, 1)
        data_rigor = st.selectbox("Data rigor", ["Balanced", "Strict", "Expand"], index=0, help="Strict=開示TRRのみ / Expand=推計も積極採用")
        min_conf = st.selectbox("Min confidence", ["Low", "Med", "High"], index=0)
        weighting = st.selectbox("Weighting (for portfolio analytics)", ["theme_mktcap", "sqrt_theme_mktcap", "confidence_adj_theme_mktcap", "equal", "inv_vol"], index=0)

        st.divider()
        etf_override_raw = st.text_input("ETF tickers override (comma)", value="", help="例：SMH, SOXX")
        etf_override = [x.strip().upper() for x in etf_override_raw.split(",") if x.strip()] or None

        run = st.button("Run Screen", type="primary", use_container_width=True)

    if not run:
        st.info("左サイドバーで条件を指定して実行してください。曖昧テーマは『運用に不適』として銘柄抽出せず、類似ETFを提示します。")
        return

    inp = ThemeInput(
        theme_text=theme_text,
        region_mode=region_mode,  # type: ignore
        n_max=int(n_max),
        data_rigor=data_rigor,    # type: ignore
        min_confidence=min_conf,  # type: ignore
        weighting=weighting,      # type: ignore
        etf_override=etf_override,
    )

    with st.spinner("Building theme portfolio…"):
        res = _run_engine_cached(asdict(inp), data_dir=data_dir)

    # Snapshot ID (stable)
    sid = snapshot_id({"input": asdict(inp), "meta": res.get("meta", {}), "asof": res.get("meta", {}).get("mktcap_asof_date")})
    res["snapshot_id"] = sid

    if res.get("status") == "ambiguous":
        st.error(res.get("message"))
        profiles = res.get("meta", {}).get("etf_profiles", [])
        st.write("### 参考ETF（類似テーマ）")
        if profiles:
            st.dataframe(pd.DataFrame(profiles), use_container_width=True)
        else:
            st.write(res.get("etfs_used", []))

        st.write("### テーマを運用可能にするヒント")
        st.markdown(
            """- 対象事業（収益源）を明確化（例：『半導体』→『半導体製造装置』/『メモリ』など）
- 除外範囲を明確化（例：『AI』→『AIインフラ(データセンター/半導体/クラウド)』など）
- ETFが複数存在し、保有銘柄の重なりが出るテーマは運用再現性が上がります"""
        )
        st.caption(f"Snapshot ID: {sid}")
        return

    if res.get("status") == "error":
        st.error(res.get("message"))
        if res.get("warnings"):
            st.warning("\n".join(res["warnings"]))
        st.caption(f"Snapshot ID: {sid}")
        return

    # OK
    df = pd.DataFrame(res.get("rows", []))
    weights = pd.Series(res.get("weights", {}), dtype=float)
    meta = res.get("meta", {}) or {}
    uni_rule = res.get("universe_rule", {}) or {}
    portfolio = res.get("portfolio", {}) or {}
    warnings = res.get("warnings", []) or []
    op = meta.get("operationalizability", {}) or {}

    # KPI cards
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="ng-card"><div class="ng-kpi">As-of (MktCap)</div><div class="ng-kpi-val">{meta.get("mktcap_asof_date","-")}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="ng-card"><div class="ng-kpi">Benchmark</div><div class="ng-kpi-val">{meta.get("benchmark","-")}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="ng-card"><div class="ng-kpi">Operationalizability</div><div class="ng-kpi-val">{op.get("score","-")}</div></div>', unsafe_allow_html=True)
    with c4:
        wtrr = portfolio.get("wavg_trr", None)
        st.markdown(f'<div class="ng-card"><div class="ng-kpi">Wgt Avg TRR</div><div class="ng-kpi-val">{(wtrr*100):.1f}%</div></div>' if wtrr is not None else '<div class="ng-card"><div class="ng-kpi">Wgt Avg TRR</div><div class="ng-kpi-val">-</div></div>', unsafe_allow_html=True)
    with c5:
        cagr = portfolio.get("portfolio_cagr_3y", None)
        st.markdown(f'<div class="ng-card"><div class="ng-kpi">3Y CAGR</div><div class="ng-kpi-val">{(cagr*100):.1f}%</div></div>' if cagr is not None else '<div class="ng-card"><div class="ng-kpi">3Y CAGR</div><div class="ng-kpi-val">-</div></div>', unsafe_allow_html=True)

    # Meta tags
    tags = []
    tags.append(f"Universe: {uni_rule.get('name','-')}")
    tags.append(f"Weighting: {meta.get('weighting','-')}")
    tags.append(f"Data rigor: {meta.get('data_rigor','-')}")
    tags.append(f"Min conf: {meta.get('min_confidence','-')}")
    tags.append(f"ETFs: {', '.join(res.get('etfs_used', [])[:6]) or '-'}")
    st.markdown("".join([f'<span class="ng-tag">{t}</span>' for t in tags]), unsafe_allow_html=True)

    if warnings:
        st.warning("\n".join(warnings))

    tab_dash, tab_list, tab_ev, tab_rob, tab_export = st.tabs(["Dashboard", "Ranked List", "Evidence", "Robustness", "Export"])

    # Dashboard
    with tab_dash:
        st.subheader("Operationalizability (Investability) Score")
        if op:
            st.markdown(f"**{op.get('score','-')} / 100**  -  {op.get('headline','')}")
            with st.expander("Why this score?", expanded=False):
                for r in op.get("reasons", []) or []:
                    st.write(f"- {r}")
                st.write("")
                st.write("**Suggestions**")
                for s in op.get("suggestions", []) or []:
                    st.write(f"- {s}")

        st.subheader("Portfolio Analytics (3Y)")
        mcols = st.columns(6)
        mets = [
            ("3Y Return", portfolio.get("portfolio_return_3y")),
            ("Vol", portfolio.get("portfolio_vol_3y")),
            ("MaxDD", portfolio.get("portfolio_maxdd_3y")),
            ("Beta", portfolio.get("beta_3y")),
            ("TE", portfolio.get("tracking_error_3y")),
            ("IR", portfolio.get("information_ratio_3y")),
        ]
        for col, (lab, val) in zip(mcols, mets):
            with col:
                if val is None:
                    st.metric(lab, "-")
                else:
                    if lab in ["Beta", "IR"]:
                        st.metric(lab, f"{val:.2f}")
                    else:
                        st.metric(lab, f"{val*100:.1f}%")

        st.write("### Theme Purity vs Scale")
        if not df.empty:
            plot_df = df.copy()
            plot_df["TRR(%)"] = plot_df["trr"] * 100.0
            plot_df["FF MktCap (USD)"] = plot_df["free_float_mktcap_usd"]
            plot_df["ThemeMktCap_R (USD)"] = plot_df["theme_mktcap_r_usd"]
            plot_df["Label"] = plot_df["company"] + " (" + plot_df["ticker"] + ")"
            if _PLOTLY:
                fig = px.scatter(
                    plot_df,
                    x="TRR(%)",
                    y="FF MktCap (USD)",
                    size="ThemeMktCap_R (USD)",
                    hover_name="Label",
                    color="trr_confidence",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("plotly未導入の場合は散布図は省略されます（plotly推奨）。")

        st.write("### Concentration (weights)")
        if not weights.empty:
            w = weights.sort_values(ascending=False)
            w_df = pd.DataFrame({"ticker": w.index, "weight": w.values})
            if _PLOTLY:
                fig = px.bar(w_df.head(15), x="ticker", y="weight", height=320)
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(w_df.head(15), use_container_width=True)

    # Ranked List
    with tab_list:
        st.subheader("Ranked List (ThemeMktCap_R desc)")
        show_cols = [
            "rank","company","ticker","exchange","listed_country","sector",
            "trr","trr_confidence","trr_method",
            "free_float_mktcap_usd","theme_mktcap_r_usd",
            "other_theme_examples",
            "ret_3y","cagr_3y","vol_3y","maxdd_3y","beta_3y",
            "mom_12m","mom_3m",
        ]
        exist = [c for c in show_cols if c in df.columns]
        st.dataframe(df[exist], use_container_width=True, height=560)
        st.caption("Other_theme_examples はETF overlapのベストエフォート。商用ではETF holdings feed接続で精度が上がります。")

    # Evidence
    with tab_ev:
        st.subheader("Evidence Ledger (Audit-ready)")
        st.caption("TRR/TPRは必ず根拠（source/locator/excerpt）を持ちます。推計は推計として明示します。")

        for _, r in df.sort_values("rank").iterrows():
            title = f"{int(r['rank'])}. {r['company']} ({r['ticker']}) — TRR {r['trr']*100:.1f}% [{r['trr_confidence']}]"
            with st.expander(title, expanded=False):
                st.write("**Theme business**")
                st.write(r.get("theme_business_summary","") or "-")
                st.write("**Non-theme business**")
                st.write(r.get("non_theme_business_summary","") or "-")
                st.write("**Other themes (ETF overlap)**")
                st.write(r.get("other_theme_examples","") or "-")
                st.divider()
                st.write("**TRR Evidence**")
                st.json({
                    "source": r.get("trr_source",""),
                    "locator": r.get("trr_locator",""),
                    "method": r.get("trr_method",""),
                    "confidence": r.get("trr_confidence",""),
                    "excerpt": (str(r.get("trr_excerpt",""))[:200] + ("…" if str(r.get("trr_excerpt","")).strip() and len(str(r.get("trr_excerpt",""))) > 200 else "")),
                })

        st.divider()
        st.subheader("Evidence Editor (local overrides)")
        st.caption("まずはローカルJSONに保存。商用では『承認フロー（レビュー/差分/署名）』を推奨。")
        if df.empty:
            st.info("No rows")
        else:
            pick = st.selectbox("Pick ticker to override", df["ticker"].tolist())
            row = df[df["ticker"] == pick].iloc[0]
            trr_new = st.number_input("TRR (0-1)", min_value=0.0, max_value=1.0, value=float(row["trr"]), step=0.01)
            conf_new = st.selectbox("Confidence", ["High","Med","Low"], index=["High","Med","Low"].index(str(row["trr_confidence"])))
            method_new = st.selectbox("Method", ["disclosed","estimated","proxy"], index=["disclosed","estimated","proxy"].index(str(row["trr_method"])))
            source_name = st.text_input("Source name", value=str(row.get("trr_source","") or ""))
            locator = st.text_input("Locator (page/section)", value=str(row.get("trr_locator","") or ""))
            excerpt = st.text_area("Short excerpt (<=2-3 lines)", value=str(row.get("trr_excerpt","") or ""), height=120)

            if st.button("Save override (JSON)", type="secondary"):
                eng = _engine(data_dir)
                payload = {
                    "trr": float(trr_new),
                    "tpr": None,
                    "theme_business_summary": str(row.get("theme_business_summary","") or ""),
                    "non_theme_business_summary": str(row.get("non_theme_business_summary","") or ""),
                    "trr_evidence": {
                        "source_name": source_name,
                        "source_year": None,
                        "locator": locator,
                        "excerpt": excerpt,
                        "confidence": conf_new,
                        "method": method_new,
                        "notes": "",
                    }
                }
                ok, msg = eng.exposure.save_override(theme_text, pick, payload)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    # Robustness
    with tab_rob:
        st.subheader("Robustness / Sensitivity (TRR uncertainty)")
        st.caption("TRRが揺れてもTop Nが崩れないかをチェック。Low confidenceほど揺れ幅を大きくします。")
        pool = pd.DataFrame(res.get("candidate_pool", []))
        if pool.empty:
            st.info("No candidate pool available.")
        else:
            iters = st.slider("Iterations", 50, 500, 200, 50)
            freq, summary = robustness_simulation(pool, top_n=int(n_max), iters=int(iters))
            if summary.get("status") == "ok":
                st.write(f"Baseline TopN mean inclusion: **{summary.get('baseline_topN_mean_inclusion',0):.2f}**")
                st.write(f"Baseline TopN min inclusion: **{summary.get('baseline_topN_min_inclusion',0):.2f}**")
                st.write(f"Unstable names in baseline TopN (<0.60): **{summary.get('unstable_names_in_baseline_topN',0)}**")
            st.write("### Top-N inclusion rate (top 30)")
            freq_df = freq.head(30)
            if _PLOTLY and not freq_df.empty:
                fig = px.bar(freq_df, x="ticker", y="topN_inclusion_rate", height=380)
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(freq_df, use_container_width=True)

    # Export
    with tab_export:
        st.subheader("Export / Snapshot")
        st.code(f"Snapshot ID: {sid}")

        payload = {
            "snapshot_id": sid,
            "input": asdict(inp),
            "meta": meta,
            "universe_rule": uni_rule,
            "etfs_used": res.get("etfs_used", []),
            "rows": res.get("rows", []),
            "weights": res.get("weights", {}),
            "portfolio": portfolio,
            "warnings": warnings,
        }

        st.download_button(
            "Download Ranked List (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"theme_portfolio_{sid}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Full Snapshot (JSON)",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"theme_portfolio_snapshot_{sid}.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("PDF report (log)")
        st.caption("新たな分析ではなく、出力のログとしてのPDFです。監査/共有用途。")
        if st.button("Generate PDF", type="primary"):
            with st.spinner("Generating PDF…"):
                try:
                    pdf_bytes = build_pdf_report(payload)
                    st.session_state["ng_pdf_bytes"] = pdf_bytes
                    st.success("PDF ready.")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

        pdf_bytes = st.session_state.get("ng_pdf_bytes", None)
        if pdf_bytes:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"theme_portfolio_report_{sid}.pdf",
                mime="application/pdf",
            )

    st.caption(f"Snapshot ID: {sid}")
