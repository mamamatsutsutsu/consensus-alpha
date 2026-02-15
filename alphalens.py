import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================================
# DATA
# ==========================================================

@st.cache_data(ttl=1800)
def fetch_prices(tickers, period="1y"):
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")

# ==========================================================
# METRICS
# ==========================================================

def calc_metrics(series: pd.Series) -> dict:
    if series is None or len(series) < 20:
        return {"Ret": 0.0, "RS": 0.0, "Vol": 0.0}
    ret = series.pct_change().dropna()
    total_ret = (series.iloc[-1] / series.iloc[0] - 1.0) * 100.0
    vol = ret.std() * np.sqrt(252) * 100.0
    return {"Ret": round(float(total_ret), 2), "RS": round(float(total_ret), 2), "Vol": round(float(vol), 2)}

# ==========================================================
# SAFE BENCHMARK RESOLUTION
# ==========================================================

def resolve_benchmark(prices: pd.DataFrame, requested: str, market: str) -> str | None:
    if requested in prices.columns:
        return requested

    st.warning(f"BENCHMARK MISSING: requested={requested}. Using regional proxy.")

    if market == "JP":
        for proxy in ["^TOPX", "^N225", "1321.T"]:
            if proxy in prices.columns:
                return proxy
    if market == "US":
        for proxy in ["^GSPC", "SPY"]:
            if proxy in prices.columns:
                return proxy

    for col in prices.columns:
        try:
            if prices[col].notna().sum() > 30:
                return col
        except Exception:
            continue
    return None

# ==========================================================
# SPREAD SAFE
# ==========================================================

def compute_spread(stats: dict) -> float:
    try:
        vals = [float(v.get("RS", 0.0)) for v in stats.values() if isinstance(v, dict)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(max(vals) - min(vals)) if vals else 0.0
    except Exception:
        return 0.0

# ==========================================================
# FACTOR CLASSIFICATION (simple baseline)
# ==========================================================

POS_WORDS = ["上昇", "改善", "好調", "増益", "回復", "堅調", "上方修正", "利下げ", "インフレ鈍化"]
NEG_WORDS = ["下落", "悪化", "減益", "売り", "警告", "悪材料", "下方修正", "利上げ", "金利上昇", "地政学"]

def classify_factors(text: str):
    pos, neg, other = [], [], []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        core = l.lstrip("-").strip()
        if any(w in core for w in POS_WORDS):
            pos.append(core)
        elif any(w in core for w in NEG_WORDS):
            neg.append(core)
        else:
            other.append(core)
    return pos, neg, other

# ==========================================================
# FAST TOP PICK (momentum-heavy baseline)
# ==========================================================

def fast_top_pick(stats: dict) -> str | None:
    if not stats:
        return None
    scores = {}
    for k, v in stats.items():
        if not isinstance(v, dict):
            continue
        ret = float(v.get("Ret", 0.0))
        rs = float(v.get("RS", 0.0))
        vol = float(v.get("Vol", 0.0))
        scores[k] = 0.45 * ret + 0.35 * rs - 0.20 * vol
    return max(scores, key=scores.get) if scores else None

# ==========================================================
# MAIN
# ==========================================================

def run():
    st.title("ALPHALENS")

    market = st.selectbox("Market", ["US", "JP"])
    window = st.selectbox("Window", ["1Y", "6M", "3M"])
    period = {"1Y": "1y", "6M": "6mo", "3M": "3mo"}[window]

    if market == "US":
        bench = "SPY"
        universe = ["AAPL", "MSFT", "NVDA", "AMZN"]
    else:
        bench = "1306.T"
        universe = ["7203.T", "6758.T", "9984.T", "8035.T"]

    tickers = sorted(set(universe + [bench]))
    prices = fetch_prices(tickers, period=period)

    if prices is None or prices.empty:
        st.error("No price data available.")
        return

    bench_used = resolve_benchmark(prices, bench, market)
    if bench_used is None:
        st.error("No usable benchmark.")
        return

    stats = {}
    for t in universe:
        if t in prices.columns:
            stats[t] = calc_metrics(prices[t])

    spread = compute_spread(stats)
    bench_stats = calc_metrics(prices[bench_used])
    regime = "Bull" if bench_stats["Ret"] > 0 else "Bear"

    st.subheader(f"MARKET PULSE ({period})")
    st.write(f"Spread: {spread:.1f}pt | Regime: {regime}")

    # Baseline example for factor block (replace with your AI output)
    sample_text = """テクノロジー株が上昇
金利上昇懸念で市場下落
決算好調で改善
地政学リスク悪化"""

    pos, neg, other = classify_factors(sample_text)

    st.markdown("### 【主な変動要因】")
    st.markdown("**(+) 上昇要因**")
    for p in pos:
        st.write(f"- {p}")
    st.markdown("**(−) 下落要因**")
    for n in neg:
        st.write(f"- {n}")
    if other:
        st.markdown("**(補足)**")
        for o in other:
            st.write(f"- {o}")

    st.markdown("### 🦅🤖 AI AGENT SECTOR REPORT")
    top = fast_top_pick(stats)
    if top:
        st.write(f"Top Pick (3M horizon): **{top}**")
        st.write("Rationale (baseline):")
        st.write("- Strong recent momentum")
        st.write("- High relative strength")
        st.write("- Volatility not extreme")
    else:
        st.write("No valid selection.")
