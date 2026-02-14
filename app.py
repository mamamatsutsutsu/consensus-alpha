# app.py — AlphaLens Apex v27.1 "Discipline × Names × News"
# - Company names shown (table + cards)
# - Sector stock details + deterministic AI analysis/reco
# - News aggregation via yfinance.Ticker().news
# - Integrity gatekeeper; no pandas Styler / no matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(page_title="AlphaLens Apex", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .main { background-color:#0d1117; }
  .command-deck { background:#161b22; padding:14px; border-radius:12px; border:1px solid #30363d; margin-bottom:14px; }
  .kpi { background:#1c2128; border-radius:10px; padding:12px; border-left:6px solid #30363d; margin-bottom:10px; }
  .status-green { border-left-color:#238636 !important; }
  .status-yellow { border-left-color:#d29922 !important; }
  .status-red { border-left-color:#da3633 !important; }
  .chip { display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #30363d; color:#c9d1d9; font-size:12px; margin-right:6px; }
  .cert { background:#0d1117; border:1px dashed #238636; border-radius:10px; padding:10px 12px; color:#7ee787;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  .card { background:#1c2128; border:1px solid #30363d; border-radius:12px; padding:12px; margin-bottom:10px; }
  .muted { color:#8b949e; font-size:12px; }
  .headline { font-size:18px; font-weight:800; color:#e6edf3; }
  .warn { color:#ff7b72; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Windows
# -----------------------------
LOOKBACKS = {
    "1W (5d)": 5,
    "1M (21d)": 21,
    "3M (63d)": 63,
    "12M (252d)": 252,
}

# -----------------------------
# Sector ETFs (US=10, JP=17)
# -----------------------------
US_BENCH = "SPY"
JP_BENCH = "1306.T"

US_SECTOR_ETF = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Cons. Discretionary": "XLY",
    "Cons. Staples": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
}

JP_TOPIX17_ETF = {
    "Energy": "1617.T",
    "Construction/Materials": "1618.T",
    "Raw Materials/Chem": "1619.T",
    "Industrials": "1620.T",
    "Autos/Transport": "1621.T",
    "Retail/Wholesale": "1622.T",
    "Banking": "1623.T",
    "Financials": "1624.T",
    "Real Estate": "1625.T",
    "Info/Comm": "1626.T",
    "Services": "1627.T",
    "Electric/Gas": "1628.T",
    "Steel/Nonferrous": "1629.T",
    "Machinery": "1630.T",
    "Electronics": "1631.T",
    "Pharma": "1632.T",
    "Foods": "1633.T",
}

# -----------------------------
# Stock dictionaries (≈200 each, aligned to ETF sector schema)
# -----------------------------
US_STOCKS_BY_SECTOR: Dict[str, List[str]] = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTU","IBM","AMD","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","SNPS","CDNS"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","VRTX","BMY","GILD","PFE","REGN","SYK","BSX","MDT","ZTS","HCA"],
    "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","PNC","USB","TFC","MMC","AIG","MET","PRU","AFL","CB","ICE"],
    "Cons. Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ROST","GM","F","MAR","HLT","EBAY","ETSY","CMG","YUM","LULU","DHI"],
    "Cons. Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","SYY","KR","TGT","EL","HSY","STZ","KDP","WBA"],
    "Industrials": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","NSC","WM","FDX","NOC","GD","PCAR","ROK"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","HAL","BKR","DVN","HES","APA","FANG","WMB","TRGP","OKE","EQNR"],
    "Materials": ["LIN","APD","SHW","ECL","FCX","NEM","DOW","DD","NUE","VMC","MLM","ALB","CF","MOS","IP","BALL","EMN","LYB","PPG","IFF"],
    "Utilities": ["NEE","DUK","SO","EXC","AEP","SRE","XEL","D","ED","PEG","EIX","PCG","AWK","WEC","ES","PPL","ETR","CMS","CNP","NI"],
    "Real Estate": ["AMT","PLD","CCI","EQIX","SPG","O","PSA","WELL","DLR","AVB","EQR","VTR","IRM","VICI","SBAC","EXR","MAA","ARE","KIM","REG"],
}

JP_STOCKS_BY_SECTOR: Dict[str, List[str]] = {
    "Energy": ["1605.T","5020.T","5019.T","9501.T","9502.T","9503.T","9513.T","9511.T","9517.T","9531.T","9532.T","9533.T"],
    "Construction/Materials": ["1801.T","1802.T","1803.T","1812.T","1925.T","1928.T","1721.T","1719.T","1820.T","1861.T","1878.T","1963.T"],
    "Raw Materials/Chem": ["4005.T","4021.T","4042.T","4063.T","4188.T","4208.T","4452.T","4631.T","4901.T","4911.T","6988.T","3407.T"],
    "Industrials": ["6301.T","6305.T","6367.T","6471.T","6473.T","7011.T","7012.T","7013.T","5631.T","6103.T","6113.T","6273.T"],
    "Autos/Transport": ["7203.T","7267.T","6902.T","7201.T","7211.T","7269.T","7270.T","7261.T","7272.T","7259.T","6201.T","9101.T"],
    "Retail/Wholesale": ["8001.T","8002.T","8003.T","8004.T","8015.T","8031.T","8058.T","8053.T","2730.T","3382.T","7453.T","7532.T"],
    "Banking": ["8306.T","8316.T","8411.T","8308.T","8331.T","8354.T","8355.T","7186.T","7182.T","7167.T","5831.T","5830.T"],
    "Financials": ["8591.T","8604.T","8630.T","8725.T","8766.T","8750.T","8253.T","8570.T","8697.T","8703.T","8795.T","8798.T"],
    "Real Estate": ["8801.T","8802.T","8830.T","3289.T","3003.T","9001.T","9005.T","9020.T","9022.T","9042.T","9041.T","9142.T"],
    "Info/Comm": ["9432.T","9433.T","9434.T","9984.T","4689.T","6098.T","4755.T","4385.T","3923.T","3774.T","3659.T","2413.T"],
    "Services": ["4661.T","9735.T","9766.T","4324.T","2127.T","7088.T","6028.T","3038.T","6183.T","2432.T","3774.T","6098.T"],
    "Electric/Gas": ["9501.T","9503.T","9504.T","9506.T","9508.T","9511.T","9513.T","9531.T","9532.T","9533.T","9534.T","9536.T"],
    "Steel/Nonferrous": ["5401.T","5406.T","5411.T","5711.T","5713.T","5802.T","5801.T","5706.T","5707.T","3436.T","5486.T","5726.T"],
    "Machinery": ["6146.T","6268.T","6361.T","6472.T","6460.T","7011.T","7012.T","7013.T","6302.T","6331.T","6104.T","6201.T"],
    "Electronics": ["8035.T","6857.T","6723.T","6920.T","7735.T","6963.T","6762.T","6861.T","6981.T","6758.T","6501.T","6702.T"],
    "Pharma": ["4502.T","4503.T","4507.T","4519.T","4523.T","4568.T","4565.T","4578.T","4587.T","4151.T","4536.T","4901.T"],
    "Foods": ["2801.T","2802.T","2269.T","2914.T","2502.T","2503.T","2579.T","2587.T","2002.T","2201.T","2206.T","2222.T"],
}

# Explicit name fallbacks (fast + JP-friendly). yfinance infoが取れない時の保険。
NAME_FALLBACK = {
    # JP examples
    "8035.T": "東京エレクトロン", "6857.T": "アドバンテスト", "6920.T": "レーザーテック",
    "8306.T": "三菱UFJ FG", "8316.T": "三井住友FG", "7203.T": "トヨタ自動車",
    "9984.T": "ソフトバンクG", "9983.T": "ファーストリテイリング",
    "8058.T": "三菱商事", "8031.T": "三井物産",
    # US examples
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon", "TSLA": "Tesla",
    "LLY": "Eli Lilly", "JPM": "JPMorgan Chase",
}

MARKETS = {
    "🇺🇸 US": {"bench": US_BENCH, "bench_name": "S&P 500 (via SPY)", "sector_etf": US_SECTOR_ETF, "stocks_by_sector": US_STOCKS_BY_SECTOR},
    "🇯🇵 JP": {"bench": JP_BENCH, "bench_name": "TOPIX (via 1306.T)", "sector_etf": JP_TOPIX17_ETF, "stocks_by_sector": JP_STOCKS_BY_SECTOR},
}

# -----------------------------
# Cache helpers
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str = "12mo", chunk_size: int = 80) -> pd.DataFrame:
    tickers = tuple(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    frames = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            raw = yf.download(
                " ".join(chunk),
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            frames.append(raw)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name_cached(ticker: str) -> str:
    # 速度優先：fallback → yfinance info（遅いのでキャッシュ必須）
    if ticker in NAME_FALLBACK:
        return NAME_FALLBACK[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or info.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return ticker

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_cached(ticker: str, max_items: int = 8) -> List[dict]:
    # yfinance news: title, link, publisher, providerPublishTime
    try:
        news = yf.Ticker(ticker).news or []
        return news[:max_items]
    except Exception:
        return []

# -----------------------------
# Data utilities (no SciPy, no matplotlib)
# -----------------------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    mu = float(s.mean(skipna=True))
    sig = float(s.std(ddof=0, skipna=True))
    if sig == 0.0 or np.isnan(sig):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sig

def _is_multiindex(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2

def extract_close_matrix(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    close = pd.DataFrame()
    if _is_multiindex(df):
        lv0 = set(df.columns.get_level_values(0))
        lv1 = set(df.columns.get_level_values(1))
        try:
            if "Close" in lv0:
                close = df.xs("Close", axis=1, level=0)
            elif "Close" in lv1:
                close = df.xs("Close", axis=1, level=1)
            else:
                return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    else:
        # 多銘柄でSINGLEが出たら推測しない（誠実）
        return pd.DataFrame()
    close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    keep = [c for c in expected if c in close.columns]
    return close[keep]

def calc_return(series: pd.Series, win_days: int) -> float:
    s = series.tail(win_days + 1)
    if len(s) < win_days + 1 or s.isna().any():
        return np.nan
    return float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0)

def calc_max_dd(series: pd.Series, win_days: int) -> float:
    s = series.tail(win_days + 1)
    if len(s) < win_days + 1 or s.isna().any():
        return np.nan
    dd = (s / s.cummax() - 1.0) * 100.0
    return float(abs(dd.min()))

def vol_annualized(series: pd.Series, win_days: int) -> float:
    s = series.tail(win_days + 1)
    if len(s) < win_days + 1 or s.isna().any():
        return np.nan
    r = s.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(252) * 100.0)

def rs_stability(asset: pd.Series, bench: pd.Series) -> str:
    a = asset.tail(6).dropna()
    b = bench.tail(6).dropna()
    if len(a) < 6 or len(b) < 6:
        return "⚠️"
    a_ret = (a.iloc[-1] / a.iloc[0] - 1.0) * 100.0
    b_ret = (b.iloc[-1] / b.iloc[0] - 1.0) * 100.0
    rs = a_ret - b_ret
    return "✅" if np.sign(rs) != 0 else "⚠️"

def integrity_audit(expected: List[str], close_df: pd.DataFrame, win_days: int) -> Dict:
    exp = len(expected)
    if close_df is None or close_df.empty:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0,
                "mode_date": None, "computable_list": [],
                "drop": {"missing": exp, "desync": 0, "short": 0}}

    present = [t for t in expected if t in close_df.columns]
    if not present:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0,
                "mode_date": None, "computable_list": [],
                "drop": {"missing": exp, "desync": 0, "short": 0}}

    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    synced_mask = (last_dates == mode_date)
    synced = int(synced_mask.sum())

    computable = []
    drop_missing = exp - len(present)
    drop_desync = 0
    drop_short = 0

    for t in present:
        if not bool(synced_mask.get(t, False)):
            drop_desync += 1
            continue
        tail = close_df[t].tail(win_days + 1)
        if (tail.notna().sum() < (win_days + 1)) or tail.isna().any():
            drop_short += 1
            continue
        computable.append(t)

    return {
        "expected": exp, "present": len(present), "synced": synced, "computable": len(computable),
        "mode_date": mode_date, "computable_list": computable,
        "drop": {"missing": drop_missing, "desync": drop_desync, "short": drop_short},
    }

def status_cls(ratio: float) -> str:
    if ratio >= 0.9: return "status-green"
    if ratio >= 0.7: return "status-yellow"
    return "status-red"

def kpi(label: str, value: str, cls: str):
    st.markdown(f"""
    <div class="kpi {cls}">
      <div class="muted">{label}</div>
      <div style="font-size:18px;font-weight:800;color:#e6edf3">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def market_comment(bench_ret: float, leaders_ratio: float, dispersion: float) -> str:
    tone = "RISK-ON" if (bench_ret > 0 and leaders_ratio > 0.55) else "RISK-OFF" if (bench_ret < 0 and leaders_ratio < 0.45) else "MIXED"
    rot = "Rotation ACTIVE" if dispersion >= 2.0 else "Rotation QUIET"
    breadth = "Broad participation" if leaders_ratio >= 0.6 else "Narrow leadership" if leaders_ratio <= 0.4 else "Neutral breadth"
    return f"{tone} | {breadth} | {rot} — benchmark={bench_ret:+.2f}% / leaders={leaders_ratio:.0%} / dispersion={dispersion:.2f}"

def sector_overview(sec_name: str, sec_ret: float, bench_ret: float, health: float) -> str:
    rel = sec_ret - bench_ret
    stance = "LEADING" if rel > 0.5 else "LAGGING" if rel < -0.5 else "INLINE"
    qual = "High integrity" if health >= 0.9 else "Degraded integrity" if health >= 0.7 else "Low integrity"
    return f"{sec_name}: {stance} (sector={sec_ret:+.2f}% vs market={bench_ret:+.2f}%, rel={rel:+.2f}%) | {qual} (health={health:.0%})"

def compute_apex(close_df: pd.DataFrame, bench: str, targets: List[str], win_days: int) -> pd.DataFrame:
    b = close_df[bench]
    b_ret = calc_return(b, win_days)
    rows = []
    for t in targets:
        if t == bench:
            continue
        s = close_df[t]
        p_ret = calc_return(s, win_days)
        if np.isnan(p_ret) or np.isnan(b_ret):
            continue
        rs = p_ret - b_ret
        half = max(1, win_days // 2)
        seg = s.tail(half + 1)
        if len(seg) < half + 1 or seg.isna().any():
            continue
        p_half = float((seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0)
        accel = p_half - (p_ret / 2.0)
        dd = calc_max_dd(s, win_days)
        vol = vol_annualized(s, win_days)
        stable = rs_stability(s, b)
        rows.append({"Ticker": t, "RS": rs, "Accel": accel, "MaxDD": dd, "Vol(ann%)": vol, "Stable": stable})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["RS_z"] = zscore_series(df["RS"])
    df["Accel_z"] = zscore_series(df["Accel"])
    df["DD_z"] = zscore_series(df["MaxDD"])
    df["ApexScore"] = (0.6 * df["RS_z"]) + (0.25 * df["Accel_z"]) - (0.15 * df["DD_z"])
    return df.sort_values("ApexScore", ascending=False, na_position="last").reset_index(drop=True)

def reco_bucket(row: pd.Series) -> str:
    # “AI推奨”はルールベースで再現性を担保
    if row["RS"] > 0 and row["Accel"] > 0 and row["MaxDD"] <= np.nanmedian([row["MaxDD"], 10.0]) and row["Stable"] == "✅":
        return "STRONG"
    if row["RS"] > 0 and row["Accel"] >= 0:
        return "WATCH"
    return "AVOID"

# -----------------------------
# UI — Command Deck
# -----------------------------
st.title("🛰️ AlphaLens Apex — Sector Discipline Command Center")

with st.container():
    st.markdown('<div class="command-deck">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
    with c1:
        market_key = st.selectbox("Market", list(MARKETS.keys()), index=0)
    with c2:
        window_label = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
    with c3:
        st.write("")
        sync_btn = st.button("SYNC", type="primary", use_container_width=True)
    if st.session_state.get("ts"):
        st.caption(f"Last sync: {st.session_state.ts}")
    st.markdown("</div>", unsafe_allow_html=True)

win_days = LOOKBACKS[window_label]
cfg = MARKETS[market_key]
bench = cfg["bench"]
bench_name = cfg["bench_name"]
sector_etf_map = cfg["sector_etf"]
sector_names = list(sector_etf_map.keys())

# -----------------------------
# STEP 0 — Core sync (bench + sector ETFs)
# -----------------------------
core_expected = [bench] + list(sector_etf_map.values())
core_key = f"core::{market_key}"

need_core_fetch = sync_btn or (st.session_state.get("core_key") != core_key) or ("core_close" not in st.session_state)
if need_core_fetch:
    with st.status("Sync: Market + Sector ETFs ...", expanded=False):
        raw = fetch_bulk_cached(tuple(core_expected), period="12mo", chunk_size=80)
        close = extract_close_matrix(raw, expected=core_expected)
        st.session_state.core_close = close
        st.session_state.core_key = core_key
        st.session_state.ts = datetime.now().strftime("%H:%M:%S")

core_close: pd.DataFrame = st.session_state.get("core_close", pd.DataFrame())
core_audit = integrity_audit(core_expected, core_close, win_days)

st.subheader("🛡️ Integrity Panel — Gatekeeper")
exp = max(1, int(core_audit["expected"]))
k1, k2, k3, k4 = st.columns(4)
with k1: kpi("Present", f'{core_audit["present"]}/{exp}', status_cls(core_audit["present"]/exp))
with k2: kpi("Synced", f'{core_audit["synced"]}/{exp}', status_cls(core_audit["synced"]/exp))
with k3: kpi("Computable", f'{core_audit["computable"]}/{exp}', status_cls(core_audit["computable"]/exp))
with k4:
    md = core_audit["mode_date"]
    kpi("Mode Date", str(md).split()[0] if md is not None else "N/A", "status-green" if md is not None else "status-red")

dr = core_audit["drop"]
st.markdown(
    f'<span class="chip">missing: {dr["missing"]}</span>'
    f'<span class="chip">desync: {dr["desync"]}</span>'
    f'<span class="chip">short/NaN: {dr["short"]}</span>',
    unsafe_allow_html=True
)

health = core_audit["computable"] / exp
st.markdown(f'<div class="cert">[HEALTH] bench={bench} | window={window_label} | health={health:.1%} | modeDate={core_audit["mode_date"]}</div>', unsafe_allow_html=True)

if bench not in core_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced for this window. System Halted.")
    st.stop()

# -----------------------------
# STEP 1 — Market overview comment (deterministic)
# -----------------------------
st.markdown("---")
bench_ret = calc_return(core_close[bench], win_days)
comp = set(core_audit["computable_list"])
sec_returns = []
for sec, etf in sector_etf_map.items():
    if etf in comp:
        r = calc_return(core_close[etf], win_days)
        if not np.isnan(r):
            sec_returns.append(r)
leaders_ratio = float(np.mean([(r - bench_ret) > 0 for r in sec_returns])) if sec_returns else 0.0
dispersion = float(np.nanstd(sec_returns)) if sec_returns else 0.0
st.info(f"🌐 **{market_key} {bench_name} / {window_label}** — {market_comment(bench_ret, leaders_ratio, dispersion)}")

# -----------------------------
# STEP 2 — Sector comparison (horizontal, includes MARKET, click-to-drilldown)
# -----------------------------
st.markdown("---")
st.subheader("📊 Sector Comparison (click to drill down)")

rows = [{"Sector": f"MARKET ({bench})", "Ticker": bench, "Return": bench_ret, "Kind": "MARKET"}]
for sec, etf in sector_etf_map.items():
    if etf in comp:
        r = calc_return(core_close[etf], win_days)
        if not np.isnan(r):
            rows.append({"Sector": sec, "Ticker": etf, "Return": r, "Kind": "SECTOR"})

sec_df = pd.DataFrame(rows).sort_values("Return", ascending=True)
fig = px.bar(sec_df, x="Return", y="Sector", orientation="h", color="Kind", hover_data=["Ticker", "Return"],
             title=f"Market + Sector Returns — {window_label}")
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

selected_sector: Optional[str] = None
try:
    sel = getattr(event, "selection", None)
    if isinstance(sel, dict) and sel.get("points"):
        y = sel["points"][0].get("y", None)
        if isinstance(y, str) and y in sector_names:
            selected_sector = y
except Exception:
    selected_sector = None

if selected_sector is None:
    if st.session_state.get("selected_sector") in sector_names:
        selected_sector = st.session_state["selected_sector"]
    else:
        best = sec_df[sec_df["Kind"] == "SECTOR"].sort_values("Return", ascending=False)
        selected_sector = best.iloc[0]["Sector"] if not best.empty else sector_names[0]
st.session_state["selected_sector"] = selected_sector

selected_sector = st.selectbox("Selected sector", sector_names, index=sector_names.index(selected_sector))

# -----------------------------
# STEP 3 — Sector overview + stock table (Company name 중심) + AI analysis + News
# -----------------------------
st.markdown("---")
st.subheader("🔍 Sector Drilldown — Stocks (Company-first)")

sec_etf = sector_etf_map[selected_sector]
sec_ret = float(sec_df.loc[sec_df["Sector"] == selected_sector, "Return"].values[0]) if (sec_df["Sector"] == selected_sector).any() else np.nan
st.markdown(
    f"<div class='card'><div class='headline'>{sector_overview(selected_sector, sec_ret, bench_ret, health)}</div>"
    f"<div class='muted'>ETF={sec_etf} | Integrity gate enforced | Names-first display</div></div>",
    unsafe_allow_html=True
)

sector_stocks = cfg["stocks_by_sector"].get(selected_sector, [])
sector_expected = [bench] + list(dict.fromkeys(sector_stocks))

sector_key = f"sector::{market_key}::{selected_sector}"
need_sector_fetch = sync_btn or (st.session_state.get("sector_key") != sector_key) or (sector_key not in st.session_state)
if need_sector_fetch:
    with st.status(f"Sync: {selected_sector} stocks ...", expanded=False):
        raw_s = fetch_bulk_cached(tuple(sector_expected), period="12mo", chunk_size=80)
        close_s = extract_close_matrix(raw_s, expected=sector_expected)
        st.session_state[sector_key] = close_s
        st.session_state["sector_key"] = sector_key

sector_close: pd.DataFrame = st.session_state.get(sector_key, pd.DataFrame())
sector_audit = integrity_audit(sector_expected, sector_close, win_days)

sec_exp = max(1, int(sector_audit["expected"]))
sec_health = sector_audit["computable"] / sec_exp
st.markdown(
    f"<div class='cert'>[SECTOR HEALTH] sector={selected_sector} | expected={sec_exp} | computable={sector_audit['computable']} "
    f"| health={sec_health:.1%} | modeDate={sector_audit['mode_date']}</div>",
    unsafe_allow_html=True
)

if bench not in sector_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced inside this sector dataset. Halt.")
    st.stop()

computable_stocks = [t for t in sector_audit["computable_list"] if t != bench]
if len(computable_stocks) < 5:
    st.warning("⚠️ Not enough computable stocks after audit to rank for this window. Integrity-first: no imputation.")
    st.stop()

rank_df = compute_apex(sector_close, bench=bench, targets=sector_audit["computable_list"], win_days=win_days)
if rank_df.empty:
    st.error("❌ Ranking empty after computation.")
    st.stop()

# Company names (only for rows shown; cached)
rank_df["Company"] = rank_df["Ticker"].apply(get_company_name_cached)

# Deterministic "AI" analysis (reproducible)
rank_df["Reco"] = rank_df.apply(reco_bucket, axis=1)
top = rank_df.iloc[0]
top2 = rank_df.iloc[1] if len(rank_df) > 1 else None

st.subheader("🧠 AI Investment Analysis (deterministic)")
analysis_lines = []
analysis_lines.append(f"- Sector vs Market: **{sec_ret:+.2f}%** vs **{bench_ret:+.2f}%** (rel={sec_ret-bench_ret:+.2f}%)")
analysis_lines.append(f"- Integrity: sector health **{sec_health:.0%}** (computable={sector_audit['computable']}/{sec_exp})")
analysis_lines.append(f"- Leader: **{top['Company']}** (ApexScore {top['ApexScore']:.2f}, RS {top['RS']:+.2f}%, Accel {top['Accel']:+.2f}%, DD {top['MaxDD']:.2f}%, Stable {top['Stable']})")
if top2 is not None:
    analysis_lines.append(f"- Runner-up: **{top2['Company']}** (ApexScore {top2['ApexScore']:.2f})")
# risk notes
high_dd = float(np.nanpercentile(rank_df["MaxDD"].values, 75))
analysis_lines.append(f"- Risk lens: DD 75p={high_dd:.2f}% (names above this are structurally fragile in this window)")
st.markdown("<div class='card'>" + "<br>".join(analysis_lines) + "<div class='muted'>※This is an analytical instrument, not a guarantee. Uses only audited price data.</div></div>", unsafe_allow_html=True)

st.subheader("✅ AI Investment Recommendation (rules-based)")
strong = rank_df[rank_df["Reco"] == "STRONG"].head(5)
watch = rank_df[rank_df["Reco"] == "WATCH"].head(5)

def format_picks(df: pd.DataFrame) -> str:
    if df.empty:
        return "<span class='muted'>None</span>"
    items = []
    for _, r in df.iterrows():
        items.append(f"• <b>{r['Company']}</b> ({r['Ticker']}): Score {r['ApexScore']:.2f} | RS {r['RS']:+.2f}% | Accel {r['Accel']:+.2f}% | DD {r['MaxDD']:.2f}% {r['Stable']}")
    return "<br>".join(items)

st.markdown(f"<div class='card'><div class='headline'>STRONG (consider)</div>{format_picks(strong)}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='card'><div class='headline'>WATCH (monitor)</div>{format_picks(watch)}</div>", unsafe_allow_html=True)

# Next Actions
st.subheader("🎯 Next Strategic Actions")
for i, r in rank_df.head(3).iterrows():
    hint = "Continuation candidate" if r["Reco"] == "STRONG" else "Mixed signals: wait for confirmation"
    st.markdown(
        f"<div class='card'><div class='muted'>Rank #{i+1}</div>"
        f"<div style='font-size:20px;font-weight:900'>{r['Company']}</div>"
        f"<div class='muted'>{r['Ticker']} | ApexScore {r['ApexScore']:.2f} | RS {r['RS']:+.2f}% | Accel {r['Accel']:+.2f}% | MaxDD {r['MaxDD']:.2f}% | Vol {r['Vol(ann%)']:.1f}% | {r['Stable']}</div>"
        f"<div class='muted'>{hint}</div></div>",
        unsafe_allow_html=True
    )

# Tabs: table + correlation + news
t1, t2, t3 = st.tabs(["📊 Leaderboard (Company-first)", "📈 Mean Correlation", "🗞️ News (Sector + Leaders)"])

with t1:
    show = rank_df[["Company","Ticker","Reco","ApexScore","RS","Accel","MaxDD","Vol(ann%)","Stable"]].copy()
    for col in ["ApexScore","RS","Accel","MaxDD","Vol(ann%)"]:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    st.dataframe(show, use_container_width=True, height=560, hide_index=True)
    st.caption("Company-first. Ticker is secondary. No pandas Styler used.")

with t2:
    tick_list = [t for t in computable_stocks if t in sector_close.columns]
    rets = sector_close[tick_list].pct_change().tail(win_days)
    if rets.dropna(how="all").empty or len(tick_list) < 3:
        st.warning("Not enough return series to compute correlation.")
    else:
        corr = rets.corr()
        np.fill_diagonal(corr.values, np.nan)
        mean_corr = corr.mean(skipna=True).sort_values(ascending=False)
        fig2 = px.bar(mean_corr, orientation="h", title="Mean Correlation (excluding self)")
        st.plotly_chart(fig2, use_container_width=True)

with t3:
    # News sources: market ETF + sector ETF + top 3 stock tickers
    news_sources = [bench, sec_etf] + list(rank_df.head(3)["Ticker"].values)
    items = []
    for tk in news_sources:
        for n in get_news_cached(tk, max_items=10):
            title = n.get("title")
            link = n.get("link")
            pub = n.get("publisher") or ""
            tsec = n.get("providerPublishTime")
            if title and link:
                items.append({"title": title, "link": link, "publisher": pub, "ts": tsec, "source": tk})

    if not items:
        st.warning("No news returned by yfinance for the selected sources.")
    else:
        news_df = pd.DataFrame(items)
        # de-dup by title
        news_df = news_df.drop_duplicates(subset=["title"]).sort_values(by="ts", ascending=False, na_position="last").head(30)

        st.markdown("<div class='card'><div class='headline'>Sector Context (news feed)</div>"
                    "<div class='muted'>Sources = market ETF + sector ETF + top leaders. This is evidence layer for the instrument.</div></div>",
                    unsafe_allow_html=True)

        for _, r in news_df.iterrows():
            src_name = get_company_name_cached(r["source"]) if r["source"] not in [bench, sec_etf] else r["source"]
            st.markdown(f"- [{r['title']}]({r['link']})  <span class='muted'>({r['publisher']} / src:{src_name})</span>", unsafe_allow_html=True)