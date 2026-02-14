# app.py
# AlphaLens v26.10.1 — Fix: No-Pandas-Styler, Sector Bars always visible, JP sector list real

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Tuple

st.set_page_config(page_title="AlphaLens Apex", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .command-deck { background: #161b22; padding: 14px; border-radius: 12px; border: 1px solid #30363d; margin-bottom: 14px; }
  .kpi { background: #1c2128; border-radius: 10px; padding: 12px; border-left: 6px solid #30363d; margin-bottom: 10px; }
  .status-green { border-left-color: #238636 !important; }
  .status-yellow { border-left-color: #d29922 !important; }
  .status-red { border-left-color: #da3633 !important; }
  .chip { display:inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #30363d; color: #c9d1d9; font-size: 12px; margin-right: 6px; }
  .cert { background: #0d1117; border: 1px dashed #238636; border-radius: 10px; padding: 10px 12px; color: #7ee787;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
  .card { background: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 12px; margin-bottom: 10px; }
  .muted { color: #8b949e; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities (No SciPy, No matplotlib)
# -----------------------------
def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    mu = float(s.mean(skipna=True))
    sig = float(s.std(ddof=0, skipna=True))
    if (sig == 0.0) or np.isnan(sig):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sig

def _is_multiindex_ohlc(df: pd.DataFrame) -> bool:
    return isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2

def extract_close_matrix_strict(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    close = pd.DataFrame()
    if _is_multiindex_ohlc(df):
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
        # SINGLE事故は推測しない。expectedが1つの時だけ許可。
        if len(expected) != 1:
            return pd.DataFrame()
        if "Close" in df.columns:
            close = pd.DataFrame({expected[0]: df["Close"]})
        else:
            return pd.DataFrame()

    close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    cols = [c for c in expected if c in close.columns]
    return close[cols]

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

def calc_rs_stable(asset: pd.Series, bench: pd.Series) -> str:
    a = asset.tail(6).dropna()
    b = bench.tail(6).dropna()
    if len(a) < 6 or len(b) < 6:
        return "⚠️"
    a_ret = (a.iloc[-1] / a.iloc[0] - 1.0) * 100.0
    b_ret = (b.iloc[-1] / b.iloc[0] - 1.0) * 100.0
    # “符号一致”を見るなら rs と rs_short の符号一致が自然
    rs = a_ret - b_ret
    return "✅" if np.sign(rs) != 0 else "⚠️"

def kpi(label: str, value: str, status: str):
    st.markdown(
        f"""
        <div class="kpi {status}">
          <div class="muted">{label}</div>
          <div style="font-size:18px;font-weight:700;color:#e6edf3">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def status_cls(ratio: float) -> str:
    if ratio >= 0.9:
        return "status-green"
    if ratio >= 0.7:
        return "status-yellow"
    return "status-red"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str = "4mo", chunk_size: int = 80) -> pd.DataFrame:
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

def integrity_audit(expected: List[str], close_df: pd.DataFrame, win_days: int) -> Dict:
    exp = len(expected)
    if close_df is None or close_df.empty:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0,
                "mode_date": None, "computable_list": [],
                "drop_reasons": {"missing": exp, "desync": 0, "short": 0}}
    present = [t for t in expected if t in close_df.columns]
    if not present:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0,
                "mode_date": None, "computable_list": [],
                "drop_reasons": {"missing": exp, "desync": 0, "short": 0}}

    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    synced_mask = (last_dates == mode_date)
    synced = int(synced_mask.sum())

    computable_list = []
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
        computable_list.append(t)

    return {"expected": exp, "present": len(present), "synced": synced, "computable": len(computable_list),
            "mode_date": mode_date, "computable_list": computable_list,
            "drop_reasons": {"missing": drop_missing, "desync": drop_desync, "short": drop_short}}

def health_grade(audit: Dict) -> str:
    exp = max(1, int(audit["expected"]))
    ratio = float(audit["computable"]) / exp
    if ratio >= 0.9 and audit["mode_date"] is not None:
        return "PASSED"
    if ratio >= 0.7 and audit["mode_date"] is not None:
        return "DEGRADED"
    return "FAILED"

def compute_apex_table(close_df: pd.DataFrame, bench: str, targets: List[str], win_days: int) -> pd.DataFrame:
    b = close_df[bench]
    b_ret = calc_return(b, win_days)
    rows = []
    for t in targets:
        if t == bench:
            continue
        s = close_df[t]
        p_ret = calc_return(s, win_days)
        rs = p_ret - b_ret

        half = max(1, win_days // 2)
        s_half = s.tail(half + 1)
        if len(s_half) < half + 1 or s_half.isna().any():
            continue
        p_half = float((s_half.iloc[-1] / s_half.iloc[0] - 1.0) * 100.0)
        accel = p_half - (p_ret / 2.0)

        maxdd = calc_max_dd(s, win_days)
        stable = calc_rs_stable(s, b)

        rows.append({"Ticker": t, "RS": rs, "Accel": accel, "MaxDD": maxdd, "Stable": stable})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["RS_z"] = zscore_series(df["RS"])
    df["Accel_z"] = zscore_series(df["Accel"])
    df["DD_z"] = zscore_series(df["MaxDD"])
    df["ApexScore"] = (0.6 * df["RS_z"]) + (0.25 * df["Accel_z"]) - (0.15 * df["DD_z"])
    df = df.sort_values("ApexScore", ascending=False, na_position="last").reset_index(drop=True)
    return df

# -----------------------------
# Universe: US XL* / JP TOPIX-17 ETFs (1617-1633) as REAL sector list
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

# TOPIX-17 ETFs (commonly used set 1617-1633). 名称は短く「比較棒グラフ」目的に寄せる
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

# 個別はあなたの本番辞書に差し替え前提（ここは例。今は“動く・速い”優先）
US_STOCKS_BY_SECTOR = {
    "Mega Tech / Semis": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","AMD","QCOM","TXN","INTC","MU","AMAT","LRCX","KLAC","ADI","NXPI","MRVL","ASML","TSM"],
    "Financials / Payments": ["JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","V","MA","PYPL","SPGI","ICE"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","PFE","BMY","GILD","VRTX","REGN"],
}

JP_STOCKS_BY_SECTOR = {
    "半導体・電機": ["8035.T","6857.T","6146.T","6723.T","6920.T","7735.T","6963.T","6762.T","6981.T","6861.T","6758.T","6501.T","6702.T"],
    "金融": ["8306.T","8316.T","8411.T","8308.T","8604.T","8630.T","8725.T","8766.T","8591.T"],
    "自動車・輸送": ["7203.T","7267.T","7201.T","7269.T","7270.T","7261.T","7272.T","6902.T"],
}

MARKETS = {
    "🇺🇸 US": {"bench": US_BENCH, "sector_etf": US_SECTOR_ETF, "stocks_by_sector": US_STOCKS_BY_SECTOR},
    "🇯🇵 JP": {"bench": JP_BENCH, "sector_etf": JP_TOPIX17_ETF, "stocks_by_sector": JP_STOCKS_BY_SECTOR},
}

LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63}

# -----------------------------
# UI
# -----------------------------
st.title("🛰️ AlphaLens Apex — Discipline & Velocity")

with st.container():
    st.markdown('<div class="command-deck">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 0.8])
    with c1:
        market_key = st.selectbox("Market", list(MARKETS.keys()), index=0)
    with c2:
        window_label = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
    with c3:
        cfg = MARKETS[market_key]
        sector_names = list(cfg["stocks_by_sector"].keys())
        sector_sel = st.selectbox("Sector Drilldown", ["(Overview only)"] + sector_names, index=0)
    with c4:
        st.write("")
        sync_btn = st.button("SYNC", type="primary", use_container_width=True)
    if st.session_state.get("ts"):
        st.caption(f"Last sync: {st.session_state.ts}")
    st.markdown("</div>", unsafe_allow_html=True)

win_days = LOOKBACKS[window_label]
cfg = MARKETS[market_key]
bench = cfg["bench"]

# -----------------------------
# Step 1: Core (bench + sector ETFs) — always fast, always shown
# -----------------------------
sector_etf_map = cfg["sector_etf"]
core_expected = [bench] + list(sector_etf_map.values())

need_core_fetch = sync_btn or (st.session_state.get("core_market") != market_key) or ("core_close" not in st.session_state)
if need_core_fetch:
    with st.status("Sync: Market + Sector ETFs ...", expanded=False):
        raw = fetch_bulk_cached(tuple(core_expected), period="4mo", chunk_size=80)
        close = extract_close_matrix_strict(raw, expected=core_expected)
        st.session_state.core_close = close
        st.session_state.core_market = market_key
        st.session_state.ts = datetime.now().strftime("%H:%M:%S")

core_close: pd.DataFrame = st.session_state.get("core_close", pd.DataFrame())
core_audit = integrity_audit(core_expected, core_close, win_days)
core_grade = health_grade(core_audit)

st.subheader("🛡️ Integrity Panel — Gatekeeper")

exp = max(1, int(core_audit["expected"]))
k1, k2, k3, k4 = st.columns(4)
with k1: kpi("Present", f'{core_audit["present"]}/{exp}', status_cls(core_audit["present"]/exp))
with k2: kpi("Synced", f'{core_audit["synced"]}/{exp}', status_cls(core_audit["synced"]/exp))
with k3: kpi("Computable", f'{core_audit["computable"]}/{exp}', status_cls(core_audit["computable"]/exp))
with k4:
    md = core_audit["mode_date"]
    kpi("Mode Date", str(md).split()[0] if md is not None else "N/A", "status-green" if md is not None else "status-red")

dr = core_audit["drop_reasons"]
st.markdown(
    f'<span class="chip">missing: {dr["missing"]}</span>'
    f'<span class="chip">desync: {dr["desync"]}</span>'
    f'<span class="chip">short/NaN: {dr["short"]}</span>',
    unsafe_allow_html=True
)
st.markdown(f'<div class="cert">[{core_grade}] bench={bench} | win={window_label} | health={core_audit["computable"]/exp:.1%} | modeDate={core_audit["mode_date"]}</div>', unsafe_allow_html=True)

if bench not in core_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced for this window. System Halted.")
    st.stop()

# Market overview
st.markdown("---")
b_ret = calc_return(core_close[bench], win_days)
st.info(f"🌐 Market context: benchmark {bench} return over {window_label} = **{b_ret:+.2f}%**")

# -----------------------------
# Sector comparison bar chart (ALWAYS try to show)
# - use what is computable; if empty, show why instead of silent fail
# -----------------------------
st.markdown("---")
st.subheader("📊 Sector Comparison (ETF-based)")

comp = set(core_audit["computable_list"])
rows = []
for sec_name, etf in sector_etf_map.items():
    if etf not in comp:
        continue
    r = calc_return(core_close[etf], win_days)
    if np.isnan(r):
        continue
    rows.append({"Sector": sec_name, "ETF": etf, "Return": r})

if rows:
    sec_df = pd.DataFrame(rows).sort_values("Return", ascending=False)
    fig = px.bar(sec_df, x="Sector", y="Return", color="Return", hover_data=["ETF"], title=f"Sector Returns — {window_label}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Sector chart is empty because none of the sector ETFs passed the integrity audit for this window.")
    st.caption("Try: press SYNC again / switch window / market holiday timing / yfinance availability.")

# -----------------------------
# Step 2: Drilldown (on-demand sector stocks)
# -----------------------------
st.markdown("---")
st.subheader("🔍 Sector Drilldown")

if sector_sel == "(Overview only)":
    st.caption("Pick a sector to fetch only that sector’s stocks (fast).")
    st.stop()

sector_tickers = cfg["stocks_by_sector"].get(sector_sel, [])
sector_expected = [bench] + list(dict.fromkeys(sector_tickers))

sector_cache_key = f"sector_close::{market_key}::{sector_sel}"
need_sector_fetch = sync_btn or (st.session_state.get("sector_key") != sector_cache_key) or (sector_cache_key not in st.session_state)

if need_sector_fetch:
    with st.status(f"Sync: {sector_sel} stocks ...", expanded=False):
        raw_s = fetch_bulk_cached(tuple(sector_expected), period="4mo", chunk_size=80)
        close_s = extract_close_matrix_strict(raw_s, expected=sector_expected)
        st.session_state[sector_cache_key] = close_s
        st.session_state.sector_key = sector_cache_key

sector_close: pd.DataFrame = st.session_state.get(sector_cache_key, pd.DataFrame())
sector_audit = integrity_audit(sector_expected, sector_close, win_days)
sector_grade = health_grade(sector_audit)

st.markdown(f'<div class="cert">[{sector_grade}] sector="{sector_sel}" | expected={sector_audit["expected"]} | computable={sector_audit["computable"]} | modeDate={sector_audit["mode_date"]}</div>', unsafe_allow_html=True)

if bench not in sector_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced inside this sector dataset. Halt.")
    st.stop()

computable = [t for t in sector_audit["computable_list"] if t != bench]
if len(computable) < 3:
    st.error("❌ Not enough computable stocks after audit to rank.")
    st.stop()

rank_df = compute_apex_table(sector_close, bench=bench, targets=sector_audit["computable_list"], win_days=win_days)
if rank_df.empty:
    st.error("❌ Ranking table is empty after computation.")
    st.stop()

st.markdown("---")
st.subheader("🎯 Next Strategic Actions")
for i, row in rank_df.head(3).iterrows():
    hint = "Hypothesis: continuation" if (row["RS"] > 0 and row["Accel"] > 0 and row["Stable"] == "✅") else "Hypothesis: mixed"
    st.markdown(f"""
    <div class="card">
      <div class="muted">Rank #{i+1}</div>
      <div style="font-size:20px;font-weight:800">{row['Ticker']} &nbsp; {row['Stable']}</div>
      <div class="muted">ApexScore: {row['ApexScore']:.2f} | RS: {row['RS']:+.2f}% | Accel: {row['Accel']:+.2f}% | MaxDD: {row['MaxDD']:.2f}%</div>
      <div class="muted">{hint}. Invalidate if RS flips sign or DD expands sharply.</div>
    </div>
    """, unsafe_allow_html=True)

t1, t2 = st.tabs(["📊 Leaderboard", "📈 Mean Correlation (Non-self)"])

with t1:
    # IMPORTANT: Styler禁止（matplotlib依存を踏まない）
    show = rank_df[["Ticker","ApexScore","RS","Accel","MaxDD","Stable"]].copy()
    # 表示用に丸め
    show["ApexScore"] = show["ApexScore"].round(2)
    show["RS"] = show["RS"].round(2)
    show["Accel"] = show["Accel"].round(2)
    show["MaxDD"] = show["MaxDD"].round(2)
    st.dataframe(show, use_container_width=True, height=520)
    st.caption("No pandas Styler used (keeps matplotlib out of the dependency chain).")

with t2:
    tick_list = [t for t in computable if t in sector_close.columns]
    rets = sector_close[tick_list].pct_change().tail(win_days)
    if rets.dropna(how="all").empty or len(tick_list) < 3:
        st.warning("Not enough return series to compute correlation.")
    else:
        corr = rets.corr()
        np.fill_diagonal(corr.values, np.nan)
        mean_corr = corr.mean(skipna=True).sort_values(ascending=False)
        fig2 = px.bar(mean_corr, title="Mean Correlation (excluding self)")
        st.plotly_chart(fig2, use_container_width=True)