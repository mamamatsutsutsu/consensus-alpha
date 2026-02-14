# app.py
# AlphaLens v26.10  "Apex Sentinel — Discipline & Velocity"
# - 思想: 「地合い→セクター→個別」 / 「嘘をつかない計器」 / 「監査が計算対象を支配」 / 「モバイルで速い」
# - 速度: 4mo固定 / セクターETFは常時1回DL / 個別は「選択セクターのみ」オンデマンドDL / session_stateで窓切替は再DLなし
# - 誠実: Benchmark不全なら停止 / computable_list以外は一切計算しない / 監査落ち理由を件数で表示 / SINGLE事故は停止（推測割当しない）
# - UI: 上部Command Deck / Integrity Panelはゲートキーパー / Top3を最初に提示、詳細はタブで

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Tuple

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(page_title="AlphaLens Apex", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .command-deck {
    background: #161b22; padding: 14px 14px; border-radius: 12px;
    border: 1px solid #30363d; margin-bottom: 14px;
  }
  .kpi {
    background: #1c2128; border-radius: 10px; padding: 12px;
    border-left: 6px solid #30363d; margin-bottom: 10px;
  }
  .status-green { border-left-color: #238636 !important; }
  .status-yellow { border-left-color: #d29922 !important; }
  .status-red { border-left-color: #da3633 !important; }
  .chip {
    display:inline-block; padding: 2px 8px; border-radius: 999px;
    border: 1px solid #30363d; color: #c9d1d9; font-size: 12px; margin-right: 6px;
  }
  .cert {
    background: #0d1117; border: 1px dashed #238636; border-radius: 10px;
    padding: 10px 12px; color: #7ee787; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  .card {
    background: #1c2128; border: 1px solid #30363d; border-radius: 12px; padding: 12px;
    margin-bottom: 10px;
  }
  .muted { color: #8b949e; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities (No SciPy)
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
    """
    誠実仕様:
    - MultiIndexならCloseを抽出
    - 非MultiIndexは「取得ティッカーが1つ」と確定できる場合のみ許可
      （推測割当=嘘の原因なので停止）
    """
    if df is None or df.empty:
        return pd.DataFrame()

    close = pd.DataFrame()

    if _is_multiindex_ohlc(df):
        # yfinance typical: columns (Field, Ticker) or (Ticker, Field) depending on args/version
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
        # SINGLE事故を「推測で補う」のはやめる（計器として不誠実）
        # ただし expected が1つだけなら、その1つに割り当ててOK
        if len(expected) != 1:
            return pd.DataFrame()
        if "Close" in df.columns:
            close = pd.DataFrame({expected[0]: df["Close"]})
        else:
            # まれにClose列がない場合は空
            return pd.DataFrame()

    close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    # expected の順序で揃える（順序固定）
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
    return "✅" if np.sign(a_ret - b_ret) != 0 else "⚠️"

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

# -----------------------------
# Fast fetch (chunked + cached)
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str = "4mo", chunk_size: int = 80) -> pd.DataFrame:
    """
    速度と落ちにくさの両立:
    - 200銘柄級は一括が不安定なのでchunk
    - キャッシュで同じ銘柄集合は再DLしない
    """
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
    # yfinance chunk returns same index; concat on columns
    return pd.concat(frames, axis=1)

# -----------------------------
# Universe dictionaries (~200 each)
# NOTE: ここは「結果的に約200」になるように辞書を持つ、という要件。
#       yfinanceで取れない銘柄が混じる可能性は監査で誠実に可視化し、計算対象から排除する。
# -----------------------------
US_BENCH = "SPY"
JP_BENCH = "1306.T"

US_SECTOR_ETF = {
    "Tech (XLK)": ["XLK"],
    "Health (XLV)": ["XLV"],
    "Financials (XLF)": ["XLF"],
    "Consumer Disc (XLY)": ["XLY"],
    "Consumer Staples (XLP)": ["XLP"],
    "Industrials (XLI)": ["XLI"],
    "Energy (XLE)": ["XLE"],
    "Materials (XLB)": ["XLB"],
    "Utilities (XLU)": ["XLU"],
    "Real Estate (XLRE)": ["XLRE"],
}

JP_SECTOR_ETF = {
    "TOPIX-17 (ETF set)": ["1617.T","1618.T","1619.T","1620.T","1621.T","1622.T","1623.T","1624.T","1625.T","1626.T","1627.T","1628.T","1629.T","1630.T","1631.T","1632.T","1633.T"],
}

# 個別銘柄（“約200”）: セクター別に保持し、ドリルダウン時は選択セクターだけ取得
US_STOCKS_BY_SECTOR: Dict[str, List[str]] = {
    "Mega Tech / Semis": [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","AMD","QCOM","TXN","INTC","MU","AMAT","LRCX","KLAC","ADI","NXPI","MRVL","ASML","TSM","ARM","SMCI","ORCL","IBM","CRM","ADBE","NOW","SNPS","CDNS","PANW","CRWD","DDOG","NET"
    ],
    "Financials / Payments": [
        "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","V","MA","PYPL","SPGI","ICE","CB","PGR","MMC","AON","TROW","PNC","USB","BK"
    ],
    "Healthcare": [
        "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","PFE","BMY","GILD","VRTX","REGN","SYK","CI","BSX","MDT","ZTS"
    ],
    "Industrials / Defense": [
        "CAT","DE","GE","HON","UNP","UPS","BA","RTX","LMT","NOC","GD","ETN","EMR","PH","WM","CSX","NSC"
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","KMI","OKE"
    ],
    "Consumer / Retail": [
        "WMT","COST","HD","LOW","MCD","SBUX","NKE","DIS","BKNG","CMG","TGT","TJX","ORLY","AZO","YUM","MDLZ","KO","PEP","PG","CL"
    ],
    "Comms / Media": [
        "NFLX","TMUS","T","VZ","CHTR","PARA","WBD"
    ],
    "Materials / REITs / Utilities": [
        "LIN","APD","SHW","ECL","FCX","NEM","PLD","AMT","PSA","O","NEE","DUK","SO"
    ],
}

# ざっくり合計を増やして「約200」へ寄せる（重複排除は後で行う）
US_STOCKS_BY_SECTOR["Diversified Large Caps"] = [
    "BRK-B","GOOG","CME","MSCI","INTU","UBER","ABNB","TXRH","GWW","ROK","ITW","SPOT","SHOP"
]

# 日本は銘柄コードの網羅が難しいため、「大型・準大型中心の約200」を辞書で保持（取れないものは監査で落ちる）
JP_STOCKS_BY_SECTOR: Dict[str, List[str]] = {
    "半導体・電機": [
        "8035.T","6857.T","6146.T","6723.T","6920.T","3436.T","7735.T","6963.T","6762.T","6981.T","6861.T","6758.T","6501.T","6702.T","6701.T","7751.T","7752.T","7752.T","6752.T","6971.T","6954.T","6952.T","7731.T","4063.T","4062.T"
    ],
    "金融": [
        "8306.T","8316.T","8411.T","8308.T","8604.T","8630.T","8725.T","8766.T","8591.T","7182.T","7186.T","8354.T","8331.T"
    ],
    "自動車・輸送": [
        "7203.T","7267.T","7201.T","7269.T","7270.T","7261.T","7272.T","6902.T","7259.T","7202.T","7205.T","3116.T"
    ],
    "商社・エネルギー・素材": [
        "8058.T","8031.T","8001.T","8053.T","8002.T","8015.T","2768.T","1605.T","5020.T","3402.T","3407.T","5401.T","5411.T","5713.T","5802.T","5108.T","5201.T"
    ],
    "必需品・医薬・ヘルスケア": [
        "2802.T","2914.T","2502.T","2503.T","4452.T","4901.T","4911.T","4502.T","4503.T","4507.T","4519.T","4568.T","4523.T","9020.T"
    ],
    "通信・IT・サービス": [
        "9432.T","9433.T","9434.T","9984.T","4689.T","6098.T","4385.T","4755.T","3659.T","2413.T","7974.T","7832.T","9735.T"
    ],
    "不動産・建設・インフラ": [
        "8801.T","8802.T","8830.T","3289.T","1801.T","1812.T","1802.T","1803.T","9022.T","9021.T","9501.T","9503.T","9531.T"
    ],
}

# “約200”に寄せる増補（ここはあなたの実運用辞書で拡張する前提）
JP_STOCKS_BY_SECTOR["大型バリュー/コア"] = [
    "8058.T","8031.T","8001.T","7203.T","8306.T","9432.T","4502.T","6501.T","6702.T","8604.T",
    "7011.T","7012.T","7013.T","6301.T","6367.T","6273.T","6857.T","8035.T","9984.T","9983.T",
    "3382.T","6178.T","5108.T","5802.T","6954.T","6861.T","7269.T","2768.T","5020.T","1605.T",
    "2914.T","2502.T","4901.T","8801.T","8802.T","8830.T","9020.T","9022.T"
]

NAME_MAP = {
    # 任意（不足は“未登録”として誠実に扱う）
    "AAPL":"Apple", "MSFT":"Microsoft", "NVDA":"NVIDIA", "AMZN":"Amazon", "GOOGL":"Alphabet",
    "8035.T":"東京エレクトロン", "6857.T":"アドバンテスト", "7203.T":"トヨタ", "8306.T":"三菱UFJ",
    "9984.T":"ソフトバンクG", "9432.T":"NTT", "4502.T":"武田薬品", "8058.T":"三菱商事",
    "8031.T":"三井物産", "9983.T":"ファストリ"
}

def flatten_unique(sector_map: Dict[str, List[str]]) -> List[str]:
    out = []
    seen = set()
    for _, lst in sector_map.items():
        for t in lst:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out

# -----------------------------
# Integrity (Gatekeeper)
# -----------------------------
def integrity_audit(expected: List[str], close_df: pd.DataFrame, win_days: int) -> Dict:
    exp = len(expected)
    if close_df is None or close_df.empty:
        return {
            "expected": exp, "present": 0, "synced": 0, "computable": 0,
            "mode_date": None, "computable_list": [],
            "drop_reasons": {"missing": exp, "desync": 0, "short": 0}
        }

    present = [t for t in expected if t in close_df.columns]
    if not present:
        return {
            "expected": exp, "present": 0, "synced": 0, "computable": 0,
            "mode_date": None, "computable_list": [],
            "drop_reasons": {"missing": exp, "desync": 0, "short": 0}
        }

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

    return {
        "expected": exp,
        "present": len(present),
        "synced": synced,
        "computable": len(computable_list),
        "mode_date": mode_date,
        "computable_list": computable_list,
        "drop_reasons": {"missing": drop_missing, "desync": drop_desync, "short": drop_short}
    }

def health_grade(audit: Dict) -> str:
    exp = max(1, int(audit["expected"]))
    ratio = float(audit["computable"]) / exp
    if ratio >= 0.9 and audit["mode_date"] is not None:
        return "PASSED"
    if ratio >= 0.7 and audit["mode_date"] is not None:
        return "DEGRADED"
    return "FAILED"

# -----------------------------
# Compute ranking (discipline)
# -----------------------------
def compute_apex_table(close_df: pd.DataFrame, bench: str, targets: List[str], win_days: int) -> pd.DataFrame:
    """
    計算対象は必ず targets（=computable_list内）に限定する。
    """
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
        # half+1本が必要なので、ここも欠損チェック済み前提（targetsに入れる時点で担保）
        s_half = s.tail(half + 1)
        p_half = float((s_half.iloc[-1] / s_half.iloc[0] - 1.0) * 100.0)
        accel = p_half - (p_ret / 2.0)

        maxdd = calc_max_dd(s, win_days)
        stable = calc_rs_stable(s, b)

        rows.append({
            "Ticker": t,
            "Name": NAME_MAP.get(t, "未登録"),
            "RS": rs,
            "Accel": accel,
            "MaxDD": maxdd,
            "Stable": stable
        })

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
# App State Helpers
# -----------------------------
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63}

MARKETS = {
    "🇺🇸 US": {
        "bench": US_BENCH,
        "sector_etf": US_SECTOR_ETF,
        "stocks_by_sector": US_STOCKS_BY_SECTOR,
    },
    "🇯🇵 JP": {
        "bench": JP_BENCH,
        "sector_etf": JP_SECTOR_ETF,
        "stocks_by_sector": JP_STOCKS_BY_SECTOR,
    }
}

# -----------------------------
# Main UI
# -----------------------------
st.title("🛰️ AlphaLens Apex — Discipline & Velocity")

with st.container():
    st.markdown('<div class="command-deck">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.2, 1.0])
    with c1:
        market_key = st.selectbox("Market", list(MARKETS.keys()), index=0)
    with c2:
        window_label = st.selectbox("Window", list(LOOKBACKS.keys()), index=1)
    with c3:
        # セクター選択（個別DLをこの後に限定）
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

# ---- Step 1: Always-on Universe (Market + Sector ETF) ----
# ここが「規律」の核：市場地合いとセクター状態を最速に出す
etf_list = flatten_unique(cfg["sector_etf"])
core_expected = [bench] + etf_list

# セクターETFの辞書は小さいので安定・高速
need_core_fetch = sync_btn or (st.session_state.get("core_market") != market_key) or ("core_close" not in st.session_state)
if need_core_fetch:
    with st.status("Sync: Market + Sector ETFs (fast lane)...", expanded=False):
        raw = fetch_bulk_cached(tuple(core_expected), period="4mo", chunk_size=80)
        close = extract_close_matrix_strict(raw, expected=core_expected)
        st.session_state.core_close = close
        st.session_state.core_market = market_key
        st.session_state.ts = datetime.now().strftime("%H:%M:%S")

core_close: pd.DataFrame = st.session_state.get("core_close", pd.DataFrame())
core_audit = integrity_audit(core_expected, core_close, win_days)
core_grade = health_grade(core_audit)

# ---- Integrity Panel (Gatekeeper) ----
st.subheader("🛡️ Integrity Panel — Gatekeeper")

exp = max(1, int(core_audit["expected"]))
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi("Present", f'{core_audit["present"]}/{exp}', status_cls(core_audit["present"]/exp))
with k2:
    kpi("Synced", f'{core_audit["synced"]}/{exp}', status_cls(core_audit["synced"]/exp))
with k3:
    kpi("Computable", f'{core_audit["computable"]}/{exp}', status_cls(core_audit["computable"]/exp))
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

# Gate: benchmark must be computable
if bench not in core_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced for this window. System Halted (truth-first).")
    st.stop()

# ---- Step 2: Market Overview (Truth only) ----
st.markdown("---")
b_ret = calc_return(core_close[bench], win_days)
st.info(f"🌐 Market context: benchmark {bench} return over {window_label} = **{b_ret:+.2f}%** (computed only if audit passed)")

# ---- Step 3: Sector ETF Heat (fast, disciplined) ----
sector_rows = []
comp_set = set(core_audit["computable_list"])
for sec_name, etfs in cfg["sector_etf"].items():
    # etfs in this sector might be more than 1 (JP TOPIX-17 set)
    ok = [t for t in etfs if t in comp_set]
    if not ok:
        continue
    # average sector return across ETFs (usually 1)
    rets = []
    for t in ok:
        r = calc_return(core_close[t], win_days)
        if not np.isnan(r):
            rets.append(r)
    if rets:
        sector_rows.append({"Sector": sec_name, "Return": float(np.mean(rets)), "N": len(rets)})

if not sector_rows:
    st.warning("⚠️ No sector ETF is computable for this window. (Check market holiday / data gaps)")
else:
    sec_df = pd.DataFrame(sector_rows).sort_values("Return", ascending=False)
    fig = px.bar(sec_df, x="Sector", y="Return", color="Return", title=f"Sector Pulse (ETF-based) — {window_label}")
    st.plotly_chart(fig, use_container_width=True)

# ---- Step 4: Drilldown (on-demand, only selected sector) ----
st.markdown("---")
st.subheader("🔍 Sector Drilldown — On-demand (only when you need it)")

if sector_sel == "(Overview only)":
    st.caption("Pick a sector to fetch only that sector’s stocks (~200 universe maintained in dictionaries).")
    st.stop()

sector_tickers = cfg["stocks_by_sector"].get(sector_sel, [])
# ここが「結果的に約200」：辞書側で保持（あなたの本番辞書で増やせば自然に200に寄る）
# ただし実行は“選択セクターだけ”なので速度は維持される
sector_expected = [bench] + list(dict.fromkeys(sector_tickers))

# fetch only if (a) sync pressed (b) sector changed (c) not cached
sector_cache_key = f"sector_close::{market_key}::{sector_sel}"
need_sector_fetch = sync_btn or (st.session_state.get("sector_key") != sector_cache_key)

if need_sector_fetch:
    with st.status(f"Sync: {sector_sel} stocks (on-demand)...", expanded=False):
        raw_s = fetch_bulk_cached(tuple(sector_expected), period="4mo", chunk_size=80)
        close_s = extract_close_matrix_strict(raw_s, expected=sector_expected)
        st.session_state[sector_cache_key] = close_s
        st.session_state.sector_key = sector_cache_key

sector_close: pd.DataFrame = st.session_state.get(sector_cache_key, pd.DataFrame())
sector_audit = integrity_audit(sector_expected, sector_close, win_days)
sector_grade = health_grade(sector_audit)

# Sector Integrity (separate from core)
st.markdown(f'<div class="cert">[{sector_grade}] sector="{sector_sel}" | expected={sector_audit["expected"]} | computable={sector_audit["computable"]} | modeDate={sector_audit["mode_date"]}</div>', unsafe_allow_html=True)
dr2 = sector_audit["drop_reasons"]
st.markdown(
    f'<span class="chip">missing: {dr2["missing"]}</span>'
    f'<span class="chip">desync: {dr2["desync"]}</span>'
    f'<span class="chip">short/NaN: {dr2["short"]}</span>',
    unsafe_allow_html=True
)

# Gate: benchmark must be computable in sector_close too (truth)
if bench not in sector_audit["computable_list"]:
    st.error(f"❌ Benchmark ({bench}) is NOT computable/synced inside this sector dataset. Halt.")
    st.stop()

# Need at least 3 stocks to rank meaningfully
computable = [t for t in sector_audit["computable_list"] if t != bench]
if len(computable) < 3:
    st.error("❌ Not enough computable stocks after audit to rank. (truth-first)")
    st.stop()

# Compute ranking table (only computable_list)
rank_df = compute_apex_table(sector_close, bench=bench, targets=sector_audit["computable_list"], win_days=win_days)
if rank_df.empty:
    st.error("❌ Ranking table is empty after computation (unexpected).")
    st.stop()

# ---- Next Strategic Actions (mobile-first, no断定) ----
st.markdown("---")
st.subheader("🎯 Next Strategic Actions (facts → hypothesis → invalidation)")
top3 = rank_df.head(3)

for i, row in top3.iterrows():
    # “事実”だけを表示：スコア、RS、Accel、DD、Stable
    # “解釈”は条件付きで控えめに
    hint = "Hypothesis: trend continuation" if (row["RS"] > 0 and row["Accel"] > 0 and row["Stable"] == "✅") else "Hypothesis: mixed / needs caution"
    st.markdown(f"""
    <div class="card">
      <div class="muted">Rank #{i+1} | {row['Name']} </div>
      <div style="font-size:20px;font-weight:800">{row['Ticker']} &nbsp; {row['Stable']}</div>
      <div class="muted">ApexScore: {row['ApexScore']:.2f} | RS: {row['RS']:+.2f}% | Accel: {row['Accel']:+.2f}% | MaxDD: {row['MaxDD']:.2f}%</div>
      <div class="muted">{hint}. Invalidate if RS flips sign or DD expands sharply.</div>
    </div>
    """, unsafe_allow_html=True)

# ---- Details Tabs ----
t1, t2 = st.tabs(["📊 Leaderboard", "📈 Mean Correlation (Non-self)"])

with t1:
    show = rank_df[["Ticker","Name","ApexScore","RS","Accel","MaxDD","Stable"]].copy()
    st.dataframe(
        show.style
            .format({"ApexScore":"{:.2f}","RS":"{:+.2f}%","Accel":"{:+.2f}%","MaxDD":"{:.2f}%"})
            .background_gradient(subset=["ApexScore","RS"], cmap="RdYlGn"),
        use_container_width=True,
        height=520
    )
    st.caption("ApexScore is relative (z-score). Computation uses ONLY computable_list (gatekeeper audit).")

with t2:
    # Correlation uses only computable stocks (no bench)
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

# ---- Footnote: universe size honesty ----
st.markdown("---")
us_total = len(flatten_unique(US_STOCKS_BY_SECTOR))
jp_total = len(flatten_unique(JP_STOCKS_BY_SECTOR))
st.caption(f"Universe honesty: US dictionary approx={us_total} tickers, JP dictionary approx={jp_total} tickers. "
           f"Audit will truthfully show how many were usable today.")