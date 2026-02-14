# app.py — AlphaLens v27.5 "Stability First"
# Fixes:
# - NameError(vol_annualized) eradicated (alias provided)
# - Market overview always shown at top (before charts)
# - Removed "weird bar" under title (no code/progress bar blocks)
# - Sector names JP in Japanese (TOPIX-17 style)
# - Plotly charts show numeric labels on bars
# - No pandas Styler / No matplotlib
# - Robust guards so it won't crash on empty/shape issues

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# =========================
# Page / Theme / Fonts
# =========================
st.set_page_config(page_title="AlphaLens", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;600;700&family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"]  {
  font-family: "IBM Plex Sans JP","Inter",system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans JP","Hiragino Kaku Gothic ProN","Meiryo",sans-serif !important;
}
.main { background-color:#0d1117; }
h1,h2,h3,h4 { letter-spacing: 0.2px; }
.brand { font-size:28px; font-weight:900; color:#e6edf3; margin:6px 0 2px 0; }
.subbrand { color:#8b949e; font-size:12px; margin:0 0 10px 0; }
.deck { background:#161b22; padding:12px 14px; border-radius:14px; border:1px solid #30363d; margin-bottom:10px; }
.card { background:#1c2128; border:1px solid #30363d; border-radius:14px; padding:12px 14px; margin-bottom:10px; }
.hr { height:1px; background:#30363d; margin:10px 0; }
.kpi { background:#11161d; border:1px solid #2b313a; border-radius:12px; padding:7px 10px; }
.kpi .t { font-size:11px; color:#8b949e; line-height:1.1; }
.kpi b { font-size:13px; color:#e6edf3; }
.status-green { border-left:5px solid #238636 !important; }
.status-yellow { border-left:5px solid #d29922 !important; }
.status-red { border-left:5px solid #da3633 !important; }
.cert { background:#0d1117; border:1px dashed #238636; border-radius:12px; padding:8px 10px; color:#7ee787;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; border:1px solid #30363d; font-size:12px; color:#c9d1d9; margin-right:6px; }
.badge-strong { border-color:#1f6feb; color:#cfe2ff; }
.badge-watch { border-color:#d29922; color:#ffe8a3; }
.badge-avoid { border-color:#da3633; color:#ffb4b4; }
.muted { color:#8b949e; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Lookbacks / Fetch horizon
# =========================
LOOKBACKS = {"1W (5d)": 5, "1M (21d)": 21, "3M (63d)": 63, "12M (252d)": 252}
FETCH_PERIOD = "24mo"

# =========================
# Bench / Sector ETFs
# =========================
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

# 日本はセクター名を日本語に統一（TOPIX-17の発想）
JP_TOPIX17_ETF_JA = {
    "エネルギー": "1617.T",
    "建設・資材": "1618.T",
    "素材・化学": "1619.T",
    "産業機械": "1620.T",
    "自動車・輸送機器": "1621.T",
    "商社・小売": "1622.T",
    "銀行": "1623.T",
    "金融（除く銀行）": "1624.T",
    "不動産": "1625.T",
    "情報通信": "1626.T",
    "サービス": "1627.T",
    "電力・ガス": "1628.T",
    "鉄鋼・非鉄": "1629.T",
    "機械": "1630.T",
    "電機・精密": "1631.T",
    "医薬品": "1632.T",
    "食品": "1633.T",
}

MARKETS = {
    "🇺🇸 米国": {"bench": US_BENCH, "bench_name": "S&P 500（SPY）", "sector_etf": US_SECTOR_ETF},
    "🇯🇵 日本": {"bench": JP_BENCH, "bench_name": "TOPIX（1306.T）", "sector_etf": JP_TOPIX17_ETF_JA},
}

# =========================
# Stock universes (extendable)
# ここは辞書を増やしていく前提。落ちない設計（chunk + gatekeeper）。
# =========================
US_STOCKS_BY_SECTOR: Dict[str, List[str]] = {
    "Technology": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTU","IBM","AMD","QCOM","TXN","ADI","MU","AMAT","LRCX","KLAC","SNPS","CDNS","NOW","PANW","CRWD","ANET","WDAY","ZS","NET","DDOG"],
    "Healthcare": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","DHR","ISRG","VRTX","BMY","GILD","PFE","REGN","SYK","BSX","MDT","ZTS","HCA","CI","ELV","CVS","HUM","BDX","EW","IQV"],
    "Financials": ["JPM","BAC","WFC","C","GS","MS","SCHW","BLK","AXP","COF","PNC","USB","TFC","MMC","AIG","MET","PRU","AFL","CB","ICE","SPGI","CME","BK","PGR","TRV","ALL","DFS","V","MA"],
    "Cons. Discretionary": ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","BKNG","TJX","ROST","GM","F","MAR","HLT","EBAY","CMG","YUM","LULU","DHI","LEN","RCL","ABNB","ORLY","AZO"],
    "Cons. Staples": ["PG","KO","PEP","WMT","COST","PM","MO","MDLZ","CL","KMB","GIS","KHC","KR","TGT","EL","HSY","STZ","KDP","WBA","MNST"],
    "Industrials": ["GE","CAT","DE","HON","UNP","UPS","RTX","LMT","BA","MMM","ETN","EMR","ITW","NSC","WM","FDX","NOC","GD","PCAR","ROK","CSX","ODFL","GWW","FAST","CTAS","AME","URI","DAL","LUV"],
    "Energy": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","KMI","HAL","BKR","DVN","HES","APA","FANG","WMB","TRGP","OKE","EQT"],
    "Materials": ["LIN","APD","SHW","ECL","FCX","NEM","DOW","DD","NUE","VMC","MLM","ALB","CF","MOS","IP","BALL","EMN","LYB","PPG"],
    "Utilities": ["NEE","DUK","SO","EXC","AEP","SRE","XEL","D","ED","PEG","EIX","PCG","AWK","WEC","ES","PPL","ETR","CMS","DTE"],
    "Real Estate": ["AMT","PLD","CCI","EQIX","SPG","O","PSA","WELL","DLR","AVB","EQR","VTR","IRM","VICI","SBAC","EXR","MAA","ARE","KIM","INVH"],
}

JP_STOCKS_BY_SECTOR_JA: Dict[str, List[str]] = {
    "エネルギー": ["1605.T","5020.T","5019.T"],
    "建設・資材": ["1801.T","1802.T","1803.T","1812.T","1925.T","1928.T","5201.T","5332.T"],
    "素材・化学": ["4005.T","4021.T","4042.T","4063.T","4188.T","4452.T","4631.T","4901.T","4911.T","3407.T","3402.T"],
    "産業機械": ["6301.T","6305.T","6367.T","6471.T","6473.T","6113.T","6273.T","6326.T","6361.T"],
    "自動車・輸送機器": ["7203.T","7267.T","6902.T","7201.T","7211.T","7270.T","7261.T","7272.T","9101.T","9104.T","9020.T","9022.T"],
    "商社・小売": ["8001.T","8002.T","8003.T","8031.T","8058.T","8053.T","3382.T","8267.T","9983.T","3092.T","7453.T"],
    "銀行": ["8306.T","8316.T","8411.T","8308.T","8331.T","8354.T","8355.T","7182.T","5831.T"],
    "金融（除く銀行）": ["8591.T","8604.T","8630.T","8725.T","8766.T","8750.T","8697.T"],
    "不動産": ["8801.T","8802.T","8830.T","3289.T","3003.T","3231.T"],
    "情報通信": ["9432.T","9433.T","9434.T","9984.T","4689.T","4755.T","6098.T","4385.T","3923.T","2413.T"],
    "サービス": ["4661.T","9735.T","9766.T","4324.T","2127.T","6028.T","3038.T","6183.T"],
    "電力・ガス": ["9501.T","9502.T","9503.T","9531.T","9532.T","9533.T"],
    "鉄鋼・非鉄": ["5401.T","5406.T","5411.T","5711.T","5713.T","5802.T","5801.T","3436.T"],
    "機械": ["6146.T","6268.T","6302.T","6331.T","6472.T","7004.T"],
    "電機・精密": ["8035.T","6857.T","6723.T","6920.T","7735.T","6963.T","6762.T","6861.T","6981.T","6758.T","6501.T","6702.T","6752.T","6954.T"],
    "医薬品": ["4502.T","4503.T","4507.T","4519.T","4568.T","4578.T","4587.T","4151.T"],
    "食品": ["2801.T","2802.T","2269.T","2914.T","2502.T","2503.T","2002.T","2201.T","2222.T"],
}

def get_stocks_by_sector(market_key: str) -> Dict[str, List[str]]:
    return US_STOCKS_BY_SECTOR if market_key == "🇺🇸 米国" else JP_STOCKS_BY_SECTOR_JA

# =========================
# Company Name DB (fast)
# =========================
NAME_DB: Dict[str, str] = {
    "SPY":"S&P500 ETF","XLK":"テクノロジーETF","XLV":"ヘルスケアETF","XLF":"金融ETF","XLY":"一般消費財ETF",
    "XLP":"生活必需品ETF","XLI":"資本財ETF","XLE":"エネルギーETF","XLB":"素材ETF","XLU":"公益ETF","XLRE":"不動産ETF",
    "1306.T":"TOPIX連動型上場投信",
    "1617.T":"TOPIX-17 エネルギー","1618.T":"TOPIX-17 建設・資材","1619.T":"TOPIX-17 素材・化学","1620.T":"TOPIX-17 産業機械",
    "1621.T":"TOPIX-17 自動車・輸送","1622.T":"TOPIX-17 商社・小売","1623.T":"TOPIX-17 銀行","1624.T":"TOPIX-17 金融（除く銀行）",
    "1625.T":"TOPIX-17 不動産","1626.T":"TOPIX-17 情報通信","1627.T":"TOPIX-17 サービス","1628.T":"TOPIX-17 電力・ガス",
    "1629.T":"TOPIX-17 鉄鋼・非鉄","1630.T":"TOPIX-17 機械","1631.T":"TOPIX-17 電機・精密","1632.T":"TOPIX-17 医薬品","1633.T":"TOPIX-17 食品",
    "8035.T":"東京エレクトロン","6857.T":"アドバンテスト","6920.T":"レーザーテック","6146.T":"ディスコ",
    "8306.T":"三菱UFJ FG","8316.T":"三井住友FG","8411.T":"みずほFG",
    "7203.T":"トヨタ自動車","9984.T":"ソフトバンクG","9983.T":"ファーストリテイリング","8031.T":"三井物産","8058.T":"三菱商事",
    "4661.T":"オリエンタルランド","6098.T":"リクルートHD","9432.T":"NTT","9433.T":"KDDI","9434.T":"ソフトバンク",
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMZN":"Amazon","TSLA":"Tesla","GOOGL":"Alphabet","META":"Meta",
    "LLY":"Eli Lilly","UNH":"UnitedHealth","JNJ":"J&J","JPM":"JPMorgan","BAC":"Bank of America","V":"Visa","MA":"Mastercard",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name(ticker: str) -> str:
    return NAME_DB.get(ticker, ticker)

# =========================
# Fetch engine (chunk + cache)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_bulk_cached(tickers: Tuple[str, ...], period: str = FETCH_PERIOD, chunk_size: int = 80) -> pd.DataFrame:
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
            if raw is not None and not raw.empty:
                frames.append(raw)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)

def extract_close(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            lv0 = set(df.columns.get_level_values(0))
            lv1 = set(df.columns.get_level_values(1))
            if "Close" in lv0:
                close = df.xs("Close", axis=1, level=0)
            elif "Close" in lv1:
                close = df.xs("Close", axis=1, level=1)
            else:
                return pd.DataFrame()
            close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            keep = [c for c in expected if c in close.columns]
            return close[keep]
        if "Close" in df.columns:
            return pd.DataFrame({expected[0]: pd.to_numeric(df["Close"], errors="coerce")}).dropna(how="all")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# =========================
# Analytics / Gatekeeper
# =========================
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    mu = float(s.mean(skipna=True))
    sig = float(s.std(ddof=0, skipna=True))
    if sig == 0.0 or np.isnan(sig):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sig

def calc_return(series: pd.Series, win: int) -> float:
    s = series.tail(win + 1)
    if len(s) < win + 1 or s.isna().any():
        return np.nan
    return float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0)

def max_dd(series: pd.Series, win: int) -> float:
    s = series.tail(win + 1)
    if len(s) < win + 1 or s.isna().any():
        return np.nan
    dd = (s / s.cummax() - 1.0) * 100.0
    return float(abs(dd.min()))

def vol_ann(series: pd.Series, win: int) -> float:
    s = series.tail(win + 1)
    if len(s) < win + 1 or s.isna().any():
        return np.nan
    r = s.pct_change().dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=0) * np.sqrt(252) * 100.0)

# 互換用：あなたのログに出た関数名を必ず定義（NameError根絶）
def vol_annualized(series: pd.Series, win: int) -> float:
    return vol_ann(series, win)

def rs_stable(asset: pd.Series, bench: pd.Series) -> str:
    a = asset.tail(6).dropna()
    b = bench.tail(6).dropna()
    if len(a) < 6 or len(b) < 6:
        return "⚠️"
    rs_short = (a.iloc[-1] / a.iloc[0] - 1.0) - (b.iloc[-1] / b.iloc[0] - 1.0)
    return "✅" if np.sign(rs_short) != 0 else "⚠️"

def audit_gatekeeper(expected: List[str], close_df: pd.DataFrame, win: int) -> Dict[str, Any]:
    exp = len(expected)
    if close_df is None or close_df.empty:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0, "mode_date": None, "computable_list": []}
    present = [t for t in expected if t in close_df.columns]
    if not present:
        return {"expected": exp, "present": 0, "synced": 0, "computable": 0, "mode_date": None, "computable_list": []}
    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    synced_mask = (last_dates == mode_date)
    computable = []
    for t in present:
        if not bool(synced_mask.get(t, False)):
            continue
        tail = close_df[t].tail(win + 1)
        if len(tail) < win + 1 or tail.isna().any():
            continue
        computable.append(t)
    return {
        "expected": exp, "present": len(present), "synced": int(synced_mask.sum()),
        "computable": len(computable), "mode_date": mode_date, "computable_list": computable
    }

def cls_ratio(r: float) -> str:
    if r >= 0.9: return "status-green"
    if r >= 0.7: return "status-yellow"
    return "status-red"

def kpi(label: str, value: str, ratio: float):
    st.markdown(f"<div class='kpi {cls_ratio(ratio)}'><div class='t'>{label}</div><b>{value}</b></div>", unsafe_allow_html=True)

def market_comment(bench_ret: float, leaders_ratio: float, dispersion: float) -> str:
    if np.isnan(bench_ret):
        return "指数の計算に必要なデータが不足しています。"
    if bench_ret > 0 and leaders_ratio > 0.55:
        tone = "リスクオン（上昇基調）"
    elif bench_ret < 0 and leaders_ratio < 0.45:
        tone = "リスクオフ（下落基調）"
    else:
        tone = "混在（方向感は限定的）"
    rot = "ローテーション活発" if dispersion >= 2.0 else "ローテーション小"
    breadth = "上昇の裾野が広い" if leaders_ratio >= 0.6 else "リーダー集中（幅が狭い）" if leaders_ratio <= 0.4 else "裾野は中立"
    return f"{tone} / {breadth} / {rot}（指数 {bench_ret:+.2f}%・勝ちセクター比 {leaders_ratio:.0%}・分散 {dispersion:.2f}）"

def sector_outlook(sec_name: str, sec_ret: float, bench_ret: float, cohesion: float, dispersion: float) -> str:
    if np.isnan(sec_ret) or np.isnan(bench_ret):
        return f"{sec_name}：データ不足により短期見通しの生成を保留します。"
    rel = sec_ret - bench_ret
    stance = "先行（強い）" if rel > 0.8 else "遅行（弱い）" if rel < -0.8 else "概ね市場並み"
    coh = "まとまり強" if (not np.isnan(cohesion) and cohesion >= 0.55) else "まとまり弱" if (not np.isnan(cohesion) and cohesion <= 0.35) else "中立"
    dis = "選別余地大" if (not np.isnan(dispersion) and dispersion >= 2.0) else "選別余地小" if (not np.isnan(dispersion) and dispersion <= 1.0) else "中立"
    return f"{sec_name}は直近{stance}（セクター {sec_ret:+.2f}% / 市場 {bench_ret:+.2f}% / 相対 {rel:+.2f}%）。セクター内は{coh}・{dis}。当面は『上位銘柄の持続性（Stable）』と『DD（下振れ耐性）』で取捨選別が有効です。"

def compute_rank(close_df: pd.DataFrame, bench: str, targets: List[str], win: int) -> pd.DataFrame:
    b = close_df[bench]
    b_ret = calc_return(b, win)
    rows = []
    for t in targets:
        if t == bench:
            continue
        s = close_df[t]
        p_ret = calc_return(s, win)
        if np.isnan(p_ret) or np.isnan(b_ret):
            continue
        rs = p_ret - b_ret
        half = max(1, win // 2)
        seg = s.tail(half + 1)
        if len(seg) < half + 1 or seg.isna().any():
            continue
        p_half = float((seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0)
        accel = p_half - (p_ret / 2.0)
        dd = max_dd(s, win)
        vol = vol_annualized(s, win)
        stable = rs_stable(s, b)
        rows.append({"Ticker": t, "RS": rs, "Accel": accel, "MaxDD": dd, "Vol": vol, "Stable": stable, "Ret": p_ret})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["RS_z"] = zscore(df["RS"])
    df["Accel_z"] = zscore(df["Accel"])
    df["DD_z"] = zscore(df["MaxDD"])
    df["ApexScore"] = 0.60 * df["RS_z"] + 0.25 * df["Accel_z"] - 0.15 * df["DD_z"]
    df = df.sort_values("ApexScore", ascending=False, na_position="last").reset_index(drop=True)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker: str, max_items: int = 8) -> List[dict]:
    try:
        news = yf.Ticker(ticker).news or []
        return news[:max_items]
    except Exception:
        return []

def build_news_panel(sources: List[str]) -> pd.DataFrame:
    items = []
    for src in sources:
        for n in get_news(src, max_items=8):
            title = n.get("title")
            link = n.get("link")
            pub = n.get("publisher") or ""
            ts = n.get("providerPublishTime")
            if title and link:
                items.append({"title": title, "link": link, "publisher": pub, "ts": ts, "source": src})
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items).drop_duplicates(subset=["title"])
    df = df.sort_values("ts", ascending=False, na_position="last").head(30)
    return df

# =========================
# App Header + Command Deck
# =========================
st.markdown("<div class='brand'>AlphaLens</div>", unsafe_allow_html=True)
st.markdown("<div class='subbrand'>嘘をつかない計器 × 地合い→セクター→個別（日本語）</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='deck'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.1, 1.0, 0.9])
    with c1:
        market_key = st.selectbox("地域", list(MARKETS.keys()), index=0)
    with c2:
        window_label = st.selectbox("期間", list(LOOKBACKS.keys()), index=1)
    with c3:
        st.write("")
        sync_btn = st.button("SYNC", use_container_width=True, type="primary")
    st.caption(f"データ取得期間: {FETCH_PERIOD}（12M計算の安定化バッファ）")
    if st.session_state.get("ts"):
        st.caption(f"最終同期: {st.session_state.ts}")
    st.markdown("</div>", unsafe_allow_html=True)

win = LOOKBACKS[window_label]
cfg = MARKETS[market_key]
bench = cfg["bench"]
bench_name = cfg["bench_name"]
sector_etf = cfg["sector_etf"]
sector_names = list(sector_etf.keys())
stocks_by_sector = get_stocks_by_sector(market_key)

# =========================
# STEP 0 — Core sync (bench + sector ETFs)
# =========================
core_expected = [bench] + list(sector_etf.values())
core_key = f"core::{market_key}"

need_core = sync_btn or ("core_close" not in st.session_state) or (st.session_state.get("core_key") != core_key)
if need_core:
    with st.status("市場・セクターETFを同期中…", expanded=False):
        raw = fetch_bulk_cached(tuple(core_expected), period=FETCH_PERIOD, chunk_size=80)
        close = extract_close(raw, core_expected)
        st.session_state.core_close = close
        st.session_state.core_key = core_key
        st.session_state.ts = datetime.now().strftime("%H:%M:%S")

core_close: pd.DataFrame = st.session_state.get("core_close", pd.DataFrame())
core_audit = audit_gatekeeper(core_expected, core_close, win)
exp = max(1, int(core_audit["expected"]))
health = core_audit["computable"] / exp

# Gatekeeper（小さめ配置）
g1, g2, g3, g4, g5 = st.columns([0.9, 0.9, 0.9, 1.1, 2.2])
with g1: kpi("Present", f'{core_audit["present"]}/{exp}', core_audit["present"]/exp)
with g2: kpi("Synced", f'{core_audit["synced"]}/{exp}', core_audit["synced"]/exp)
with g3: kpi("Computable", f'{core_audit["computable"]}/{exp}', core_audit["computable"]/exp)
with g4:
    md = core_audit["mode_date"]
    kpi("Mode Date", str(md).split()[0] if md is not None else "N/A", 1.0 if md is not None else 0.0)
with g5:
    st.markdown(f"<div class='cert'>[GATEKEEPER] 健全度 {health:.0%} / bench={bench}</div>", unsafe_allow_html=True)

if bench not in core_audit["computable_list"]:
    st.error(f"❌ ベンチマーク（{bench}）がこの期間で計算不能です。Gatekeeperにより停止します。")
    st.stop()

# =========================
# STEP 1 — 市況概要（必ず最初に出す）
# =========================
bench_ret = calc_return(core_close[bench], win)
sec_rows = []
sec_rets = []
for sec, etf in sector_etf.items():
    if etf in core_audit["computable_list"]:
        r = calc_return(core_close[etf], win)
        if not np.isnan(r):
            sec_rows.append({"Sector": sec, "Ticker": etf, "Return": r, "RS": r - bench_ret})
            sec_rets.append(r)

leaders_ratio = float(np.mean([(r - bench_ret) > 0 for r in sec_rets])) if sec_rets and not np.isnan(bench_ret) else 0.0
dispersion = float(np.nanstd(sec_rets)) if sec_rets else 0.0

st.info(f"🌐 **{market_key} / {bench_name} / {window_label}** — {market_comment(bench_ret, leaders_ratio, dispersion)}")

# =========================
# STEP 2 — Sector bar (horizontal + gradient + numeric labels + MARKET)
# =========================
st.subheader("📊 セクター比較（相対強度RS）")
sec_df = pd.DataFrame(sec_rows)
if sec_df.empty:
    st.warning("セクターETFの計算可能データが不足しています。SYNCを押すか、期間を短くしてください。")
    st.stop()

market_row = pd.DataFrame([{"Sector": "MARKET", "Ticker": bench, "Return": bench_ret, "RS": 0.0}])
sec_df2 = pd.concat([sec_df, market_row], ignore_index=True).sort_values("RS", ascending=True)
sec_df2["RS_text"] = sec_df2["RS"].map(lambda x: f"{x:+.2f}%")
sec_df2["Return_text"] = sec_df2["Return"].map(lambda x: f"{x:+.2f}%")

fig = px.bar(
    sec_df2,
    x="RS", y="Sector", orientation="h",
    color="RS", color_continuous_scale="RdYlGn",
    text="RS_text",
    hover_data={"Ticker": True, "Return_text": True, "RS_text": True, "RS": False, "Return": False},
    title=f"RS（セクター − 市場） / {window_label}",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
st.plotly_chart(fig, use_container_width=True)

# セクター選択（確実に動くUI）
st.markdown("<div class='muted'>セクターを選ぶと、この下に『動向・見通し → 視覚化 → 銘柄 → AI合議 → ニュース』が展開されます。</div>", unsafe_allow_html=True)
selected = st.selectbox("セクター選択", sector_names, index=0)

# =========================
# STEP 3 — Sector Drilldown
# =========================
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.subheader("🔍 セクター詳細")

sec_etf = sector_etf[selected]
sec_ret = float(sec_df.loc[sec_df["Sector"] == selected, "Return"].values[0]) if (sec_df["Sector"] == selected).any() else np.nan

sector_stocks = stocks_by_sector.get(selected, [])
sector_expected = [bench] + list(dict.fromkeys([t for t in sector_stocks if isinstance(t, str) and t.strip()]))
sector_key = f"sector::{market_key}::{selected}::{window_label}"

need_sector = sync_btn or (sector_key not in st.session_state)
if need_sector:
    with st.status(f"{selected} の銘柄を同期中…", expanded=False):
        raw_s = fetch_bulk_cached(tuple(sector_expected), period=FETCH_PERIOD, chunk_size=80)
        close_s = extract_close(raw_s, sector_expected)
        st.session_state[sector_key] = close_s

sector_close: pd.DataFrame = st.session_state.get(sector_key, pd.DataFrame())
sector_audit = audit_gatekeeper(sector_expected, sector_close, win)
sec_exp = max(1, int(sector_audit["expected"]))
sec_health = sector_audit["computable"] / sec_exp

st.markdown(f"<div class='card'><b>セクターETF:</b> {get_company_name(sec_etf)}（{sec_etf}）<br><span class='muted'>計算可能 {sector_audit['computable']}/{sec_exp}（健全度 {sec_health:.0%}）</span></div>", unsafe_allow_html=True)

if bench not in sector_audit["computable_list"]:
    st.error("❌ セクター内データでベンチマークが計算不能です（Gatekeeper停止）。")
    st.stop()

computable_stocks = [t for t in sector_audit["computable_list"] if t != bench]
if len(computable_stocks) < 5:
    st.warning("⚠️ 監査を通過した銘柄が少なすぎます。辞書の銘柄数を増やすか、期間を短くしてください。")
    st.stop()

rets = sector_close[computable_stocks].pct_change().tail(win).dropna(how="all")
cohesion = np.nan
if isinstance(rets, pd.DataFrame) and not rets.empty and rets.shape[1] >= 3:
    corr = rets.corr()
    if isinstance(corr, pd.DataFrame) and not corr.empty and corr.shape[0] == corr.shape[1] and corr.shape[0] >= 2:
        vals = corr.values.astype(float, copy=True)
        di = np.diag_indices_from(vals)
        vals[di] = np.nan
        cohesion = float(np.nanmean(vals))
disp = float(np.nanstd([calc_return(sector_close[t], win) for t in computable_stocks])) if computable_stocks else np.nan

st.markdown(f"<div class='card'><b>直近動向・見通し</b><br><span class='muted'>{sector_outlook(selected, sec_ret, bench_ret, cohesion, disp)}</span></div>", unsafe_allow_html=True)

rank_df = compute_rank(sector_close, bench=bench, targets=sector_audit["computable_list"], win=win)
if rank_df.empty:
    st.error("❌ スコア計算結果が空です（データ不足/欠損）。")
    st.stop()

rank_df["企業名"] = rank_df["Ticker"].apply(get_company_name)
rank_df["判定"] = np.where((rank_df["RS"] > 0) & (rank_df["Accel"] > 0) & (rank_df["Stable"] == "✅") & (rank_df["MaxDD"] <= 10), "強い",
                  np.where((rank_df["RS"] > 0) & (rank_df["Accel"] >= 0), "注視", "回避"))

st.subheader("📌 視覚化（セクター内の勢力図）")
top10 = rank_df.head(10).copy().sort_values("ApexScore", ascending=True)
top10["Score_text"] = top10["ApexScore"].map(lambda x: f"{x:.2f}")
fig_top = px.bar(
    top10,
    x="ApexScore", y="企業名", orientation="h",
    color="ApexScore", color_continuous_scale="Viridis",
    text="Score_text",
    hover_data=["Ticker","RS","Accel","MaxDD","Vol","Stable","判定"],
    title="ApexScore Top10（企業名）",
)
fig_top.update_traces(textposition="outside", cliponaxis=False)
fig_top.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
st.plotly_chart(fig_top, use_container_width=True)

fig_scatter = px.scatter(
    rank_df,
    x="MaxDD", y="RS",
    color="ApexScore",
    hover_data=["企業名","Ticker","Accel","Vol","Stable","判定"],
    title="RS × MaxDD（右上＝強い、左上＝強くて耐性）",
)
st.plotly_chart(fig_scatter, use_container_width=True)

leaders = rank_df.head(3)["Ticker"].tolist()
news_df = build_news_panel([bench, sec_etf] + leaders)

st.subheader("🧠 AI合議（投資推奨）")
top = rank_df.iloc[0]
st.markdown(f"<div class='card'><b>第一候補:</b> {top['企業名']}（{top['Ticker']}）<br><span class='muted'>ApexScore {top['ApexScore']:.2f} / RS {top['RS']:+.2f}% / Accel {top['Accel']:+.2f} / DD {top['MaxDD']:.2f}% / Vol {top['Vol']:.1f}% / {top['Stable']}</span></div>", unsafe_allow_html=True)

with st.expander("スコアの意味・注釈", expanded=True):
    st.markdown(
        "- **ApexScore**：RS（相対強度）/ Accel（加速）/ MaxDD（最大DD）を標準化して合成した相対順位スコア。\n"
        "- **RS（%）**：銘柄リターン − 市場（ベンチマーク）リターン。\n"
        "- **Accel**：直近半分窓が強いほどプラス（需給の加速）。\n"
        "- **MaxDD（%）**：窓内最大ドローダウン（小さいほど耐性）。\n"
        "- **Vol（%）**：年率ボラ（大きいほどブレが大きい）。\n"
        "- **Stable**：直近の相対方向が維持されているかの簡易判定。"
    )

t1, t2 = st.tabs(["📊 銘柄テーブル（企業名中心）", "🗞️ ニュース（根拠）"])
with t1:
    show = rank_df[["企業名","Ticker","判定","ApexScore","RS","Accel","MaxDD","Vol","Stable"]].copy()
    for col in ["ApexScore","RS","Accel","MaxDD","Vol"]:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(2)
    st.dataframe(show, use_container_width=True, height=560, hide_index=True)
with t2:
    if news_df is None or news_df.empty:
        st.warning("ニュースが取得できませんでした（yfinance側の欠落/制限の可能性）。")
    else:
        for _, r in news_df.iterrows():
            src = r.get("source", "")
            src_label = get_company_name(src) if isinstance(src, str) else ""
            st.markdown(f"- [{r['title']}]({r['link']}) <span class='muted'>({r['publisher']} / src:{src_label})</span>", unsafe_allow_html=True)