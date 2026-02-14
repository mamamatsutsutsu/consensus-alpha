import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Sentinel Absolute v26.3.6", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .reportview-container { background: #0d1117; }
    .command-deck { background: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #1f6feb; margin-bottom: 20px; }
    .stMetric { background: #1c2128; border-radius: 8px; padding: 12px; border-left: 5px solid #30363d; margin-bottom: 10px; }
    .status-green { border-left: 5px solid #238636 !important; }
    .status-yellow { border-left: 5px solid #d29922 !important; }
    .status-red { border-left: 5px solid #da3633 !important; }
    .audit-cert { background: #0d1117; border: 1px dashed #238636; padding: 10px; border-radius: 8px; color: #7ee787; font-family: monospace; font-size: 12px; }
    .next-action-card { background: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE LOGIC
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data_robust(tickers_tuple, period="4mo"):
    return yf.download(list(tickers_tuple), period=period, interval="1d", group_by="ticker", progress=False, auto_adjust=True, threads=True)

def extract_close_matrix(df, expected_list):
    """MultiIndex/Single両対応の決定論的抽出"""
    close = pd.DataFrame()
    if df is None or df.empty: return close
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if "Close" in df.columns.get_level_values(1): close = df.xs("Close", axis=1, level=1)
            elif "Close" in df.columns.get_level_values(0): close = df.xs("Close", axis=1, level=0)
        except: pass
    else:
        if "Close" in df.columns: close = pd.DataFrame({expected_list[0]: df["Close"]})
        else: close = df[[c for c in expected_list if c in df.columns]]
    if not close.empty:
        close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return close[[c for c in expected_list if c in close.columns]]

def get_integrity(expected_list, close_df, win_days):
    present = [t for t in expected_list if t in close_df.columns]
    if not present: return {"present":0,"computable":0,"synced":0,"expected":len(expected_list),"computable_list":[],"date":None}
    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    mode_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    synced_mask = (last_dates == mode_date)
    comp_list = [t for t in present if close_df[t].tail(win_days+1).notna().sum() >= (win_days+1) and bool(synced_mask.get(t,False))]
    return {"present":len(present),"computable":len(comp_list),"synced":int(synced_mask.sum()),"expected":len(expected_list),"computable_list":comp_list,"date":mode_date}

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.std(ddof=0) == 0: return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

# ==========================================
# 3. UI & MAIN
# ==========================================
UNIVERSES = {
    "US Sectors": {"bench": "SPY", "tickers": ["XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLE", "XLB", "XLU", "XLRE"]},
    "JP Industry": {"bench": "1306.T", "tickers": ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T", "1624.T"]}
}

def main():
    st.title("🛰️ Sentinel Absolute v26.3.6")

    # --- COMMAND DECK ---
    with st.container():
        st.markdown('<div class="command-deck">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: market = st.selectbox("Market", list(UNIVERSES.keys()), key="m_sel")
        with c2: lookback = st.selectbox("Window", ["1W (5d)", "1M (21d)", "3M (63d)"], index=1, key="w_sel")
        with c3:
            st.write("")
            sync_btn = st.button("SYNC NOW", use_container_width=True, type="primary")
        if 'ts' in st.session_state: st.caption(f"Last Intelligence Update: {st.session_state.ts}")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- INITIALIZATION LOGIC ---
    u_cfg = UNIVERSES[market]
    win_days = int(lookback.split("(")[1].replace("d)", ""))
    ordered_all = [u_cfg["bench"]] + u_cfg["tickers"]

    # 初回起動時またはボタン押下時に強制実行
    if sync_btn or 'close_df' not in st.session_state or st.session_state.get('last_m') != market:
        with st.status("Gathering Tactical Data...", expanded=False):
            try:
                raw = fetch_data_robust(tuple(ordered_all))
                st.session_state.close_df = extract_close_matrix(raw, ordered_all)
                st.session_state.last_m = market
                st.session_state.ts = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                st.error(f"Sync Failed: {e}")
                return

    # --- RENDER ---
    close_df = st.session_state.get('close_df', pd.DataFrame())
    if close_df.empty:
        st.warning("No data available. Please click SYNC NOW.")
        return

    audit = get_integrity(ordered_all, close_df, win_days)
    
    # 監査パネル
    st.subheader("🛡️ Integrity Audit")
    exp = audit["expected"]
    a1, a2, a3, a4 = st.columns(4)
    def get_c(v, e): return "status-green" if v/e >= 0.9 else "status-yellow" if v/e >= 0.7 else "status-red"
    with a1: st.markdown(f'<div class="stMetric {get_c(audit["present"],exp)}"><small>Present</small><br><b>{audit["present"]}/{exp}</b></div>', unsafe_allow_html=True)
    with a2: st.markdown(f'<div class="stMetric {get_c(audit["computable"],exp)}"><small>Computable</small><br><b>{audit["computable"]}/{exp}</b></div>', unsafe_allow_html=True)
    with a3: st.markdown(f'<div class="stMetric {get_c(audit["synced"],exp)}"><small>Synced</small><br><b>{audit["synced"]}/{exp}</b></div>', unsafe_allow_html=True)
    with a4: st.markdown(f'<div class="stMetric status-green"><small>ModeDate</small><br><b>{str(audit["date"]).split()[0] if audit["date"] else "N/A"}</b></div>', unsafe_allow_html=True)

    # 分析実行
    bench = u_cfg["bench"]
    if bench not in audit["computable_list"]:
        st.error(f"Benchmark {bench} is missing or out of sync. Change Market or Window.")
        return

    # 計算対象の抽出
    valid = [t for t in u_cfg["tickers"] if t in audit["computable_list"]]
    if not valid:
        st.warning("No tickers passed the integrity audit.")
        return

    # 指標計算
    b_s = close_df[bench].tail(win_days+1)
    b_ret = (b_s.iloc[-1] / b_s.iloc[0] - 1) * 100
    
    res = []
    for t in valid:
        s = close_df[t].tail(win_days+1)
        p_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
        rs = p_ret - b_ret
        half = max(1, win_days // 2)
        p_half = (s.iloc[-1] / s.iloc[-(half+1)] - 1) * 100
        dd = abs(((s / s.cummax() - 1) * 100).min())
        
        # Stable判定
        s6, b6 = close_df[t].tail(6).dropna(), close_df[bench].tail(6).dropna()
        stable = "✅" if (len(s6)>=6 and len(b6)>=6 and np.sign(rs) == np.sign(((s6.iloc[-1]/s6.iloc[0]-1)*100)-((b6.iloc[-1]/b6.iloc[0]-1)*100))) else "⚠️"
        res.append({"Ticker": t, "RS": rs, "Accel": p_half - (p_ret/2), "MaxDD": dd, "Stable": stable})

    res_df = pd.DataFrame(res)
    res_df["ApexScore"] = (0.6 * zscore(res_df["RS"])) + (0.25 * zscore(res_df["Accel"])) - (0.15 * zscore(res_df["MaxDD"]))
    res_df = res_df.sort_values("ApexScore", ascending=False)

    # --- TACTICAL DISPLAY ---
    st.markdown("---")
    st.subheader("🎯 Tactical Priority")
    cols = st.columns(3)
    for i, (_, row) in enumerate(res_df.head(3).iterrows()):
        with cols[i]:
            st.markdown(f'<div class="next-action-card"><small>Rank #{i+1} | Score {row["ApexScore"]:.2f}</small><br>'
                        f'<b>{row["Ticker"]}</b> {row["Stable"]}<br><span style="color:#8b949e; font-size:12px;">RS: {row["RS"]:.1f}%</span></div>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["📊 Leaderboard", "📈 Correlation"])
    with t1:
        st.dataframe(res_df[["Ticker", "ApexScore", "RS", "Accel", "MaxDD", "Stable"]], 
                     column_config={"RS": st.column_config.ProgressColumn("RS", format="%.1f%%", min_value=-15, max_value=15),
                                    "ApexScore": st.column_config.NumberColumn("Score", format="%.2f")},
                     use_container_width=True, hide_index=True)
    with t2:
        rets = close_df[valid].pct_change().tail(win_days)
        avg_corr = rets.corr().replace(1.0, np.nan).mean()
        st.plotly_chart(px.bar(avg_corr, title="Sector Internal Correlation"), use_container_width=True)

if __name__ == "__main__":
    main()