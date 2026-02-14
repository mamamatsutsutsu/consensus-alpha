import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime

# ==========================================
# CONFIGURATION & THEME
# ==========================================
st.set_page_config(page_title="Sentinel Prime v26.3.4", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .reportview-container { background: #0d1117; }
    .command-deck { background: #161b22; padding: 15px; border-radius: 12px; border: 1px solid #1f6feb; margin-bottom: 20px; }
    .stMetric { background: #1c2128; border-radius: 8px; padding: 12px; border-left: 5px solid #30363d; margin-bottom: 10px; }
    .status-green { border-left: 5px solid #238636 !important; }
    .status-yellow { border-left: 5px solid #d29922 !important; }
    .status-red { border-left: 5px solid #da3633 !important; }
    .next-action-card { background: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
    .audit-cert { background: #0d1117; padding: 10px; border-radius: 8px; font-family: monospace; font-weight: bold; }
    .cert-passed { border: 1px solid #238636; color: #238636; }
    .cert-degraded { border: 1px solid #d29922; color: #d29922; }
    .cert-failed { border: 1px solid #da3633; color: #da3633; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DETERMINISTIC CORE ENGINE
# ==========================================

@st.cache_data(ttl=1800)
def fetch_data_cached(tickers_tuple, period="4mo"):
    """
    auto_adjust=True により分配金・分割を考慮した誠実なデータを取得。
    threads=True で速度も確保。
    """
    return yf.download(
        list(tickers_tuple), 
        period=period, 
        interval="1d", 
        group_by='ticker', 
        progress=False,
        auto_adjust=True,
        threads=True
    )

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.isna().all() or s.std(ddof=0) == 0: return pd.Series([0.0]*len(s), index=s.index)
    return (s - s.mean()) / s.std(ddof=0)

def extract_close_robust(df, expected_cols):
    """
    SINGLE事故を決定論的に回避。
    expected_cols[0]（常にベンチマーク）を優先割り当てする誠実な設計。
    """
    close = pd.DataFrame()
    if df is None or df.empty: return close

    if isinstance(df.columns, pd.MultiIndex):
        try:
            lv0 = set(df.columns.get_level_values(0))
            lv1 = set(df.columns.get_level_values(1))
            if "Close" in lv0: close = df.xs("Close", axis=1, level=0)
            elif "Close" in lv1: close = df.xs("Close", axis=1, level=1)
        except: pass
    else:
        # 非MultiIndex時の推測をベンチマークに固定
        if "Close" in df.columns:
            close = pd.DataFrame({expected_cols[0]: df["Close"]})
        else:
            # カラム名がそのままティッカーの場合
            close = df[[c for c in expected_cols if c in df.columns]]

    close = close.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    keep = [c for c in expected_cols if c in close.columns]
    return close[keep]

def calculate_integrity(expected_list, close_df, win_days):
    present = [t for t in expected_list if t in close_df.columns]
    if not present:
        return {"present":0,"computable":0,"synced":0,"expected":len(expected_list),
                "most_common_date":None, "computable_list":[]}

    last_dates = close_df[present].apply(lambda x: x.last_valid_index())
    most_common_date = last_dates.mode().iloc[0] if not last_dates.mode().empty else None
    synced_mask = (last_dates == most_common_date)
    
    computable_list = []
    for t in present:
        tail = close_df[t].tail(win_days + 1)
        if tail.notna().sum() >= (win_days + 1) and bool(synced_mask.get(t, False)):
            computable_list.append(t)

    return {
        "present": len(present), "computable": len(computable_list),
        "synced": int(synced_mask.sum()), "expected": len(expected_list),
        "most_common_date": most_common_date, "computable_list": computable_list
    }

def kpi_card(label, value, status_cls):
    st.markdown(f'<div class="stMetric {status_cls}"><div style="font-size:12px;color:#8b949e">{label}</div>'
                f'<div style="font-size:18px;font-weight:700;color:#e6edf3">{value}</div></div>', unsafe_allow_html=True)

# ==========================================
# MAIN COMMAND CENTER
# ==========================================
UNIVERSES = {
    "US Sectors": {"bench": "SPY", "tickers": ["XLK", "XLV", "XLF", "XLY", "XLP", "XLI", "XLE", "XLB", "XLU", "XLRE"]},
    "JP Industry": {"bench": "1306.T", "tickers": ["1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T", "1623.T", "1624.T"]}
}

def main():
    st.title("🛰️ Sentinel Prime v26.3.4")

    # Command Deck
    with st.container():
        st.markdown('<div class="command-deck">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: market = st.selectbox("Market Universe", list(UNIVERSES.keys()))
        with c2: lookback = st.selectbox("Analysis Window", ["1W (5d)", "1M (21d)", "3M (63d)"], index=1)
        with c3:
            st.write("")
            sync_btn = st.button("SYNC ENGINE", use_container_width=True, type="primary")
        
        # 最終同期時刻の表示
        if 'last_sync_ts' in st.session_state:
            st.caption(f"Last Sync: {st.session_state.last_sync_ts}")
        st.markdown('</div>', unsafe_allow_html=True)

    win_days = int(lookback.split("(")[1].replace("d)", ""))
    u_cfg = UNIVERSES[market]
    
    # 決定論的な順序固定：ベンチマークを常に先頭へ
    bench_t = u_cfg["bench"]
    tickers_raw = u_cfg["tickers"]
    all_t = [bench_t] + [t for t in tickers_raw if t != bench_t]

    if sync_btn or 'close_df' not in st.session_state or st.session_state.get('last_market') != market:
        with st.status("Engaging Data Stream...", expanded=False):
            raw_data = fetch_data_cached(tuple(all_t)) # 順序固定によりキャッシュキーが安定
            st.session_state.close_df = extract_close_robust(raw_data, all_t)
            st.session_state.last_market = market
            st.session_state.last_sync_ts = datetime.now().strftime("%H:%M:%S")

    if 'close_df' in st.session_state:
        audit = calculate_integrity(all_t, st.session_state.close_df, win_days)
        
        # --- AUDIT PANEL ---
        st.subheader("🛡️ Integrity Audit")
        exp = audit["expected"]
        def get_cls(v, e): return "status-green" if v/e >= 0.9 else "status-yellow" if v/e >= 0.7 else "status-red"
        
        a1, a2, a3, a4 = st.columns(4)
        with a1: kpi_card("Present", f"{audit['present']}/{exp}", get_cls(audit['present'], exp))
        with a2: kpi_card("Computable", f"{audit['computable']}/{exp}", get_cls(audit['computable'], exp))
        with a3: kpi_card("Synced", f"{audit['synced']}/{exp}", get_cls(audit['synced'], exp))
        with a4: kpi_card("Last Date", str(audit['most_common_date']).split()[0] if audit['most_common_date'] else "N/A", "status-green")

        # 3段階ヘルス証明
        health_ratio = audit['computable'] / exp
        if health_ratio >= 0.9: status_msg, status_cls = "[PASSED]", "cert-passed"
        elif health_ratio >= 0.7: status_msg, status_cls = "[DEGRADED]", "cert-degraded"
        else: status_msg, status_cls = "[FAILED]", "cert-failed"
        
        st.markdown(f'<div class="audit-cert {status_cls}">{status_msg} Bench:{bench_t} | Health:{health_ratio:.1%} | Synced:{audit["synced"]}</div>', unsafe_allow_html=True)

        # --- ANALYSIS ---
        if bench_t not in audit["computable_list"]:
            st.error(f"❌ Critical Failure: Benchmark ({bench_t}) is unreliable for this window.")
            st.stop()

        close_df = st.session_state.close_df
        b_series = close_df[bench_t].tail(win_days + 1)
        b_ret = (b_series.iloc[-1] / b_series.iloc[0] - 1) * 100
        
        results = []
        valid_tickers = [t for t in tickers_raw if t in audit['computable_list'] and t != bench_t]
        
        for t in valid_tickers:
            s = close_df[t].tail(win_days + 1)
            p_ret = (s.iloc[-1] / s.iloc[0] - 1) * 100
            
            # Acceleration
            half = max(1, win_days // 2)
            p_half = (s.iloc[-1] / s.iloc[-(half+1)] - 1) * 100
            
            # 堅牢なStable判定
            s6 = close_df[t].tail(6).dropna()
            b6 = close_df[bench_t].tail(6).dropna()
            if len(s6) < 6 or len(b6) < 6:
                stable = "?"
            else:
                rs_short = ((s6.iloc[-1]/s6.iloc[0]-1)*100) - ((b6.iloc[-1]/b6.iloc[0]-1)*100)
                stable = "✅" if np.sign(p_ret - b_ret) == np.sign(rs_short) else "⚠️"
            
            results.append({
                'Ticker': t, 
                'RS': p_ret - b_ret, 
                'Accel': p_half - (p_ret / 2),
                'MaxDD': abs(((s / s.cummax() - 1) * 100).min()),
                'Stable': stable
            })
        
        if results:
            res_df = pd.DataFrame(results)
            res_df['RS_z'], res_df['Accel_z'], res_df['DD_z'] = zscore_series(res_df['RS']), zscore_series(res_df['Accel']), zscore_series(res_df['MaxDD'])
            res_df['ApexScore'] = (0.6 * res_df['RS_z']) + (0.25 * res_df['Accel_z']) - (0.15 * res_df['DD_z'])
            res_df = res_df.sort_values('ApexScore', ascending=False)

            # --- STRATEGIC ACTIONS ---
            st.markdown("---")
            st.subheader("🎯 Tactical Priority")
            for i, (_, row) in enumerate(res_df.head(3).iterrows()):
                st.markdown(f"""<div class="next-action-card">
                    <small>Rank #{i+1} | Score: {row['ApexScore']:.2f}</small><br>
                    <b style="font-size:20px;">{row['Ticker']}</b> {row['Stable']}<br>
                    <span style="color:#8b949e;">{"高モメンタム + 低ドローダウン" if row['RS_z'] > 0 and row['DD_z'] < 0 else "トレンド優位"}</span></div>""", unsafe_allow_html=True)

            t1, t2 = st.tabs(["📊 Leaderboard", "📈 Mean Correlation"])
            with t1:
                st.dataframe(res_df[['Ticker', 'ApexScore', 'RS', 'Accel', 'MaxDD', 'Stable']].style.background_gradient(subset=['ApexScore', 'RS'], cmap='RdYlGn')
                             .format({'ApexScore': '{:.2f}', 'RS': '{:.2f}%', 'Accel': '{:.2f}%', 'MaxDD': '{:.2f}%'}), use_container_width=True)
            with t2:
                rets = close_df[valid_tickers].pct_change().tail(win_days)
                corr = rets.corr()
                np.fill_diagonal(corr.values, np.nan)
                st.plotly_chart(px.bar(corr.mean(skipna=True), title="Mean Correlation (Diagonal Excluded)"), use_container_width=True)

if __name__ == "__main__":
    main()