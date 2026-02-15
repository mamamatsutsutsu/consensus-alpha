import time
import streamlit as st
import alphalens

# NEXT GEN APP (Theme Portfolio Builder)
try:
    from next_gen_app_tab import render_next_gen_tab  # type: ignore
except Exception:
    render_next_gen_tab = None  # type: ignore


st.set_page_config(page_title="AlphaLens Pro", layout="wide", initial_sidebar_state="collapsed", page_icon="🦅")

def _gate():
    # Optional password gate: set APP_PASSWORD in Streamlit Secrets to enable.
    pw = None
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None
    if not pw:
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.markdown("""<div style="max-width:520px;margin:60px auto;padding:24px;border-radius:18px;
        border:1px solid rgba(255,255,255,0.12);background:rgba(0,0,0,0.55);backdrop-filter: blur(10px);">
        <div style="font-family:Orbitron,sans-serif;font-size:22px;letter-spacing:0.12em;color:#00f2fe;">ALPHALENS</div>
        <div style="opacity:0.85;margin-top:6px;">Enter password to continue.</div>
    </div>""", unsafe_allow_html=True)
    x = st.text_input("Password", type="password")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Unlock", type="primary", use_container_width=True):
            if x == pw:
                st.session_state.auth_ok = True
                st.toast("Unlocked", icon="✅")
                st.rerun()
            else:
                st.error("Wrong password.")
    with col2:
        st.button("Reset", use_container_width=True, on_click=lambda: st.session_state.pop("auth_ok", None))
    return False

def main():
    if not _gate():
        st.stop()

    t1, t2 = st.tabs(["ALPHALENS", "NEXT GEN APP"])
    with t1:
        try:
            alphalens.run()
        except Exception as e:
            st.error("App error. Details below (also recorded in logs).")
            st.exception(e)
            with st.expander("System logs (last 120)"):
                logs = st.session_state.get("system_logs", [])
                st.text("\n".join(logs[-120:]) if logs else "(empty)")
    with t2:
        st.markdown("<h1 style='font-family:Orbitron, sans-serif; color:#00f2fe;'>NEXT GEN APP</h1>", unsafe_allow_html=True)
        st.caption("AI (Gemini)でテーマ投資のユニバース構築・TRR推定・ランキングを行うモジュールです。")

        if render_next_gen_tab is None:
            st.warning("next_gen_app_tab.py が見つかりません。プロジェクト直下に追加してから再起動してください。")
        else:
            try:
                render_next_gen_tab(data_dir="data")
            except Exception as e:
                st.error("NEXT GEN APP error. Details below (also recorded in logs).")
                st.exception(e)

if __name__ == "__main__":
    main()