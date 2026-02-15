import streamlit as st
import alphalens

st.set_page_config(
    page_title="AlphaLens Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🦅"
)

t1, t2 = st.tabs(["ALPHALENS", "COMING SOON"])

with t1:
    try:
        alphalens.run()
    except Exception as e:
        st.error("This app has encountered an error. 詳細はログを確認してください。")
        st.exception(e)

with t2:
    st.markdown("<h1 style='font-family:Orbitron, sans-serif; color:#00f2fe;'>NEXT GEN APP</h1>", unsafe_allow_html=True)
    st.caption("This module is under development. Stay tuned.")
