import streamlit as st
import alphalens

# -----------------------------------------------------
# MAIN LAUNCHER CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="AlphaLens Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🦅"
)

# -----------------------------------------------------
# TABS FOR MULTI-APP
# -----------------------------------------------------
t1, t2 = st.tabs(["ALPHALENS", "COMING SOON"])

with t1:
    alphalens.run()

with t2:
    st.markdown("<h1 style='font-family:Orbitron, sans-serif; color:#00f2fe;'>NEXT GEN APP</h1>", unsafe_allow_html=True)
    st.caption("This module is under development. Stay tuned.")