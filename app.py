import streamlit as st
import alphalens

st.set_page_config(
    page_title="AlphaLens Pro",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🦅"
)

def password_gate() -> bool:
    """Optional password gate. Set APP_PASSWORD in Streamlit secrets to enable."""
    pw = None
    try:
        pw = st.secrets.get("APP_PASSWORD", None)
    except Exception:
        pw = None

    if not pw:
        return True

    if st.session_state.get("_authed", False):
        return True

    with st.container():
        st.markdown("### 🔒 Access")
        entered = st.text_input("Password", type="password")
        if st.button("Enter", use_container_width=True):
            if entered == pw:
                st.session_state["_authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False

def main():
    if not password_gate():
        st.stop()

    t1, t2 = st.tabs(["ALPHALENS", "COMING SOON"])

    with t1:
        try:
            alphalens.run()
        except Exception as e:
            st.error("This app has encountered an error.")
            st.exception(e)

    with t2:
        st.markdown(
            "<h1 style='font-family:Orbitron, sans-serif; color:#00f2fe;'>NEXT GEN APP</h1>",
            unsafe_allow_html=True
        )
        st.caption("This module is under development. Stay tuned.")

if __name__ == "__main__":
    main()
