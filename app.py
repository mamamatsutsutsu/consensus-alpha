import streamlit as st
import alphalens

# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="ALPHALENS /THEMELENS",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🦅",
)


# ==========================================================
# Global styling
#   - Fixes white mobile background
#   - Dark glass + subtle grid
#   - Higher-contrast, more legible inputs
#   - Sidebar matches the theme
# ==========================================================

def _inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;700&display=swap');

        :root{
            --safe-top: env(safe-area-inset-top, 0px);
            --safe-bottom: env(safe-area-inset-bottom, 0px);

            --bg0: #05060a;
            --bg1: #070A12;

            --text: rgba(255,255,255,0.92);
            --muted: rgba(255,255,255,0.70);
            --hair: rgba(255,255,255,0.12);
            --hair2: rgba(255,255,255,0.18);

            --glass: rgba(255,255,255,0.06);
            --glass2: rgba(255,255,255,0.09);

            --accentA: rgba(0,242,254,1);
            --accentB: rgba(79,172,254,1);

            --grid: rgba(255,255,255,0.035);
            --grid2: rgba(255,255,255,0.020);
        }

        html, body{
            height: 100%;
            background: var(--bg0); /* prevents iOS overscroll from showing white */
            color-scheme: dark;
            overscroll-behavior-y: none;
        }

        *, *::before, *::after{ box-sizing: border-box; }
        *{ -webkit-tap-highlight-color: transparent; }

        /* Streamlit root */
        .stApp{
            background: transparent;
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            text-rendering: geometricPrecision;
        }

        /* Main app background (mobile white fix) */
        div[data-testid="stAppViewContainer"]{
            background:
                radial-gradient(1200px 800px at 12% 8%, rgba(0,242,254,0.18), transparent 60%),
                radial-gradient(900px 700px at 88% 18%, rgba(79,172,254,0.14), transparent 55%),
                radial-gradient(900px 600px at 50% 110%, rgba(0,0,0,0.70), transparent 60%),
                linear-gradient(var(--grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--grid2) 1px, transparent 1px),
                linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 45%, var(--bg0) 100%);
            background-size:
                auto,
                auto,
                auto,
                56px 56px,
                56px 56px,
                auto;
            background-position:
                center,
                center,
                center,
                0 0,
                0 0,
                center;
        }

        /* Make sure inner containers don't paint white */
        section.main, div[data-testid="stMain"], div[data-testid="stMainBlockContainer"]{
            background: transparent !important;
        }

        /* Hide Streamlit chrome */
        #MainMenu{visibility:hidden;}
        footer{visibility:hidden;}
        header[data-testid="stHeader"]{background: transparent;}

        /* Typography defaults */
        h1,h2,h3{ letter-spacing: -0.01em; }
        p,li{ color: var(--text); }

        /* Inputs: higher contrast + more legible */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input{
            background: var(--glass) !important;
            border: 1px solid var(--hair) !important;
            border-radius: 14px !important;
            padding: 0.85rem 1rem !important;
            color: var(--text) !important;
            caret-color: rgba(0,242,254,0.95) !important;
            font-size: 18px !important;
            font-weight: 650 !important;
            letter-spacing: 0.02em !important;
            line-height: 1.2 !important;
            -webkit-text-fill-color: rgba(255,255,255,0.92) !important;
            text-shadow: 0 0 20px rgba(0,0,0,0.55);
        }

        div[data-testid="stTextInput"] input::placeholder,
        div[data-testid="stTextArea"] textarea::placeholder{
            color: rgba(255,255,255,0.45) !important;
        }

        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus,
        div[data-testid="stNumberInput"] input:focus{
            border-color: rgba(0,242,254,0.55) !important;
            box-shadow: 0 0 0 4px rgba(0,242,254,0.12) !important;
        }

        /* iOS / Chrome autofill can force light backgrounds; neutralize it */
        input:-webkit-autofill,
        input:-webkit-autofill:hover,
        input:-webkit-autofill:focus,
        textarea:-webkit-autofill,
        textarea:-webkit-autofill:hover,
        textarea:-webkit-autofill:focus{
            -webkit-text-fill-color: rgba(255,255,255,0.92) !important;
            transition: background-color 600000s 0s, color 600000s 0s;
            box-shadow: 0 0 0px 1000px rgba(255,255,255,0.06) inset !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
        }

        /* Buttons */
        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button{
            border-radius: 14px !important;
            padding: 0.78rem 1rem !important;
            border: 1px solid var(--hair2) !important;
            background: var(--glass) !important;
            color: var(--text) !important;
            transition: transform 120ms ease, filter 120ms ease, background 120ms ease, border-color 120ms ease;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover{
            transform: translateY(-1px);
            background: var(--glass2) !important;
            border-color: rgba(255,255,255,0.26) !important;
        }
        div[data-testid="stButton"] button:active,
        div[data-testid="stFormSubmitButton"] button:active{
            transform: translateY(0px);
            filter: brightness(0.98);
        }
        div[data-testid="stButton"] button[kind="primary"],
        div[data-testid="stFormSubmitButton"] button[kind="primary"]{
            background: linear-gradient(135deg, rgba(0,242,254,0.92), rgba(79,172,254,0.78)) !important;
            border: 1px solid rgba(0,242,254,0.35) !important;
            color: #031015 !important;
            font-weight: 800 !important;
        }

        /* Sidebar: match the glassy dark theme (even if collapsed) */
        section[data-testid="stSidebar"]{
            background:
                radial-gradient(700px 420px at 20% 10%, rgba(0,242,254,0.10), transparent 60%),
                linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border-right: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(14px);
        }
        section[data-testid="stSidebar"] > div{
            background: transparent;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p{
            color: rgba(255,255,255,0.86) !important;
        }

        /* Alerts: keep them tight and on-brand */
        div[data-testid="stAlert"]{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
        }

        a{ color: rgba(0,242,254,0.85); }
        hr{ border-color: rgba(255,255,255,0.10); }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_css()


# ==========================================================
# Passcode gate
# ==========================================================

def _gate() -> bool:
    """Optional password gate: set APP_PASSWORD in Streamlit Secrets to enable."""

    pw = None
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None

    if not pw:
        return True
    if st.session_state.get("auth_ok"):
        return True

    # Gate-only layout: centered glass card (works well on mobile)
    st.markdown(
        """
        <style>
        /* Center everything while the gate is active */
        div[data-testid="stAppViewContainer"] .block-container{
            padding-top: 0rem !important;
            padding-bottom: calc(1.2rem + var(--safe-bottom)) !important;
            min-height: calc(100vh - var(--safe-top));
            display: flex;
            align-items: center;
            justify-content: center;
        }
        @media (max-width: 768px){
            div[data-testid="stAppViewContainer"] .block-container{
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
            }
        }

        /* Glass card */
        div[data-testid="stForm"]{
            width: min(560px, 100%);
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.14);
            background:
                radial-gradient(800px 400px at 10% 10%, rgba(0,242,254,0.12), transparent 55%),
                radial-gradient(700px 420px at 85% 15%, rgba(79,172,254,0.10), transparent 55%),
                linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
            box-shadow: 0 22px 70px rgba(0,0,0,0.55);
            backdrop-filter: blur(14px);
        }
        div[data-testid="stForm"] > form{ padding: 26px 22px 18px 22px; }

        .gate-brand{
            font-family: Orbitron, Inter, sans-serif;
            font-size: 22px;
            letter-spacing: 0.14em;
            line-height: 1.15;
            color: rgba(255,255,255,0.92);
        }
        .gate-brand .accent{
            background: linear-gradient(135deg, rgba(0,242,254,1), rgba(79,172,254,1));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .gate-sub{
            margin-top: 10px;
            color: rgba(255,255,255,0.72);
            font-size: 14px;
            line-height: 1.55;
        }
        .gate-meta{
            margin-top: 14px;
            display:flex;
            justify-content: space-between;
            gap: 10px;
            color: rgba(255,255,255,0.55);
            font-size: 12px;
        }
        .gate-meta span{
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("gate_form", clear_on_submit=False):
        st.markdown(
            '<div class="gate-brand">ALPHALENS <span class="accent">/THEMELENS</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="gate-sub">Enter the passcode to continue.</div>',
            unsafe_allow_html=True,
        )

        x = st.text_input(
            "Passcode",
            type="password",
            key="gate_pw",
            label_visibility="collapsed",
            placeholder="Passcode",
        )

        unlock = st.form_submit_button("Unlock", type="primary", use_container_width=True)

        st.markdown(
            '<div class="gate-meta"><span>Private preview</span><span>Secure access</span></div>',
            unsafe_allow_html=True,
        )

    if unlock:
        if x == pw:
            st.session_state.auth_ok = True
            st.toast("Unlocked", icon="✅")
            st.rerun()
        else:
            st.session_state["gate_pw"] = ""
            st.error("Incorrect passcode.")

    return False


# ==========================================================
# App entry
# ==========================================================

def main() -> None:
    if not _gate():
        st.stop()

    # NEXT GEN APP — import after set_page_config & gate
    from next_gen_app_tab import render_next_gen_tab

    # UI SAFETY: keep top tabs reachable on mobile browsers (safe-area / browser chrome overlap)
    st.markdown(
        """
        <style>
        :root{ --safe-top: env(safe-area-inset-top, 0px); }
        /* Enforce comfortable top padding so st.tabs doesn't get hidden under toolbars */
        div[data-testid="stAppViewContainer"] .block-container{
            padding-top: calc(5.0rem + var(--safe-top)) !important;
        }
        @media (max-width: 768px){
            div[data-testid="stAppViewContainer"] .block-container{
                padding-top: calc(5.8rem + var(--safe-top)) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # NOTE: Tab logic untouched (as requested)
    t1, t2 = st.tabs(["ALPHALENS", "THEMELENS"])

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
        try:
            render_next_gen_tab(data_dir="data")
        except Exception as e:
            st.error("NEXT GEN APP error. Details below.")
            st.exception(e)


if __name__ == "__main__":
    main()
