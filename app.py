import streamlit as st
import alphalens


# ==========================================================
# Branding / look & feel
# ==========================================================
APP_TITLE = "ALPHALENS /THEMELENS"
APP_ICON = "🦅"

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=APP_ICON,
)


def _inject_global_css() -> None:
    """Global, always-on styling.
    - Fixes white mobile background (iOS overscroll)
    - Adds a premium dark / glass visual theme
    """

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
            --glass2: rgba(255,255,255,0.10);

            --accentA: rgba(0,242,254,1);
            --accentB: rgba(79,172,254,1);
            --accentGlow: rgba(0,242,254,0.16);

            --r-md: 14px;
            --r-lg: 16px;
            --r-xl: 22px;
        }

        html, body{
            height: 100%;
            background: var(--bg0); /* prevents iOS overscroll from flashing white */
        }
        *{ -webkit-tap-highlight-color: transparent; }

        /* Streamlit root */
        .stApp{
            background: transparent;
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Main app background (mobile white fix + subtle grid) */
        div[data-testid="stAppViewContainer"]{
            background:
                radial-gradient(1200px 800px at 12% 8%, rgba(0,242,254,0.16), transparent 60%),
                radial-gradient(900px 700px at 88% 18%, rgba(79,172,254,0.12), transparent 55%),
                radial-gradient(900px 600px at 50% 112%, rgba(0,0,0,0.76), transparent 60%),
                repeating-linear-gradient(0deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 52px),
                repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 52px),
                linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 45%, var(--bg0) 100%);
        }

        /* Hide Streamlit chrome */
        #MainMenu{visibility:hidden;}
        footer{visibility:hidden;}
        header[data-testid="stHeader"]{background: transparent;}
        div[data-testid="stToolbar"]{visibility:hidden;}

        /* Typography */
        h1,h2,h3{
            letter-spacing: -0.01em;
        }
        p,li{ color: var(--text); }

        /* Sidebar: navigation rail vibe */
        section[data-testid="stSidebar"]{
            background:
                radial-gradient(700px 500px at 18% 0%, rgba(0,242,254,0.10), transparent 60%),
                radial-gradient(700px 520px at 85% 18%, rgba(79,172,254,0.08), transparent 58%),
                linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border-right: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(12px);
        }
        section[data-testid="stSidebar"] .block-container{
            padding-top: calc(1.05rem + var(--safe-top)) !important;
            padding-bottom: calc(1.0rem + var(--safe-bottom)) !important;
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
        }

        .sb-brand{
            margin-top: 6px;
            margin-bottom: 14px;
        }
        .sb-title{
            font-family: Orbitron, Inter, sans-serif;
            font-size: 14px;
            letter-spacing: 0.14em;
            line-height: 1.25;
            color: rgba(255,255,255,0.92);
            text-transform: uppercase;
        }
        .sb-title .accent{
            background: linear-gradient(135deg, rgba(0,242,254,1), rgba(79,172,254,1));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .sb-sub{
            margin-top: 7px;
            color: rgba(255,255,255,0.58);
            font-size: 11px;
            letter-spacing: 0.22em;
            text-transform: uppercase;
        }
        .sb-divider{
            height: 1px;
            background: rgba(255,255,255,0.10);
            margin: 14px 0;
        }
        .sb-foot{
            margin-top: 16px;
            color: rgba(255,255,255,0.46);
            font-size: 11px;
        }

        /* Inputs */
        div[data-testid="stTextInput"] input{
            background: rgba(255,255,255,0.055) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: var(--r-md) !important;
            padding: 0.86rem 1rem !important;
            color: rgba(255,255,255,0.94) !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            caret-color: rgba(0,242,254,0.90) !important;
        }
        div[data-testid="stTextInput"] input::placeholder{ color: rgba(255,255,255,0.42) !important; }
        div[data-testid="stTextInput"] input:focus{
            border-color: rgba(0,242,254,0.55) !important;
            box-shadow: 0 0 0 4px rgba(0,242,254,0.12) !important;
        }

        /* Buttons (secondary + primary) */
        div[data-testid="stButton"] button{
            border-radius: var(--r-md) !important;
            padding: 0.78rem 1rem !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            background: rgba(255,255,255,0.06) !important;
            color: rgba(255,255,255,0.92) !important;
            transition: transform 120ms ease, filter 120ms ease, background 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
        }
        div[data-testid="stButton"] button:hover{
            transform: translateY(-1px);
            background: rgba(255,255,255,0.09) !important;
            border-color: rgba(255,255,255,0.26) !important;
        }
        div[data-testid="stButton"] button:active{
            transform: translateY(0px);
            filter: brightness(0.98);
        }
        div[data-testid="stButton"] button[kind="primary"]{
            background: linear-gradient(135deg, rgba(0,242,254,0.92), rgba(79,172,254,0.78)) !important;
            border: 1px solid rgba(0,242,254,0.35) !important;
            color: #031015 !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 30px rgba(0,242,254,0.12);
        }

        /* Sidebar button alignment */
        section[data-testid="stSidebar"] div[data-testid="stButton"] button{
            justify-content: flex-start !important;
            gap: 10px !important;
            font-weight: 800 !important;
            letter-spacing: 0.02em !important;
        }

        /* Alerts: keep them sleek */
        div[data-testid="stAlert"]{
            border-radius: var(--r-lg);
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
        }

        a{ color: rgba(0,242,254,0.85); }
        hr{ border-color: rgba(255,255,255,0.10); }

        /* Scrollbar polish (desktop browsers) */
        ::-webkit-scrollbar{ width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb{
            background: rgba(255,255,255,0.12);
            border-radius: 999px;
            border: 2px solid rgba(0,0,0,0.0);
            background-clip: padding-box;
        }
        ::-webkit-scrollbar-thumb:hover{ background: rgba(255,255,255,0.18); }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_css()


def _get_password():
    try:
        return st.secrets.get("APP_PASSWORD")
    except Exception:
        return None


def _gate() -> bool:
    """Optional passcode gate: set APP_PASSWORD in Streamlit Secrets to enable."""

    pw = _get_password()
    if not pw:
        return True
    if st.session_state.get("auth_ok"):
        return True

    # Gate-only layout (center card; looks great on mobile; hides nav so first impression is clean)
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"]{ display:none !important; }

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
            border-radius: 26px;
            border: 1px solid rgba(255,255,255,0.14);
            background:
                radial-gradient(800px 420px at 10% 8%, rgba(0,242,254,0.12), transparent 55%),
                radial-gradient(700px 420px at 88% 16%, rgba(79,172,254,0.10), transparent 58%),
                linear-gradient(180deg, rgba(255,255,255,0.085), rgba(255,255,255,0.03));
            box-shadow: 0 26px 90px rgba(0,0,0,0.62);
            backdrop-filter: blur(16px);
        }
        div[data-testid="stForm"] > form{ padding: 28px 22px 18px 22px; }

        .gate-brand{
            font-family: Orbitron, Inter, sans-serif;
            font-size: 22px;
            letter-spacing: 0.14em;
            line-height: 1.15;
            color: rgba(255,255,255,0.92);
            text-transform: uppercase;
        }
        .gate-brand .accent{
            background: linear-gradient(135deg, rgba(0,242,254,1), rgba(79,172,254,1));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .gate-sub{
            margin-top: 10px;
            color: rgba(255,255,255,0.70);
            font-size: 13px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .gate-label{
            margin-top: 18px;
            margin-bottom: 8px;
            color: rgba(255,255,255,0.62);
            font-size: 11px;
            letter-spacing: 0.22em;
            text-transform: uppercase;
        }
        .gate-meta{
            margin-top: 14px;
            display:flex;
            justify-content: space-between;
            gap: 10px;
            color: rgba(255,255,255,0.52);
            font-size: 12px;
        }

        /* Passcode input: boost readability without affecting the rest of the apps too much */
        div[data-testid="stForm"] div[data-testid="stTextInput"] input{
            font-size: 18px !important;
            font-weight: 800 !important;
            letter-spacing: 0.14em !important;
            text-shadow: 0 0 16px rgba(0,242,254,0.12);
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
        st.markdown('<div class="gate-sub">Secure access</div>', unsafe_allow_html=True)

        st.markdown('<div class="gate-label">Access code</div>', unsafe_allow_html=True)
        code = st.text_input(
            "Access code",
            type="password",
            key="gate_pw",
            label_visibility="collapsed",
            placeholder="Enter access code…",
        )

        unlock = st.form_submit_button("Unlock", type="primary", use_container_width=True)

        st.markdown(
            '<div class="gate-meta"><span>Private preview</span><span>Encrypted session</span></div>',
            unsafe_allow_html=True,
        )

    if unlock:
        if code == pw:
            st.session_state.auth_ok = True
            st.toast("Access granted", icon="✅")
            st.rerun()
        else:
            st.error("Invalid access code.")

    return False


def _read_query_param(name: str):
    """Compat wrapper for query params across Streamlit versions."""
    try:
        qp = st.query_params  # new API
        v = qp.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        try:
            qp = st.experimental_get_query_params()  # older API
            v = qp.get(name)
            if isinstance(v, list):
                return v[0] if v else None
            return v
        except Exception:
            return None


def _set_query_param(name: str, value: str) -> None:
    """Best-effort: persist state in URL (nice for sharing links)."""
    try:
        st.query_params[name] = value
    except Exception:
        try:
            st.experimental_set_query_params(**{name: value})
        except Exception:
            pass


def _app_switcher() -> str:
    """App switcher (stable, doesn't execute both apps like st.tabs).

    Why: st.tabs can still execute code in ALL tabs each rerun. If either app crashes or shares widget keys,
    the whole page can fail and you get 'tabs not showing' or flaky rendering. This switcher runs ONLY the
    selected app.
    """

    # Initial selection (supports ?app=alphalens|themelens)
    if "active_app" not in st.session_state:
        raw = (_read_query_param("app") or "").strip().lower()
        if raw in ("themelens", "theme", "tl"):
            st.session_state["active_app"] = "THEMELENS"
        else:
            st.session_state["active_app"] = "ALPHALENS"

    active = st.session_state["active_app"]

    with st.sidebar:
        st.markdown(
            '<div class="sb-brand">'
            '<div class="sb-title">ALPHALENS <span class="accent">/THEMELENS</span></div>'
            '<div class="sb-sub">App switcher</div>'
            '</div>'
            '<div class="sb-divider"></div>',
            unsafe_allow_html=True,
        )

        # Buttons keep styling simple and reliable
        if active == "ALPHALENS":
            st.button("🦅  ALPHALENS", type="primary", use_container_width=True, key="nav_al")
            if st.button("🎛️  THEMELENS", use_container_width=True, key="nav_th"):
                st.session_state["active_app"] = "THEMELENS"
                _set_query_param("app", "themelens")
                st.rerun()
        else:
            if st.button("🦅  ALPHALENS", use_container_width=True, key="nav_al"):
                st.session_state["active_app"] = "ALPHALENS"
                _set_query_param("app", "alphalens")
                st.rerun()
            st.button("🎛️  THEMELENS", type="primary", use_container_width=True, key="nav_th")

        st.markdown('<div class="sb-foot">v0 · private preview</div>', unsafe_allow_html=True)

    return st.session_state["active_app"]


def main():
    if not _gate():
        st.stop()

    # Import after set_page_config & gate
    from next_gen_app_tab import render_next_gen_tab

    active_app = _app_switcher()

    # Render only the selected app (prevents cross-tab widget key collisions)
    if active_app == "ALPHALENS":
        try:
            alphalens.run()
        except Exception as e:
            st.error("ALPHALENS crashed. Details below (also recorded in logs).")
            st.exception(e)
            with st.expander("System logs (last 120)"):
                logs = st.session_state.get("system_logs", [])
                st.text("\n".join(logs[-120:]) if logs else "(empty)")
    else:
        try:
            render_next_gen_tab(data_dir="data")
        except Exception as e:
            st.error("THEMELENS crashed. Details below.")
            st.exception(e)


if __name__ == "__main__":
    main()
