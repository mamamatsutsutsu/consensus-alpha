import streamlit as st
import streamlit.components.v1 as components

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

        /* Tabs: glass rail */
        div[data-testid="stTabs"]{
            gap: 8px;
            padding: 6px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(10px);
        }
        div[data-testid="stTabs"] button[role="tab"]{
            border-radius: 12px;
            padding: 10px 14px;
            color: rgba(255,255,255,0.72);
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
            color: rgba(255,255,255,0.95);
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(0,242,254,0.22);
            box-shadow: 0 0 0 4px rgba(0,242,254,0.08);
        }

        /* Disabled tab (JS will add .tab-disabled) */
        div[data-testid="stTabs"] button[role="tab"].tab-disabled{
            opacity: 0.42 !important;
            filter: saturate(0.7) !important;
            cursor: not-allowed !important;
            pointer-events: none !important;
            border: 1px dashed rgba(255,255,255,0.16) !important;
            background: rgba(255,255,255,0.03) !important;
            box-shadow: none !important;
        }

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

        /* Sidebar */
        section[data-testid="stSidebar"]{
            border-right: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(12px);
        }

        a{ color: rgba(0,242,254,0.85); }
        hr{ border-color: rgba(255,255,255,0.10); }

        /* WIP panel */
        .wip-card{
            margin-top: 22px;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.14);
            background:
                radial-gradient(700px 360px at 12% 18%, rgba(0,242,254,0.10), transparent 58%),
                radial-gradient(680px 360px at 88% 22%, rgba(79,172,254,0.08), transparent 58%),
                linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            box-shadow: 0 18px 62px rgba(0,0,0,0.48);
            padding: 22px 20px;
            backdrop-filter: blur(12px);
        }
        .wip-kicker{
            font-size: 11px;
            letter-spacing: 0.22em;
            color: rgba(255,255,255,0.62);
            text-transform: uppercase;
        }
        .wip-title{
            margin-top: 10px;
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: rgba(255,255,255,0.94);
        }
        .wip-title .accent{
            background: linear-gradient(135deg, rgba(0,242,254,1), rgba(79,172,254,1));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .wip-sub{
            margin-top: 8px;
            font-size: 14px;
            line-height: 1.7;
            color: rgba(255,255,255,0.74);
        }
        .wip-meta{
            margin-top: 14px;
            display:flex;
            flex-wrap: wrap;
            gap: 10px;
            color: rgba(255,255,255,0.55);
            font-size: 12px;
        }
        .wip-meta span{
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_css()

# ==========================================================
# Passcode gate
# ==========================================================


def _inject_gate_autofocus_js() -> None:
    """Best-effort autofocus for the passcode field.

    Note: some mobile browsers (notably iOS Safari) restrict auto-focus behavior.
    """
    components.html(
        """
        <script>
        (function(){
          const focusGate = () => {
            const form = parent.document.querySelector('div[data-testid="stForm"]');
            if (!form) return false;

            // Prefer the password input inside the gate form
            const input = form.querySelector('input[type="password"]');
            if (!input) return false;

            // Avoid fighting the user if they already focused something
            const active = parent.document.activeElement;
            if (active && active.tagName && (active.tagName.toLowerCase() === 'input' || active.tagName.toLowerCase() === 'textarea')) {
              return true;
            }

            try { input.focus({ preventScroll: true }); } catch (e) { try { input.focus(); } catch (e2) {} }
            return true;
          };

          if (focusGate()) return;

          const obs = new MutationObserver(() => { if (focusGate()) obs.disconnect(); });
          obs.observe(parent.document.body, { childList: true, subtree: true });

          // Safety: stop observing after a short window
          setTimeout(() => { try { obs.disconnect(); } catch (e) {} }, 4000);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


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

    # Autofocus after render (best-effort)
    _inject_gate_autofocus_js()

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
# Tabs helpers
# ==========================================================


def _inject_disable_tab_js(match_prefix: str) -> None:
    """Disable (non-clickable) a tab whose label starts with match_prefix.

    This is purely a UI lock. The tab content should still be safe / placeholder on the Python side.
    """
    components.html(
        f"""
        <script>
        (function() {{
          const MATCH = {match_prefix!r};

          const disable = () => {{
            const buttons = Array.from(parent.document.querySelectorAll('div[data-testid="stTabs"] button[role="tab"]'));
            if (!buttons || buttons.length === 0) return false;

            const target = buttons.find(b => (b.innerText || '').trim().startsWith(MATCH));
            if (!target) return false;

            // If the disabled tab is currently selected, move away to the first tab.
            try {{
              if (target.getAttribute('aria-selected') === 'true') {{
                const first = buttons[0];
                if (first && first !== target) first.click();
              }}
            }} catch (e) {{}}

            // Mark disabled
            try {{
              target.classList.add('tab-disabled');
              target.setAttribute('aria-disabled', 'true');
              target.setAttribute('disabled', 'true');
              target.tabIndex = -1;
              target.title = 'Under development';
              target.style.pointerEvents = 'none';
            }} catch (e) {{}}

            return true;
          }};

          if (disable()) return;

          const obs = new MutationObserver(() => {{ if (disable()) obs.disconnect(); }});
          obs.observe(parent.document.body, {{ childList: true, subtree: true }});
          setTimeout(() => {{ try {{ obs.disconnect(); }} catch (e) {{}} }}, 5000);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _wip_panel(product_name: str) -> None:
    st.markdown(
        f"""
        <div class="wip-card">
          <div class="wip-kicker">Under development</div>
          <div class="wip-title">{product_name} <span class="accent">WIP</span></div>
          <div class="wip-sub">
            This section is currently locked while we build and polish the experience.
          </div>
          <div class="wip-meta">
            <span>Coming soon</span>
            <span>Design in progress</span>
            <span>Private build</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# App entry
# ==========================================================


def main() -> None:
    if not _gate():
        st.stop()

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

    # Tabs (kept as-is style/placement; only labels + lock applied)
    t1, t2, t3 = st.tabs(["ALPHALENS", "THEMELENS · WIP", "LAB · WIP"])

    # Lock THEMELENS tab UI (non-clickable)
    _inject_disable_tab_js("THEMELENS")

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
        # Locked / placeholder (do not import or run the real app here while WIP)
        _wip_panel("THEMELENS")

    with t3:
        _wip_panel("LAB")


if __name__ == "__main__":
    main()
