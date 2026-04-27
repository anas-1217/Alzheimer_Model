import streamlit as st


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a2a 0%, #2d5a3d 100%);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .info-box {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.2);
}

.metric-card {
    background: linear-gradient(135deg, #1e1e3f 0%, #2d1b69 100%);
    border: 1px solid #7c3aed44;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.15);
}
.metric-card h2 { color: #a78bfa; font-family: 'Space Mono'; font-size: 2.2rem; margin: 0; }
.metric-card p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.85rem; letter-spacing: 0.05em; }

.section-header {
    border-left: 4px solid #7c3aed;
    padding-left: 14px;
    margin: 30px 0 18px 0;
}
.section-header h3 { color: #e2e8f0; margin: 0; font-size: 1.3rem; font-weight: 800; }
.section-header p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.82rem; }

.pred-box {
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    font-family: 'Space Mono';
    font-size: 1.6rem;
    font-weight: 700;
    margin: 20px 0;
}
.pred-nondemented { background: linear-gradient(135deg, #052e16,#166534); border: 2px solid #22c55e; color: #4ade80; }
.pred-demented    { background: linear-gradient(135deg, #450a0a,#991b1b); border: 2px solid #ef4444; color: #f87171; }
.pred-converted   { background: linear-gradient(135deg, #451a03,#92400e); border: 2px solid #f59e0b; color: #fbbf24; }

.info-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.info-box strong { color: #a78bfa; }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
