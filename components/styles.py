import streamlit as st


def _theme_tokens(theme: str) -> dict:
    if theme == "Light":
        return {
            "sidebar_bg": "linear-gradient(180deg, #dbeafe 0%, #bfdbfe 100%)",
            "sidebar_text": "#0f172a",
            "sidebar_box_bg": "rgba(255,255,255,0.75)",
            "sidebar_box_border": "rgba(15,23,42,0.15)",
            "metric_bg": "linear-gradient(135deg, #ffffff 0%, #eff6ff 100%)",
            "metric_border": "#93c5fd66",
            "metric_shadow": "rgba(59,130,246,0.12)",
            "metric_val": "#1d4ed8",
            "metric_text": "#334155",
            "section_border": "#2563eb",
            "section_title": "#0f172a",
            "section_sub": "#475569",
            "info_bg": "#f8fafc",
            "info_border": "#cbd5e1",
            "info_text": "#334155",
            "info_strong": "#1d4ed8",
        }
    return {
        "sidebar_bg": "linear-gradient(180deg, #1a3a2a 0%, #2d5a3d 100%)",
        "sidebar_text": "#ffffff",
        "sidebar_box_bg": "rgba(255,255,255,0.1)",
        "sidebar_box_border": "rgba(255,255,255,0.2)",
        "metric_bg": "linear-gradient(135deg, #1e1e3f 0%, #2d1b69 100%)",
        "metric_border": "#7c3aed44",
        "metric_shadow": "rgba(124,58,237,0.15)",
        "metric_val": "#a78bfa",
        "metric_text": "#94a3b8",
        "section_border": "#7c3aed",
        "section_title": "#e2e8f0",
        "section_sub": "#94a3b8",
        "info_bg": "#1e293b",
        "info_border": "#334155",
        "info_text": "#cbd5e1",
        "info_strong": "#a78bfa",
    }
def get_plot_theme(theme: str) -> dict:
    if theme == "Light":
        return {
            "fig_bg": "#ffffff",
            "ax_bg": "#f8fafc",
            "title": "#0f172a",
            "text": "#334155",
            "spine": "#cbd5e1",
            "legend_bg": "#ffffff",
            "bar_colors": ["#ef4444", "#22c55e", "#f59e0b"],
            "accent": "#2563eb",
            "heatmap_cmap": "Blues",
        }
    return {
        "fig_bg": "#0f172a",
        "ax_bg": "#1e293b",
        "title": "#e2e8f0",
        "text": "#94a3b8",
        "spine": "#334155",
        "legend_bg": "#1e293b",
        "bar_colors": ["#ef4444", "#22c55e", "#f59e0b"],
        "accent": "#f472b6",
        "heatmap_cmap": "Purples",
    }
def inject_css(theme: str = "Dark"):
    t = _theme_tokens(theme)
    custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    transition: background-color 0.25s ease;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stSidebar"] {
    background: __SIDEBAR_BG__;
}
[data-testid="stSidebar"] * {
    color: __SIDEBAR_TEXT__ !important;
}
[data-testid="stSidebar"] .info-box {
    background: __SIDEBAR_BOX_BG__;
    border-color: __SIDEBAR_BOX_BORDER__;
}
.metric-card {
    background: __METRIC_BG__;
    border: 1px solid __METRIC_BORDER__;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px __METRIC_SHADOW__;
}
.metric-card h2 { color: __METRIC_VAL__; font-family: 'Space Mono'; font-size: 2.2rem; margin: 0; }
.metric-card p  { color: __METRIC_TEXT__; margin: 4px 0 0 0; font-size: 0.85rem; letter-spacing: 0.05em; }
.section-header {
    margin: 30px 0 18px 0;
    border-left: 4px solid __SECTION_BORDER__;
    padding-left: 14px;
}
.section-header h3 { color: __SECTION_TITLE__; margin: 0; font-size: 1.3rem; font-weight: 800; }
.section-header p  { color: __SECTION_SUB__; margin: 4px 0 0 0; font-size: 0.82rem; }

.pred-box {
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 18px 0;
}
.pred-nondemented { background: linear-gradient(135deg, #052e16,#166534); border: 2px solid #22c55e; color: #86efac; }
.pred-demented    { background: linear-gradient(135deg, #450a0a,#991b1b); border: 2px solid #ef4444; color: #fca5a5; }
.pred-converted   { background: linear-gradient(135deg, #451a03,#92400e); border: 2px solid #f59e0b; color: #fcd34d; }

.info-box {
    background: __INFO_BG__;
    border: 1px solid __INFO_BORDER__;
    border-radius: 10px;
    padding: 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: __INFO_TEXT__;
    line-height: 1.7;
}
.info-box strong { color: __INFO_STRONG__; }
</style>
"""
    custom_css = (
        custom_css
        .replace("__SIDEBAR_BG__", t["sidebar_bg"])
        .replace("__SIDEBAR_TEXT__", t["sidebar_text"])
        .replace("__SIDEBAR_BOX_BG__", t["sidebar_box_bg"])
        .replace("__SIDEBAR_BOX_BORDER__", t["sidebar_box_border"])
        .replace("__METRIC_BG__", t["metric_bg"])
        .replace("__METRIC_BORDER__", t["metric_border"])
        .replace("__METRIC_SHADOW__", t["metric_shadow"])
        .replace("__METRIC_VAL__", t["metric_val"])
        .replace("__METRIC_TEXT__", t["metric_text"])
        .replace("__SECTION_BORDER__", t["section_border"])
        .replace("__SECTION_TITLE__", t["section_title"])
        .replace("__SECTION_SUB__", t["section_sub"])
        .replace("__INFO_BG__", t["info_bg"])
        .replace("__INFO_BORDER__", t["info_border"])
        .replace("__INFO_TEXT__", t["info_text"])
        .replace("__INFO_STRONG__", t["info_strong"])
    )
    st.markdown(custom_css, unsafe_allow_html=True)