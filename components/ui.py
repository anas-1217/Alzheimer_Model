import streamlit as st


def section(title: str, subtitle: str = ""):
    """Render a styled section header with optional subtitle."""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
        <h3>{title}</h3>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def metric_card(value: str, title: str, subtitle: str = ""):
    """Render a single metric card and return its HTML."""
    sub_html = f'<p style="font-size:0.75rem;color:#64748b">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="metric-card">
        <h2>{value}</h2>
        <p>{title}</p>
        {sub_html}
    </div>"""


def info_box(html_content: str, extra_style: str = ""):
    """Render a styled info / explanation box."""
    st.markdown(
        f'<div class="info-box" style="{extra_style}">{html_content}</div>',
        unsafe_allow_html=True
    )


def prediction_box(label: str, emoji: str):
    """Render the big coloured prediction result box."""
    css_class = {
        'Nondemented': 'pred-nondemented',
        'Demented':    'pred-demented',
        'Converted':   'pred-converted',
    }.get(label, 'pred-nondemented')

    st.markdown(
        f'<div class="pred-box {css_class}">{emoji} Prediction: {label}</div>',
        unsafe_allow_html=True
    )
