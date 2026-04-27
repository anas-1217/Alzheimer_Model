import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from components.ui import section


def render(data: dict):
    st.title("📊 Data Explorer")

    df       = data['df']
    features = data['features']

    # ── RAW DATA ────────────────────────────────────────────────────────────
    section("Raw Dataset (First 50 rows)")
    st.dataframe(df.head(50), use_container_width=True)

    # ── STATISTICS ──────────────────────────────────────────────────────────
    section("Descriptive Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)

    # ── FEATURE HISTOGRAM ───────────────────────────────────────────────────
    section("Feature Distributions")
    feat_sel = st.selectbox("Select feature to plot", features)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    ax.hist(df[feat_sel].dropna(), bins=30,
            color='#7c3aed', edgecolor='#4c1d95', alpha=0.9)
    ax.set_title(f'Distribution of {feat_sel}', color='#e2e8f0', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    fig.patch.set_facecolor('#0f172a')
    st.pyplot(fig)
    plt.close()

    # ── CORRELATION HEATMAP ─────────────────────────────────────────────────
    section("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.4, linecolor='#1e293b',
                annot_kws={'size': 9, 'color': 'white'})
    ax.tick_params(colors='#94a3b8')
    fig.patch.set_facecolor('#0f172a')
    plt.title("Feature Correlation Matrix", color='#e2e8f0', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
