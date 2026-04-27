import warnings
warnings.filterwarnings('ignore')

import streamlit as st

from components.styles import inject_css
from utils.pipeline import load_and_preprocess
import views.overview           as pg_overview
import views.data_explorer      as pg_data
import views.pipeline_explained as pg_pipeline
import views.model_results      as pg_results
import views.predict_patient    as pg_predict


# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlzDetect AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AlzDetect AI")
    st.markdown("*Alzheimer's Classification System*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Data Explorer",
        "⚙️ Pipeline Explained",
        "🎯 Model Results",
        "🔬 Predict Patient",
    ])
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Dataset:</strong> OASIS Longitudinal<br>
    <strong>Model:</strong> Decision Tree (Hunt's Algo)<br>
    <strong>Preprocessing:</strong> Mean Imputation → Robust Scaling → SMOTE
    </div>""", unsafe_allow_html=True)

# ─── LOAD DATA (cached) ────────────────────────────────────────────────────────
with st.spinner("Running full ML pipeline..."):
    data = load_and_preprocess()

# ─── ROUTE TO PAGE ─────────────────────────────────────────────────────────────
PAGES = {
    "🏠 Overview":           pg_overview.render,
    "📊 Data Explorer":      pg_data.render,
    "⚙️ Pipeline Explained": pg_pipeline.render,
    "🎯 Model Results":      pg_results.render,
    "🔬 Predict Patient":    pg_predict.render,
}

PAGES[page](data)
