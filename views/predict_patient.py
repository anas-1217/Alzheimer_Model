import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from components.ui import section, prediction_box


def render(data: dict):
    st.title("🔬 Patient Prediction")
    st.markdown("Enter patient clinical values below to get a real-time Alzheimer's classification.")
    st.markdown("---")

    model_pipeline = data['model_pipeline']      # Calibrated decision tree
    tree_pipeline = data['tree_pipeline']        # Raw tree pipeline for leaf diagnostics
    logistic_pipeline = data['logistic_pipeline']
    label_names = data['label_names']

    # ── INPUT FORM ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        age  = st.slider("Age", 60, 98, 75)
        sex  = st.selectbox("Gender", ["Female (0)", "Male (1)"])
        educ = st.slider("Education (years)", 1, 23, 12)
        ses  = st.slider("Socioeconomic Status (1=high, 5=low)", 1, 5, 2)
        mmse = st.slider("MMSE Score (0-30)", 0, 30, 25)

    with col2:
        cdr  = st.selectbox("CDR (Clinical Dementia Rating)", [0.0, 0.5, 1.0, 2.0])
        etiv = st.slider("eTIV (Estimated Total Intracranial Volume)", 1100, 2300, 1600)
        nwbv = st.slider("nWBV (Normalized Whole Brain Volume)", 0.60, 0.90, 0.75, step=0.01)
        asf  = st.slider("ASF (Atlas Scaling Factor)", 0.80, 1.60, 1.10, step=0.01)

    sex_val = 1 if "Male" in sex else 0

    model_choice = st.radio(
        "Prediction model",
        ["Calibrated Decision Tree (Hunt's)", "Logistic Regression (Softer)"],
        horizontal=True,
    )

    # ── CLASSIFY ────────────────────────────────────────────────────────────
    if st.button("🔍 Classify Patient", use_container_width=True):
        patient = np.array([[age, sex_val, educ, ses, mmse, cdr, etiv, nwbv, asf]])
        if model_choice == "Calibrated Decision Tree (Hunt's)":
            pred = model_pipeline.predict(patient)[0]
            proba = model_pipeline.predict_proba(patient)[0]
        else:
            pred = logistic_pipeline.predict(patient)[0]
            proba = logistic_pipeline.predict_proba(patient)[0]

        pred_label = label_names[pred]

        emoji = {'Nondemented': '✅', 'Demented': '⚠️', 'Converted': '🔄'}.get(pred_label, '')
        prediction_box(pred_label, emoji)

        # ── PROBABILITY BAR CHART ───────────────────────────────────────────
        section("Class Probabilities")
        fig, ax = plt.subplots(figsize=(7, 3), facecolor='#0f172a')
        ax.set_facecolor('#1e293b')
        colors = ['#ef4444', '#22c55e', '#f59e0b']
        bars   = ax.bar(label_names, proba,
                        color=colors[:len(label_names)], edgecolor='#1e293b')
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{p*100:.1f}%',
                    ha='center', color='#e2e8f0', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.set_ylabel('Probability', color='#94a3b8')
        fig.patch.set_facecolor('#0f172a')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if model_choice == "Calibrated Decision Tree (Hunt's)":
            # Leaf-level support explains repeated probability patterns in trees.
            x_imp = tree_pipeline.named_steps["imputer"].transform(patient)
            x_scaled = tree_pipeline.named_steps["scaler"].transform(x_imp)
            leaf_id = tree_pipeline.named_steps["clf"].apply(x_scaled)[0]
            leaf_support = int(tree_pipeline.named_steps["clf"].tree_.n_node_samples[leaf_id])

            st.markdown(
                f"""
                <div class='info-box'>
                <strong>Leaf Support:</strong> {leaf_support} training samples reached this tree leaf.<br>
                Decision tree probabilities are leaf-based, so nearby inputs can share identical percentages until a split threshold is crossed.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── DISCLAIMER ──────────────────────────────────────────────────────
        st.markdown("""
        <div class='info-box'>
        <strong>⚠️ Disclaimer:</strong> This tool is for educational and research purposes only.
        It should <em>not</em> be used as a substitute for professional medical diagnosis.
        Always consult a qualified physician for clinical decisions.
        </div>""", unsafe_allow_html=True)
