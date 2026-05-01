import streamlit as st
import matplotlib.pyplot as plt
from components.ui import section, metric_card


def render(data: dict):
    st.title("🧠 AlzDetect AI")
    st.markdown("""
    <div class="info-box">
    <strong>AlzDetect AI</strong><br>
    Alzheimer's Classification System<br>
    <strong>Dataset:</strong> OASIS Longitudinal<br>
    <strong>Model:</strong> Decision Tree (Hunt's Algo)<br>
    <strong>Preprocessing:</strong> Mean Imputation → Robust Scaling → SMOTE
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── METRICS ROW ─────────────────────────────────────────────────────────
    acc       = data['acc']
    bal_acc   = data['balanced_acc']
    cv_scores = data['cv_scores']
    pt = data.get("plot_theme", {})

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{acc*100:.1f}%",              "Test Accuracy",   "Decision Tree on test set"),
        (f"{bal_acc*100:.1f}%",          "Balanced Accuracy", "More reliable for imbalanced classes"),
        ("3",                            "Classes",         "Nondemented / Demented / Converted"),
        (f"{cv_scores.mean()*100:.1f}%", "CV Balanced Acc", "5-fold cross-validation mean"),
    ]
    for col, (val, title, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(metric_card(val, title, sub), unsafe_allow_html=True)

    # ── PIPELINE STEPS ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Project Pipeline", "The 3-step preprocessing + classification flow")

    st.markdown("""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px">
        <div class="info-box" style="flex:1;min-width:160px;border-color:#3b82f644">
            <strong style="color:#60a5fa">STEP 1 — Mean Imputation</strong><br><br>
            Fill all missing (NaN) values with the column mean. Keeps all rows without distorting the data distribution.
        </div>
        <div class="info-box" style="flex:1;min-width:160px;border-color:#8b5cf644">
            <strong style="color:#a78bfa">STEP 2 — Robust Scaling</strong><br><br>
            Scales features using median & IQR instead of mean/std. Resistant to outliers — perfect for medical data.
        </div>
        <div class="info-box" style="flex:1;min-width:160px;border-color:#ec489944">
            <strong style="color:#f472b6">STEP 3 — SMOTE</strong><br><br>
            Synthetic Minority Oversampling. Generates synthetic samples for minority classes so no class dominates training.
        </div>
        <div class="info-box" style="flex:1;min-width:160px;border-color:#22c55e44">
            <strong style="color:#4ade80">STEP 4 — Hunt's Decision Tree</strong><br><br>
            Recursive top-down splitting. Selects best feature at each node to partition patients into Alzheimer's categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CLASS DISTRIBUTION ──────────────────────────────────────────────────
    section("Class Distribution: Before vs After SMOTE")

    b           = data['class_dist_before']
    a           = data['class_dist_after']
    label_names = list(data['label_names'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=pt.get("fig_bg", "#0f172a"))
    for ax, dist, title in [(axes[0], b, "Before SMOTE"), (axes[1], a, "After SMOTE")]:
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        vals = [dist.get(i, 0) for i in range(len(label_names))]
        bars = ax.bar(label_names, vals,
                      color=pt.get("bar_colors", ['#ef4444', '#22c55e', '#f59e0b']),
                      edgecolor=pt.get("spine", "#334155"), linewidth=1.2)
        ax.set_title(title, color=pt.get("title", "#e2e8f0"), fontsize=13, fontweight='bold')
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        for spine in ax.spines.values():
            spine.set_edgecolor(pt.get("spine", "#334155"))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(v),
                    ha='center', color=pt.get("title", "#e2e8f0"), fontsize=10)

    fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
