import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from components.ui import section, metric_card


def render(data: dict):
    st.title("🎯 Model Results & Evaluation")
    st.markdown("---")

    clf         = data['clf']
    acc         = data['acc']
    bal_acc     = data['balanced_acc']
    f1_macro    = data['f1_macro']
    precision_m = data['precision_macro']
    recall_m    = data['recall_macro']
    rmse        = data['rmse']
    log_acc     = data['log_acc']
    log_bal_acc = data['log_bal_acc']
    cv_scores   = data['cv_scores']
    cv_scores_log = data['cv_scores_log']
    cm          = data['cm']
    report      = data['report']
    features    = data['features']
    label_names = data['label_names']
    bias_by_sex = data['bias_by_sex']
    bias_by_age = data['bias_by_age']
    bias_summary = data['bias_summary']
    pt = data.get("plot_theme", {})

    # ── METRICS ─────────────────────────────────────────────────────────────
    section("Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card(f"{acc*100:.1f}%",               "Test Accuracy"),              unsafe_allow_html=True)
    c2.markdown(metric_card(f"{bal_acc*100:.1f}%",           "Test Balanced Accuracy"),     unsafe_allow_html=True)
    c3.markdown(metric_card(f"{f1_macro*100:.1f}%",          "F1 Macro"), unsafe_allow_html=True)
    c4.markdown(metric_card(f"{cv_scores.mean()*100:.1f}%",  "CV Balanced Accuracy (Mean)"), unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    c7.markdown(metric_card(f"{precision_m*100:.1f}%", "Precision Macro"), unsafe_allow_html=True)
    c8.markdown(metric_card(f"{recall_m*100:.1f}%", "Recall Macro"), unsafe_allow_html=True)
    c9.markdown(metric_card(f"{rmse:.3f}", "RMSE (Probabilities)", "Lower is better"), unsafe_allow_html=True)

    section("Softer Model Comparison (Logistic Regression)")
    c4, c5, c6 = st.columns(3)
    c4.markdown(metric_card(f"{log_acc*100:.1f}%",                 "LogReg Test Accuracy"),              unsafe_allow_html=True)
    c5.markdown(metric_card(f"{log_bal_acc*100:.1f}%",             "LogReg Test Balanced Accuracy"),     unsafe_allow_html=True)
    c6.markdown(metric_card(f"{cv_scores_log.mean()*100:.1f}%",    "LogReg CV Balanced Accuracy (Mean)"), unsafe_allow_html=True)

    # ── CONFUSION MATRIX ────────────────────────────────────────────────────
    section("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=pt.get("fig_bg", "#0f172a"))
    ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
    sns.heatmap(cm, annot=True, fmt='d', cmap=pt.get("heatmap_cmap", "Purples"), ax=ax,
                xticklabels=label_names, yticklabels=label_names,
                linewidths=1, linecolor=pt.get("fig_bg", "#0f172a"),
                annot_kws={'size': 14, 'weight': 'bold', 'color': pt.get("title", "#e2e8f0")})
    ax.set_xlabel('Predicted', color=pt.get("text", "#94a3b8"))
    ax.set_ylabel('Actual',    color=pt.get("text", "#94a3b8"))
    ax.tick_params(colors=pt.get("text", "#94a3b8"))
    fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── CLASSIFICATION REPORT ───────────────────────────────────────────────
    section("Classification Report")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap='Purples'),
                 use_container_width=True)

    # ── CROSS-VALIDATION ────────────────────────────────────────────────────
    section("Cross-Validation Scores (5-Fold, Balanced Accuracy)")
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=pt.get("fig_bg", "#0f172a"))
    ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
    ax.bar([f'Fold {i+1}' for i in range(5)], cv_scores,
           color=['#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe'] if data.get("ui_theme") == "Dark" else ['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
           edgecolor=pt.get("spine", "#334155"))
    ax.axhline(cv_scores.mean(), color=pt.get("accent", "#f472b6"), linestyle='--', linewidth=1.5,
               label=f'Mean BA: {cv_scores.mean():.3f}')
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors=pt.get("text", "#94a3b8"))
    ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"))
    for spine in ax.spines.values():
        spine.set_edgecolor(pt.get("spine", "#334155"))
    fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── BIAS CHECK ───────────────────────────────────────────────────────────
    section("Bias Check (Demographic Slices)")
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(metric_card(f"{bias_summary['sex_accuracy_gap']*100:.1f}%", "Sex Accuracy Gap"), unsafe_allow_html=True)
    b2.markdown(metric_card(f"{bias_summary['sex_f1_gap']*100:.1f}%", "Sex F1 Gap"), unsafe_allow_html=True)
    b3.markdown(metric_card(f"{bias_summary['age_accuracy_gap']*100:.1f}%", "Age Accuracy Gap"), unsafe_allow_html=True)
    b4.markdown(metric_card(f"{bias_summary['age_f1_gap']*100:.1f}%", "Age F1 Gap"), unsafe_allow_html=True)

    st.caption("Gap = max(group metric) - min(group metric). Smaller gaps indicate more consistent behavior across groups.")

    st.markdown("#### By Sex")
    st.dataframe(bias_by_sex.round(3), use_container_width=True)
    st.markdown("#### By Age Group")
    st.dataframe(bias_by_age.round(3), use_container_width=True)

    # ── FEATURE IMPORTANCE ──────────────────────────────────────────────────
    section("Feature Importance")
    fi_df = (pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
               .sort_values('Importance', ascending=True))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=pt.get("fig_bg", "#0f172a"))
    ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
    ax.barh(fi_df['Feature'], fi_df['Importance'],
            color=pt.get("accent", "#7c3aed"), edgecolor=pt.get("spine", "#334155"))
    ax.tick_params(colors=pt.get("text", "#94a3b8"))
    for spine in ax.spines.values():
        spine.set_edgecolor(pt.get("spine", "#334155"))
    ax.set_xlabel('Importance', color=pt.get("text", "#94a3b8"))
    ax.set_title('Feature Importance from Decision Tree', color=pt.get("title", "#e2e8f0"), fontsize=13)
    fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── TREE VISUALIZATION ──────────────────────────────────────────────────
    section("Decision Tree Visualization (Depth 3)")
    fig, ax = plt.subplots(figsize=(18, 8), facecolor=pt.get("fig_bg", "#0f172a"))
    ax.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plot_tree(clf, feature_names=features, class_names=label_names,
              filled=True, rounded=True, max_depth=3, ax=ax,
              fontsize=9, proportion=False)
    fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
