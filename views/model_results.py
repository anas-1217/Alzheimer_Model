import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, mean_squared_error,
)
from components.ui import section, metric_card


# ── Helper: compute group-level bias metrics for any model's predictions ──────
def _compute_bias(X_test, y_test, y_pred):
    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["y_true"] = y_test
    eval_df["y_pred"] = y_pred
    eval_df["sex_group"] = eval_df["M/F"].map({0: "Female", 1: "Male"}).fillna("Unknown")

    age_bins   = [-np.inf, 69, 79, np.inf]
    age_labels = ["<70", "70-79", "80+"]
    eval_df["age_group"] = pd.cut(eval_df["Age"], bins=age_bins, labels=age_labels)
    eval_df["age_group"] = eval_df["age_group"].astype(str).replace("nan", "Unknown")

    def _group_metrics(frame, group_col):
        rows = []
        for group_name, gf in frame.groupby(group_col):
            if len(gf) == 0:
                continue
            rows.append({
                "group":        str(group_name),
                "n_samples":    int(len(gf)),
                "accuracy":     accuracy_score(gf["y_true"], gf["y_pred"]),
                "f1_macro":     f1_score(gf["y_true"], gf["y_pred"], average="macro", zero_division=0),
                "recall_macro": recall_score(gf["y_true"], gf["y_pred"], average="macro", zero_division=0),
            })
        return pd.DataFrame(rows)

    by_sex = _group_metrics(eval_df, "sex_group")
    by_age = _group_metrics(eval_df, "age_group")

    def _gap(df, col):
        if df.empty or df[col].isna().all():
            return 0.0
        return float(df[col].max() - df[col].min())

    summary = {
        "sex_accuracy_gap": _gap(by_sex, "accuracy"),
        "sex_f1_gap":       _gap(by_sex, "f1_macro"),
        "age_accuracy_gap": _gap(by_age, "accuracy"),
        "age_f1_gap":       _gap(by_age, "f1_macro"),
    }
    return by_sex, by_age, summary


# ── Helper: render bias cards + tables + charts ───────────────────────────────
def _render_bias_section(by_sex, by_age, summary, pt):
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(metric_card(f"{summary['sex_accuracy_gap']*100:.1f}%", "Sex Accuracy Gap"),  unsafe_allow_html=True)
    b2.markdown(metric_card(f"{summary['sex_f1_gap']*100:.1f}%",       "Sex F1 Gap"),        unsafe_allow_html=True)
    b3.markdown(metric_card(f"{summary['age_accuracy_gap']*100:.1f}%", "Age Accuracy Gap"),  unsafe_allow_html=True)
    b4.markdown(metric_card(f"{summary['age_f1_gap']*100:.1f}%",       "Age F1 Gap"),        unsafe_allow_html=True)
    st.caption("Gap = max(group metric) - min(group metric). Smaller gaps indicate fairer behaviour across groups.")

    st.markdown("#### By Sex")
    st.dataframe(by_sex.round(3), use_container_width=True)
    st.markdown("#### By Age Group")
    st.dataframe(by_age.round(3), use_container_width=True)

    if not by_sex.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=pt.get("fig_bg", "#0f172a"))
        for ax, metric, title in [
            (axes[0], "accuracy", "Accuracy by Sex"),
            (axes[1], "f1_macro", "F1 Macro by Sex"),
        ]:
            ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
            bars = ax.bar(by_sex["group"], by_sex[metric],
                          color=["#f472b6", "#60a5fa"], edgecolor=pt.get("spine", "#334155"))
            for bar, v in zip(bars, by_sex[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v*100:.1f}%", ha="center", color=pt.get("title", "#e2e8f0"), fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.set_title(title, color=pt.get("title", "#e2e8f0"), fontsize=11)
            ax.tick_params(colors=pt.get("text", "#94a3b8"))
            for sp in ax.spines.values():
                sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if not by_age.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=pt.get("fig_bg", "#0f172a"))
        age_colors = ["#34d399", "#f59e0b", "#f87171"]
        for ax, metric, title in [
            (axes[0], "accuracy", "Accuracy by Age Group"),
            (axes[1], "f1_macro", "F1 Macro by Age Group"),
        ]:
            ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
            bars = ax.bar(by_age["group"], by_age[metric],
                          color=age_colors[:len(by_age)], edgecolor=pt.get("spine", "#334155"))
            for bar, v in zip(bars, by_age[metric]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v*100:.1f}%", ha="center", color=pt.get("title", "#e2e8f0"), fontsize=10)
            ax.set_ylim(0, 1.15)
            ax.set_title(title, color=pt.get("title", "#e2e8f0"), fontsize=11)
            ax.tick_params(colors=pt.get("text", "#94a3b8"))
            for sp in ax.spines.values():
                sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
def render(data: dict):
    st.title("🎯 Model Results & Evaluation")
    st.markdown("---")

    clf               = data['clf']
    acc               = data['acc']
    bal_acc           = data['balanced_acc']
    f1_macro          = data['f1_macro']
    precision_m       = data['precision_macro']
    recall_m          = data['recall_macro']
    rmse              = data['rmse']
    log_acc           = data['log_acc']
    log_bal_acc       = data['log_bal_acc']
    cv_scores         = data['cv_scores']
    cv_scores_log     = data['cv_scores_log']
    cm                = data['cm']
    report            = data['report']
    features          = data['features']
    label_names       = data['label_names']
    logistic_pipeline = data['logistic_pipeline']
    X_test            = data['X_test']
    y_test            = data['y_test']
    y_pred_dt         = data['y_pred']
    pt = data.get("plot_theme", {})

    # ── Compute LR metrics ───────────────────────────────────────────────────
    y_pred_log  = logistic_pipeline.predict(X_test)
    y_proba_log = logistic_pipeline.predict_proba(X_test)
    log_f1        = f1_score(y_test, y_pred_log, average="macro")
    log_precision = precision_score(y_test, y_pred_log, average="macro", zero_division=0)
    log_recall    = recall_score(y_test, y_pred_log, average="macro", zero_division=0)
    y_true_oh     = np.eye(len(label_names))[y_test]
    log_rmse      = np.sqrt(mean_squared_error(y_true_oh, y_proba_log))
    cm_log        = confusion_matrix(y_test, y_pred_log)
    report_log    = classification_report(y_test, y_pred_log, target_names=label_names, output_dict=True)

    # ── Compute bias SEPARATELY for each model using its own predictions ─────
    dt_bias_sex, dt_bias_age, dt_bias_summary = _compute_bias(X_test, y_test, y_pred_dt)
    lr_bias_sex, lr_bias_age, lr_bias_summary = _compute_bias(X_test, y_test, y_pred_log)

    # ── TOP-LEVEL TABS ───────────────────────────────────────────────────────
    tab_dt, tab_lr, tab_cmp = st.tabs([
        "🌳 Decision Tree (Hunt's)",
        "📈 Logistic Regression",
        "⚖️ Model Comparison",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — DECISION TREE
    # ════════════════════════════════════════════════════════════════════════
    with tab_dt:
        section("Performance Metrics — Decision Tree")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card(f"{acc*100:.1f}%",              "Test Accuracy"),          unsafe_allow_html=True)
        c2.markdown(metric_card(f"{bal_acc*100:.1f}%",          "Balanced Accuracy"),      unsafe_allow_html=True)
        c3.markdown(metric_card(f"{f1_macro*100:.1f}%",         "F1 Macro"),               unsafe_allow_html=True)
        c4.markdown(metric_card(f"{cv_scores.mean()*100:.1f}%", "CV Balanced Acc (Mean)"), unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        c5.markdown(metric_card(f"{precision_m*100:.1f}%", "Precision Macro"),                          unsafe_allow_html=True)
        c6.markdown(metric_card(f"{recall_m*100:.1f}%",    "Recall Macro"),                             unsafe_allow_html=True)
        c7.markdown(metric_card(f"{rmse:.3f}",             "RMSE (Probabilities)", "Lower is better"),  unsafe_allow_html=True)

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

        section("Classification Report")
        st.dataframe(pd.DataFrame(report).T.round(3).style.background_gradient(cmap='Purples'),
                     use_container_width=True)

        section("Cross-Validation Scores (5-Fold, Balanced Accuracy)")
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        ax.bar([f'Fold {i+1}' for i in range(5)], cv_scores,
               color=['#7c3aed','#8b5cf6','#a78bfa','#c4b5fd','#ddd6fe'] if data.get("ui_theme") == "Dark"
               else  ['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#93c5fd'],
               edgecolor=pt.get("spine", "#334155"))
        ax.axhline(cv_scores.mean(), color=pt.get("accent", "#f472b6"), linestyle='--', linewidth=1.5,
                   label=f'Mean BA: {cv_scores.mean():.3f}')
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"))
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Bias — uses DT predictions only
        section("Bias Check — Decision Tree", "Based on Decision Tree predictions on the test set")
        st.markdown("""
        <div class="info-box">
        These gaps reflect <strong>where the Decision Tree specifically makes unfair errors</strong>
        across sex and age groups. Even though both models train on the same SMOTE-balanced data,
        their different decision boundaries can produce very different fairness profiles.
        The Logistic Regression tab shows its own independent bias check.
        </div>""", unsafe_allow_html=True)
        _render_bias_section(dt_bias_sex, dt_bias_age, dt_bias_summary, pt)

        section("Feature Importance")
        fi_df = (pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_})
                   .sort_values('Importance', ascending=True))
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        ax.barh(fi_df['Feature'], fi_df['Importance'],
                color=pt.get("accent", "#7c3aed"), edgecolor=pt.get("spine", "#334155"))
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        ax.set_xlabel('Importance', color=pt.get("text", "#94a3b8"))
        ax.set_title('Feature Importance from Decision Tree', color=pt.get("title", "#e2e8f0"), fontsize=13)
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

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

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — LOGISTIC REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    with tab_lr:
        section("Performance Metrics — Logistic Regression")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card(f"{log_acc*100:.1f}%",              "Test Accuracy"),          unsafe_allow_html=True)
        c2.markdown(metric_card(f"{log_bal_acc*100:.1f}%",          "Balanced Accuracy"),      unsafe_allow_html=True)
        c3.markdown(metric_card(f"{log_f1*100:.1f}%",               "F1 Macro"),               unsafe_allow_html=True)
        c4.markdown(metric_card(f"{cv_scores_log.mean()*100:.1f}%", "CV Balanced Acc (Mean)"), unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        c5.markdown(metric_card(f"{log_precision*100:.1f}%", "Precision Macro"),                         unsafe_allow_html=True)
        c6.markdown(metric_card(f"{log_recall*100:.1f}%",    "Recall Macro"),                            unsafe_allow_html=True)
        c7.markdown(metric_card(f"{log_rmse:.3f}",           "RMSE (Probabilities)", "Lower is better"), unsafe_allow_html=True)

        section("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        sns.heatmap(cm_log, annot=True, fmt='d',
                    cmap="Blues" if data.get("ui_theme") == "Light" else "Purples",
                    ax=ax, xticklabels=label_names, yticklabels=label_names,
                    linewidths=1, linecolor=pt.get("fig_bg", "#0f172a"),
                    annot_kws={'size': 14, 'weight': 'bold', 'color': pt.get("title", "#e2e8f0")})
        ax.set_xlabel('Predicted', color=pt.get("text", "#94a3b8"))
        ax.set_ylabel('Actual',    color=pt.get("text", "#94a3b8"))
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        section("Classification Report")
        st.dataframe(pd.DataFrame(report_log).T.round(3).style.background_gradient(cmap='Blues'),
                     use_container_width=True)

        section("Cross-Validation Scores (5-Fold, Balanced Accuracy)")
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        ax.bar([f'Fold {i+1}' for i in range(5)], cv_scores_log,
               color=['#0ea5e9','#38bdf8','#7dd3fc','#bae6fd','#e0f2fe'] if data.get("ui_theme") == "Dark"
               else  ['#0369a1','#0284c7','#0ea5e9','#38bdf8','#7dd3fc'],
               edgecolor=pt.get("spine", "#334155"))
        ax.axhline(cv_scores_log.mean(), color=pt.get("accent", "#f472b6"), linestyle='--', linewidth=1.5,
                   label=f'Mean BA: {cv_scores_log.mean():.3f}')
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"))
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Bias — uses LR predictions only
        section("Bias Check — Logistic Regression", "Based on Logistic Regression predictions on the test set")
        st.markdown("""
        <div class="info-box">
        These gaps reflect <strong>where Logistic Regression specifically makes unfair errors</strong>.
        Because LR draws a linear decision boundary, it may be systematically more or less accurate
        for older patients or a specific sex — differently from the Decision Tree.
        Compare with the Decision Tree tab to see which model is fairer for your use case.
        </div>""", unsafe_allow_html=True)
        _render_bias_section(lr_bias_sex, lr_bias_age, lr_bias_summary, pt)

        section("Model Coefficients (per class)")
        lr_clf  = logistic_pipeline.named_steps["clf"]
        coef_df = pd.DataFrame(lr_clf.coef_, index=label_names, columns=features).T.round(4)
        st.markdown("""
        <div class="info-box">
        Each column is one class. A large <strong>positive</strong> weight means that feature
        strongly increases the predicted probability of that class; a large <strong>negative</strong>
        weight reduces it.
        </div>""", unsafe_allow_html=True)
        st.dataframe(coef_df.style.background_gradient(cmap='RdBu_r', axis=None), use_container_width=True)

        fig, axes = plt.subplots(1, len(label_names), figsize=(14, 5), facecolor=pt.get("fig_bg", "#0f172a"))
        palette = ['#ef4444', '#22c55e', '#f59e0b']
        for i, (ax, cls) in enumerate(zip(axes, label_names)):
            ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
            vals   = coef_df[cls].values
            colors = [palette[i] if v >= 0 else '#64748b' for v in vals]
            ax.barh(features, vals, color=colors, edgecolor=pt.get("spine", "#334155"))
            ax.axvline(0, color=pt.get("spine", "#334155"), linewidth=0.8)
            ax.set_title(cls, color=pt.get("title", "#e2e8f0"), fontsize=11, fontweight='bold')
            ax.tick_params(colors=pt.get("text", "#94a3b8"), labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.suptitle("Logistic Regression Coefficients by Class", color=pt.get("title", "#e2e8f0"), fontsize=13)
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab_cmp:
        section("Side-by-Side Model Comparison", "Decision Tree (Hunt's) vs Logistic Regression")

        # Performance table
        metrics_data = {
            "Metric": [
                "Test Accuracy", "Balanced Accuracy", "F1 Macro",
                "Precision Macro", "Recall Macro", "RMSE (Proba)",
                "CV Balanced Acc (Mean)", "CV Balanced Acc (Std)",
            ],
            "Decision Tree": [
                f"{acc*100:.1f}%",              f"{bal_acc*100:.1f}%",
                f"{f1_macro*100:.1f}%",         f"{precision_m*100:.1f}%",
                f"{recall_m*100:.1f}%",         f"{rmse:.3f}",
                f"{cv_scores.mean()*100:.1f}%", f"{cv_scores.std()*100:.1f}%",
            ],
            "Logistic Regression": [
                f"{log_acc*100:.1f}%",               f"{log_bal_acc*100:.1f}%",
                f"{log_f1*100:.1f}%",                f"{log_precision*100:.1f}%",
                f"{log_recall*100:.1f}%",            f"{log_rmse:.3f}",
                f"{cv_scores_log.mean()*100:.1f}%",  f"{cv_scores_log.std()*100:.1f}%",
            ],
        }
        cmp_df = pd.DataFrame(metrics_data).set_index("Metric")

        def highlight_winner(row):
            try:
                dt_val = float(row["Decision Tree"].replace('%', ''))
                lr_val = float(row["Logistic Regression"].replace('%', ''))
                dt_wins = dt_val <= lr_val if "RMSE" in row.name else dt_val >= lr_val
                return [
                    'background-color: #14532d; color: #86efac' if dt_wins     else '',
                    'background-color: #14532d; color: #86efac' if not dt_wins else '',
                ]
            except Exception:
                return ['', '']

        st.dataframe(cmp_df.style.apply(highlight_winner, axis=1), use_container_width=True)
        st.caption("🟢 Green cell = better performing model for that metric.")

        # Bar chart
        section("Metric Bar Chart Comparison")
        bar_labels = ["Test Accuracy", "Balanced Accuracy", "F1 Macro",
                      "Precision Macro", "Recall Macro", "CV Balanced Acc (Mean)"]
        dt_vals = [acc, bal_acc, f1_macro, precision_m, recall_m, cv_scores.mean()]
        lr_vals = [log_acc, log_bal_acc, log_f1, log_precision, log_recall, cv_scores_log.mean()]
        x, w = np.arange(len(bar_labels)), 0.35

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        b1_ = ax.bar(x - w/2, dt_vals, w, label="Decision Tree",       color="#7c3aed", edgecolor=pt.get("spine", "#334155"))
        b2_ = ax.bar(x + w/2, lr_vals, w, label="Logistic Regression", color="#0ea5e9", edgecolor=pt.get("spine", "#334155"))
        for bar in list(b1_) + list(b2_):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height()*100:.1f}%', ha='center',
                    color=pt.get("title", "#e2e8f0"), fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=15, ha='right', color=pt.get("text", "#94a3b8"), fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", color=pt.get("text", "#94a3b8"))
        ax.set_title("Model Performance Comparison", color=pt.get("title", "#e2e8f0"), fontsize=13)
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"), fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Side-by-side confusion matrices
        section("Confusion Matrices — Side by Side")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=pt.get("fig_bg", "#0f172a"))
        for ax, matrix, title, cmap in [
            (axes[0], cm,     "Decision Tree",       "Purples"),
            (axes[1], cm_log, "Logistic Regression", "Blues"),
        ]:
            ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
            sns.heatmap(matrix, annot=True, fmt='d', cmap=cmap, ax=ax,
                        xticklabels=label_names, yticklabels=label_names,
                        linewidths=1, linecolor=pt.get("fig_bg", "#0f172a"),
                        annot_kws={'size': 13, 'weight': 'bold', 'color': pt.get("title", "#e2e8f0")})
            ax.set_title(title, color=pt.get("title", "#e2e8f0"), fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', color=pt.get("text", "#94a3b8"))
            ax.set_ylabel('Actual',    color=pt.get("text", "#94a3b8"))
            ax.tick_params(colors=pt.get("text", "#94a3b8"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # CV line chart
        section("Cross-Validation Score Distribution")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        folds = [f'Fold {i+1}' for i in range(5)]
        ax.plot(folds, cv_scores,     marker='o', linewidth=2, color='#7c3aed',
                label=f'Decision Tree  (mean={cv_scores.mean():.3f})')
        ax.plot(folds, cv_scores_log, marker='s', linewidth=2, color='#0ea5e9',
                label=f'Logistic Reg   (mean={cv_scores_log.mean():.3f})')
        ax.fill_between(folds, cv_scores,     alpha=0.15, color='#7c3aed')
        ax.fill_between(folds, cv_scores_log, alpha=0.15, color='#0ea5e9')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Balanced Accuracy", color=pt.get("text", "#94a3b8"))
        ax.set_title("5-Fold CV Balanced Accuracy per Fold", color=pt.get("title", "#e2e8f0"), fontsize=13)
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"), fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Bias comparison ───────────────────────────────────────────────────
        section("Bias Comparison — Both Models", "Same test set, different predictions → different fairness profiles")
        st.markdown("""
        <div class="info-box">
        Both models were trained on the <strong>same SMOTE-balanced data</strong>, yet they can produce
        <strong>different disparity gaps</strong> because they draw fundamentally different decision
        boundaries. The Decision Tree splits on threshold rules; Logistic Regression draws a linear
        hyperplane. Either can be systematically biased against a sex or age group in its own way.<br><br>
        A model with <strong>lower gaps is fairer</strong> — it makes similarly accurate predictions
        regardless of a patient's demographic group.
        </div>""", unsafe_allow_html=True)

        bias_cmp_data = {
            "Bias Metric": [
                "Sex — Accuracy Gap", "Sex — F1 Gap",
                "Age — Accuracy Gap", "Age — F1 Gap",
            ],
            "Decision Tree": [
                f"{dt_bias_summary['sex_accuracy_gap']*100:.1f}%",
                f"{dt_bias_summary['sex_f1_gap']*100:.1f}%",
                f"{dt_bias_summary['age_accuracy_gap']*100:.1f}%",
                f"{dt_bias_summary['age_f1_gap']*100:.1f}%",
            ],
            "Logistic Regression": [
                f"{lr_bias_summary['sex_accuracy_gap']*100:.1f}%",
                f"{lr_bias_summary['sex_f1_gap']*100:.1f}%",
                f"{lr_bias_summary['age_accuracy_gap']*100:.1f}%",
                f"{lr_bias_summary['age_f1_gap']*100:.1f}%",
            ],
        }
        bias_cmp_df = pd.DataFrame(bias_cmp_data).set_index("Bias Metric")

        def highlight_fairer(row):
            try:
                dt_val = float(row["Decision Tree"].replace('%', ''))
                lr_val = float(row["Logistic Regression"].replace('%', ''))
                dt_fairer = dt_val <= lr_val   # lower gap = fairer
                return [
                    'background-color: #14532d; color: #86efac' if dt_fairer     else '',
                    'background-color: #14532d; color: #86efac' if not dt_fairer else '',
                ]
            except Exception:
                return ['', '']

        st.dataframe(bias_cmp_df.style.apply(highlight_fairer, axis=1), use_container_width=True)
        st.caption("🟢 Green cell = fairer model (smaller disparity gap) for that bias metric.")

        # Bias bar chart
        bias_labels = ["Sex Acc Gap", "Sex F1 Gap", "Age Acc Gap", "Age F1 Gap"]
        dt_gaps = [
            dt_bias_summary['sex_accuracy_gap'], dt_bias_summary['sex_f1_gap'],
            dt_bias_summary['age_accuracy_gap'], dt_bias_summary['age_f1_gap'],
        ]
        lr_gaps = [
            lr_bias_summary['sex_accuracy_gap'], lr_bias_summary['sex_f1_gap'],
            lr_bias_summary['age_accuracy_gap'], lr_bias_summary['age_f1_gap'],
        ]
        xb, wb = np.arange(len(bias_labels)), 0.35

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=pt.get("fig_bg", "#0f172a"))
        ax.set_facecolor(pt.get("ax_bg", "#1e293b"))
        bg1 = ax.bar(xb - wb/2, dt_gaps, wb, label="Decision Tree",       color="#7c3aed", edgecolor=pt.get("spine", "#334155"))
        bg2 = ax.bar(xb + wb/2, lr_gaps, wb, label="Logistic Regression", color="#0ea5e9", edgecolor=pt.get("spine", "#334155"))
        for bar in list(bg1) + list(bg2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{bar.get_height()*100:.1f}%', ha='center',
                    color=pt.get("title", "#e2e8f0"), fontsize=9)
        ax.set_xticks(xb)
        ax.set_xticklabels(bias_labels, color=pt.get("text", "#94a3b8"), fontsize=10)
        ax.set_ylabel("Disparity Gap", color=pt.get("text", "#94a3b8"))
        ax.set_title("Bias Disparity Gaps — Decision Tree vs Logistic Regression\n(lower = fairer)",
                     color=pt.get("title", "#e2e8f0"), fontsize=12)
        ax.tick_params(colors=pt.get("text", "#94a3b8"))
        ax.legend(labelcolor=pt.get("text", "#94a3b8"), facecolor=pt.get("legend_bg", "#1e293b"), fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor(pt.get("spine", "#334155"))
        fig.patch.set_facecolor(pt.get("fig_bg", "#0f172a"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Verdict
        section("Verdict")
        dt_score    = np.mean([acc, bal_acc, f1_macro, cv_scores.mean()])
        lr_score    = np.mean([log_acc, log_bal_acc, log_f1, cv_scores_log.mean()])
        dt_bias_avg = np.mean(dt_gaps)
        lr_bias_avg = np.mean(lr_gaps)
        perf_winner = "Decision Tree (Hunt's)" if dt_score    >= lr_score    else "Logistic Regression"
        fair_winner = "Decision Tree (Hunt's)" if dt_bias_avg <= lr_bias_avg else "Logistic Regression"
        st.markdown(f"""
        <div class="info-box">
        <strong>Performance winner</strong> (avg of Accuracy, Balanced Acc, F1, CV):
        <span style="color:#22c55e;font-size:1.05rem;font-weight:700"> {perf_winner}</span><br><br>
        <strong>Fairness winner</strong> (lower avg disparity gap across sex &amp; age):
        <span style="color:#22c55e;font-size:1.05rem;font-weight:700"> {fair_winner}</span><br><br>
        <strong>Decision Tree strengths:</strong> Interpretable decision paths you can trace split by split,
        handles non-linear boundaries, scale-invariant.<br><br>
        <strong>Logistic Regression strengths:</strong> Smoother, better-calibrated probabilities, less prone
        to overfitting on small datasets, coefficients directly reveal each feature's influence direction.<br><br>
        <strong>Key insight on bias:</strong> The same SMOTE-balanced training set does <em>not</em> guarantee
        the same fairness outcome — each model's decision boundary interacts with the demographic distribution
        of the test set in its own way. Always compute bias independently per model.
        </div>""", unsafe_allow_html=True)