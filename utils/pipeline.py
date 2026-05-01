import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


@st.cache_data
def load_and_preprocess():
    """Load OASIS data and train a leakage-safe, reliable pipeline."""

    # ── LOAD ────────────────────────────────────────────────────────────────
    df = pd.read_csv("data/oasis_longitudinal.csv")
    df['M/F'] = LabelEncoder().fit_transform(df['M/F'])   # M=1, F=0

    features = ['Age', 'M/F', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    target   = 'Group'

    df_model = df[features + [target]].dropna(subset=[target]).copy()

    X = df_model[features]
    le = LabelEncoder().fit(df_model[target])
    y  = le.transform(df_model[target])
    label_names = le.classes_

    # Split FIRST to prevent leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Required three-step preprocessing + Hunt's Decision Tree classifier.
    tree_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            ccp_alpha=0.002,
            random_state=42,
        )),
    ])
    tree_pipeline.fit(X_train, y_train)

    # Use direct tree probabilities (no calibration).
    model_pipeline = tree_pipeline

    # Optional softer baseline for smoother probability changes.
    logistic_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    logistic_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)
    y_pred_log = logistic_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    y_true_one_hot = np.eye(len(label_names))[y_test]
    rmse = np.sqrt(mean_squared_error(y_true_one_hot, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    log_acc = accuracy_score(y_test, y_pred_log)
    log_bal_acc = balanced_accuracy_score(y_test, y_pred_log)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(tree_pipeline, X, y, cv=cv, scoring="balanced_accuracy")
    cv_scores_log = cross_val_score(logistic_pipeline, X, y, cv=cv, scoring="balanced_accuracy")

    # For "before vs after SMOTE" visualization, show class distribution of full data.
    viz_imputer = SimpleImputer(strategy="mean")
    viz_scaler = RobustScaler()
    X_viz = viz_scaler.fit_transform(viz_imputer.fit_transform(X))
    X_resampled_viz, y_resampled_viz = SMOTE(random_state=42).fit_resample(X_viz, y)

    class_dist_before = dict(zip(*np.unique(y, return_counts=True)))
    class_dist_after = dict(zip(*np.unique(y_resampled_viz, return_counts=True)))

    # Bias / fairness check on protected and demographic slices.
    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["y_true"] = y_test
    eval_df["y_pred"] = y_pred
    eval_df["sex_group"] = eval_df["M/F"].map({0: "Female", 1: "Male"}).fillna("Unknown")

    age_bins = [-np.inf, 69, 79, np.inf]
    age_labels = ["<70", "70-79", "80+"]
    eval_df["age_group"] = pd.cut(eval_df["Age"], bins=age_bins, labels=age_labels)
    eval_df["age_group"] = eval_df["age_group"].astype(str).replace("nan", "Unknown")

    def _group_metrics(frame: pd.DataFrame, group_col: str):
        rows = []
        for group_name, group_frame in frame.groupby(group_col):
            if len(group_frame) == 0:
                continue
            rows.append({
                "group": str(group_name),
                "n_samples": int(len(group_frame)),
                "accuracy": accuracy_score(group_frame["y_true"], group_frame["y_pred"]),
                "f1_macro": f1_score(group_frame["y_true"], group_frame["y_pred"], average="macro", zero_division=0),
                "recall_macro": recall_score(group_frame["y_true"], group_frame["y_pred"], average="macro", zero_division=0),
            })
        return pd.DataFrame(rows)

    bias_by_sex = _group_metrics(eval_df, "sex_group")
    bias_by_age = _group_metrics(eval_df, "age_group")

    def _max_disparity(group_df: pd.DataFrame, metric: str):
        if group_df.empty or group_df[metric].isna().all():
            return 0.0
        return float(group_df[metric].max() - group_df[metric].min())

    bias_summary = {
        "sex_accuracy_gap": _max_disparity(bias_by_sex, "accuracy"),
        "sex_f1_gap": _max_disparity(bias_by_sex, "f1_macro"),
        "age_accuracy_gap": _max_disparity(bias_by_age, "accuracy"),
        "age_f1_gap": _max_disparity(bias_by_age, "f1_macro"),
    }

    # Keep direct access to fitted tree for interpretability pages.
    clf = tree_pipeline.named_steps["clf"]

    return {
        "model_pipeline": model_pipeline,
        "tree_pipeline": tree_pipeline,
        "logistic_pipeline": logistic_pipeline,
        "clf": clf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "acc": acc,
        "balanced_acc": bal_acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "rmse": rmse,
        "cm": cm,
        "report": report,
        "log_acc": log_acc,
        "log_bal_acc": log_bal_acc,
        "cv_scores": cv_scores,
        "cv_scores_log": cv_scores_log,
        "features": features,
        "label_names": label_names,
        "class_dist_before": class_dist_before,
        "class_dist_after": class_dist_after,
        "bias_by_sex": bias_by_sex,
        "bias_by_age": bias_by_age,
        "bias_summary": bias_summary,
        "df": df_model,
    }
