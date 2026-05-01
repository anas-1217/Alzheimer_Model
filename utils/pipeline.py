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
        "df": df_model,
    }
