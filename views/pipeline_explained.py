import streamlit as st
from sklearn.tree import export_text
from components.ui import section


def render(data: dict):
    st.title("⚙️ Pipeline — Concepts Explained")
    st.markdown("---")

    clf      = data['clf']
    features = data['features']

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Mean Imputation",
        "📐 Robust Scaling",
        "🔁 SMOTE",
        "🌳 Hunt's Algorithm",
    ])

    # ── TAB 1: MEAN IMPUTATION ───────────────────────────────────────────────
    with tab1:
        st.markdown("### Step 1 — Mean Imputation")
        st.markdown("""
        <div class="info-box">
        <strong>What is it?</strong> When a dataset has missing values (NaN), we can't just delete those rows — we'd lose too much data.
        Mean Imputation fills each missing value with the <strong>average of that column</strong>.<br><br>
        <strong>Formula:</strong> x_missing = mean(column) = Σxᵢ / n<br><br>
        <strong>Why use it?</strong> Simple, fast, and preserves the dataset size. Best when data is Missing At Random (MAR).<br><br>
        <strong>In our data:</strong> Columns like SES and MMSE have missing values — we fill them with their respective column means.
        </div>""", unsafe_allow_html=True)
        st.code("""
# Python code — new pandas syntax (avoids SettingWithCopyWarning)
for col in features:
    df[col] = df[col].fillna(df[col].mean())
        """, language='python')

    # ── TAB 2: ROBUST SCALING ────────────────────────────────────────────────
    with tab2:
        st.markdown("### Step 2 — Robust Scaling")
        st.markdown("""
        <div class="info-box">
        <strong>What is it?</strong> Scales features so they have similar ranges — but uses <strong>Median</strong> and
        <strong>IQR (Interquartile Range)</strong> instead of mean/std. This makes it <em>robust to outliers</em>.<br><br>
        <strong>Formula:</strong> x_scaled = (x − median) / IQR<br><br>
        <strong>Why not StandardScaler?</strong> StandardScaler uses mean — one extreme outlier can distort all values.
        Robust Scaler ignores outliers by design.<br><br>
        <strong>In our data:</strong> Brain volumes (eTIV) vary widely. Robust scaling handles this gracefully.
        </div>""", unsafe_allow_html=True)
        st.code("""
from sklearn.preprocessing import RobustScaler

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X_train_only)
        """, language='python')

    # ── TAB 3: SMOTE ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Step 3 — SMOTE")
        st.markdown("""
        <div class="info-box">
        <strong>SMOTE = Synthetic Minority Oversampling Technique</strong><br><br>
        Our OASIS dataset is imbalanced: Nondemented=190, Demented=146, Converted=37.
        If we train on this as-is, the model ignores the "Converted" class (only 37 samples).<br><br>
        <strong>How SMOTE works:</strong><br>
        1. Pick a minority sample (e.g., a Converted patient)<br>
        2. Find its K nearest neighbors in the same class<br>
        3. Randomly pick one neighbor<br>
        4. Create a new synthetic point = original + random × (neighbor − original)<br>
        5. Repeat until all classes are balanced<br><br>
        <strong>Result:</strong> 190 → 190 → 190 (all balanced!) — No actual patient data is duplicated.<br><br>
        <strong>Reliability rule:</strong> Apply SMOTE only on the training split, never on the full dataset, to avoid data leakage.
        </div>""", unsafe_allow_html=True)
        st.code("""
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        """, language='python')

    # ── TAB 4: HUNT'S ALGORITHM ──────────────────────────────────────────────
    with tab4:
        st.markdown("### Step 4 — Hunt's Algorithm (Decision Tree)")
        st.markdown("""
        <div class="info-box">
        <strong>What is Hunt's Algorithm?</strong><br>
        It's the foundational recursive procedure for building Decision Trees.<br><br>
        <strong>The 3-step logic:</strong><br>
        1. <strong>Base Case:</strong> If all records belong to one class → make it a leaf node<br>
        2. <strong>Best Split:</strong> Find the feature + threshold that best separates classes (using Gini Impurity or Entropy)<br>
        3. <strong>Recurse:</strong> Split the data and repeat for each child node<br><br>
        <strong>Gini Impurity Formula:</strong> Gini = 1 − Σ(pᵢ)²<br>
        A Gini of 0 = perfectly pure node (all one class). Lower is better.<br><br>
        <strong>sklearn maps directly to Hunt's:</strong> DecisionTreeClassifier uses this exact recursive procedure.
        </div>""", unsafe_allow_html=True)
        st.code("""
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(
    criterion='gini',      # Hunt's splitting criterion
    max_depth=4,           # Limits overfitting
    min_samples_split=20,
    min_samples_leaf=10,
    ccp_alpha=0.002,
    random_state=42
)
clf.fit(X_train_res, y_train_res)
        """, language='python')

        section("Decision Tree Structure (Text, Depth 3)")
        tree_text = export_text(clf, feature_names=features, max_depth=3)
        st.code(tree_text, language='text')
