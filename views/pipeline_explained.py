import streamlit as st
from sklearn.tree import export_text
from components.ui import section


def render(data: dict):
    st.title("⚙️ Pipeline — Concepts Explained")
    st.markdown("---")

    clf      = data['clf']
    features = data['features']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Mean Imputation",
        "📐 Robust Scaling",
        "🔁 SMOTE",
        "🌳 Hunt's Algorithm",
        "📈 Logistic Regression",
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
        <strong>In our data:</strong> Columns like SES and MMSE have missing values — we fill them with their respective column means.<br><br>
        <strong>Used in both models?</strong> Yes — Mean Imputation is the first step in <em>both</em> the Decision Tree pipeline and the Logistic Regression pipeline. Both models cannot handle NaN values, so this step is always applied first.
        </div>""", unsafe_allow_html=True)
        st.code("""
# Python code — new pandas syntax (avoids SettingWithCopyWarning)
for col in features:
    df[col] = df[col].fillna(df[col].mean())

# In sklearn Pipeline (used for both models):
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X_train)
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
        <strong>In our data:</strong> Brain volumes (eTIV) vary widely. Robust scaling handles this gracefully.<br><br>
        <strong>Used in both models?</strong> Yes — Robust Scaling is applied in both pipelines. While Decision Trees don't strictly
        need feature scaling (splits are threshold-based), it helps keep the SMOTE step well-behaved and makes the pipeline
        consistent. Logistic Regression, however, <em>does</em> benefit strongly from scaling — gradient-based
        optimization converges faster and more reliably when all features share a similar range.
        </div>""", unsafe_allow_html=True)
        st.code("""
from sklearn.preprocessing import RobustScaler

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Key difference: Logistic Regression benefits more from scaling than
# Decision Trees, but we apply it consistently in both pipelines for
# correctness and comparability.
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
        <strong>Reliability rule:</strong> Apply SMOTE only on the training split, never on the full dataset, to avoid data leakage.<br><br>
        <strong>Used in both models?</strong> Yes — SMOTE is applied identically in both pipelines, before the classifier step.
        Class imbalance harms both Decision Trees and Logistic Regression equally, so balancing the training data
        benefits both. After SMOTE, the logistic model sees equal representation of all three classes during training,
        making its decision boundary less biased toward the majority class.
        </div>""", unsafe_allow_html=True)
        st.code("""
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Same exact call in both pipelines — only the classifier that follows differs.
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

    # ── TAB 5: LOGISTIC REGRESSION ───────────────────────────────────────────
    with tab5:
        st.markdown("### Logistic Regression — How It Works")
        st.markdown("""
        <div class="info-box">
        <strong>What is Logistic Regression?</strong><br>
        Despite its name, Logistic Regression is a <em>classification</em> model. It estimates the probability
        that a sample belongs to each class by fitting a linear decision boundary in feature space,
        then passing it through a <strong>sigmoid (logistic) function</strong> to squash outputs into [0, 1].<br><br>
        <strong>For binary classification (e.g., Demented vs Not):</strong><br>
        P(y=1 | X) = 1 / (1 + e^(−(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))<br><br>
        <strong>For multi-class (Nondemented / Demented / Converted):</strong><br>
        sklearn uses <em>One-vs-Rest (OvR)</em> or <em>Softmax (multinomial)</em> — it trains one boundary per
        class and picks the class with the highest probability.<br><br>
        <strong>Same preprocessing?</strong> The exact same Mean Imputation → Robust Scaling → SMOTE
        pipeline is applied before Logistic Regression. Scaling matters <em>more</em> for logistic regression
        than for decision trees because gradient descent (used during fitting) is sensitive to feature magnitude.
        </div>""", unsafe_allow_html=True)

        st.code("""
        from sklearn.linear_model import LogisticRegression
        from imblearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        from imblearn.over_sampling import SMOTE
        
        # Build the full pipeline — identical preprocessing, different classifier
        logistic_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),   # Step 1: fill NaNs
            ("scaler",  RobustScaler()),                   # Step 2: scale features
            ("smote",   SMOTE(random_state=42)),           # Step 3: balance classes
            ("clf", LogisticRegression(
                max_iter=2000,           # More iterations for convergence
                class_weight="balanced", # Extra weight for minority classes
                random_state=42,
            )),
        ])
        
        logistic_pipeline.fit(X_train, y_train)
        
        # ── Prediction on new patient ──────────────────────────────────────
        import pandas as pd
        
        patient = pd.DataFrame([[75, 0, 12, 2, 25, 0.5, 1600, 0.75, 1.10]],
                                columns=features)
        
        # Pipeline automatically runs imputer → scaler → predict
        # (SMOTE is skipped at predict time — only active during fit)
        pred  = logistic_pipeline.predict(patient)          # e.g., array([0]) → 'Converted'
        proba = logistic_pipeline.predict_proba(patient)    # e.g., [[0.12, 0.65, 0.23]]
        
        print("Predicted class:", label_names[pred[0]])
        print("Probabilities:  ", dict(zip(label_names, proba[0].round(3))))
                """, language='python')
        st.markdown("""
        <div class="info-box">
        <strong>Why does Logistic Regression give "softer" probabilities?</strong><br>
        Decision Trees assign probabilities based on the fraction of training samples in a leaf node —
        so many inputs that land in the same leaf get <em>identical</em> probabilities. Logistic Regression
        computes a continuous function of the input, so every unique input gets a slightly different
        probability vector, making the output smoother and more gradient-like as you move the sliders.
        </div>""", unsafe_allow_html=True)