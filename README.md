# AlzDetect AI

AlzDetect AI is a Streamlit-based machine learning application for multi-class Alzheimer's progression classification using the OASIS longitudinal dataset.

## Features

- End-to-end ML pipeline: mean imputation -> robust scaling -> SMOTE -> Decision Tree classifier.
- Real-time web prediction interface with auto-updating inference.
- Evaluation dashboard with key metrics: Accuracy, Balanced Accuracy, F1, Precision, Recall, RMSE (probability error), and confusion matrix.
- Bias check dashboard by sex and age groups with group-wise performance and disparity gaps.

## Project Structure

- `app.py`: Streamlit entry point and page routing.
- `utils/pipeline.py`: data loading, preprocessing, model training, evaluation, and bias checks.
- `views/`: multipage UI (`overview`, `data_explorer`, `pipeline_explained`, `model_results`, `predict_patient`).
- `report/final_report.tex`: technical report template in LaTeX.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hosting the Web Interface

This app can be hosted on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub.
2. Open Streamlit Cloud and create a new app linked to your repo.
3. Set:
   - Main file path: `app.py`
   - Python version: 3.10+ (recommended)
4. Deploy.

The `Predict Patient` page supports real-time predictions when "Real-time prediction mode" is enabled.

## Evaluation and Bias Check

Current evaluation includes:

- `Accuracy`
- `Balanced Accuracy`
- `F1 (macro)`
- `Precision (macro)`
- `Recall (macro)`
- `RMSE` over class-probability outputs
- 5-fold cross-validation (balanced accuracy)

Bias checks include:

- Group metrics by sex (`Female`, `Male`)
- Group metrics by age bands (`<70`, `70-79`, `80+`)
- Disparity gaps: max-min across groups for Accuracy and F1

## Documentation and Final Report

A LaTeX technical report template is provided at:

- `report/final_report.tex`

It already includes sections for:

- Methodology
- Architecture diagrams
- Challenges faced during implementation (environment setup, convergence, class imbalance, fairness)
- Results and analysis
