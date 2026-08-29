# Machine Failure Prediction

A supervised machine learning model that predicts machine failure from real-time operating conditions (temperature, rotational speed, torque, tool wear, machine type) — built for predictive maintenance use cases.

## Overview

Unplanned machine failure is costly. This project trains and compares multiple classifiers on a 10,000-record industrial dataset (3.39% failure rate) to flag at-risk machines before they break down, using proper handling of severe class imbalance.

**Final model:** Tuned Random Forest, custom decision threshold (0.525)
**Performance:** 98.3% accuracy · 77.4% precision · 70.6% recall · F1 0.738 · ROC-AUC 0.964

Full reasoning, EDA findings, and all modeling iterations are documented in [`Technical_Report.md`](./Technical_Report.md).

## Key Results

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression (unweighted) | 0.636 | 0.103 | 0.177 |
| Logistic Regression (balanced) | 0.142 | 0.824 | 0.242 |
| Random Forest (default) | 0.734 | 0.691 | 0.712 |
| **Random Forest (tuned + threshold-optimized)** | **0.774** | **0.706** | **0.738** |

The final model catches **48 of 68 real failures (71%)** in the test set, with only **14 false alarms** across 1,932 healthy machines.

## What Drives Failure (Feature Importance)

1. **Torque [Nm]** — 32%
2. **Rotational speed [rpm]** — 29%
3. **Tool wear [min]** — 21%
4. Air temperature — 10%
5. Process temperature — 6%
6. Machine type (L/M) — under 2% combined

Torque and rotational speed together account for ~61% of the model's decisions, matching the pattern found in EDA: failures cluster when these two operate outside a safe combined range.

## Project Structure

```
├── data/
│   └── ai4i2020.csv         # raw dataset
├── models/
│   ├── final_model.pkl      # trained pipeline (preprocessing + tuned Random Forest)
│   └── final_threshold.pkl  # optimal decision threshold (0.525)
├── Notebook/
│   └── 01_data_exploration.ipynb   # full workflow: EDA → preprocessing → modeling → evaluation
├── Technical_Report.md       # full case study: EDA, modeling iterations, error analysis
├── requirements.txt
└── README.md
```

> This project was built end-to-end in a single Jupyter notebook. A modular `src/` package (separate `train.py` / `evaluate.py` / `predict.py` scripts) is planned for future projects going forward.

## Getting Started

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Run the notebook**

Open `Notebook/01_data_exploration.ipynb` and run all cells top to bottom. This covers the full workflow: EDA → preprocessing → model training (Logistic Regression, Random Forest) → hyperparameter tuning → threshold optimization → evaluation → saving the final model to `models/`.

**3. Use the saved model for predictions**

```python
import joblib
import pandas as pd

model = joblib.load("models/final_model.pkl")
threshold = joblib.load("models/final_threshold.pkl")

machine = pd.DataFrame([{
    "Type": "L",
    "Air temperature [K]": 302.5,
    "Process temperature [K]": 311.0,
    "Rotational speed [rpm]": 2400,
    "Torque [Nm]": 65,
    "Tool wear [min]": 180,
}])

proba = model.predict_proba(machine)[:, 1][0]
prediction = "FAILURE" if proba >= threshold else "NO FAILURE"

print(f"Failure probability: {proba:.3f} -> {prediction}")
```

## Data

The dataset (`data/ai4i2020.csv`) is the AI4I 2020 Predictive Maintenance Dataset — 10,000 records, 6 input features, and a binary failure label.

## Methodology Summary

- **Preprocessing:** `StandardScaler` for numerical features, `OneHotEncoder` for machine type, wrapped in a single `sklearn` `Pipeline` to prevent data leakage
- **Class imbalance:** Addressed via `class_weight="balanced"` rather than resampling, after comparing against an unweighted baseline
- **Model selection:** Logistic Regression (baseline) vs. Random Forest — Random Forest won on every metric due to its ability to capture feature interactions
- **Tuning:** 5-fold cross-validated `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`
- **Threshold optimization:** Selected via maximum F1 on the precision-recall curve, rather than the default 0.5 cutoff

See [`Technical_Report.md`](./Technical_Report.md) for the full write-up, including EDA findings, all model comparisons, error analysis, and limitations.

