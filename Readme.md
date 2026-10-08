# Machine Failure Prediction

An applied machine learning case study for industrial **predictive maintenance** using the AI4I 2020 dataset. The project compares traditional linear baselines with feature-engineered and hyperparameter-tuned **Random Forest** classifiers to identify machine failure risks from operating sensor telemetry.

---

## Project Overview

Unplanned machine breakdowns cause costly production downtime, emergency maintenance expenses, and equipment degradation. The objective of this project is to build a binary classification model that accurately flags machines at risk of failure based on continuous operating conditions (temperature, speed, torque, tool wear, and machine type).

In predictive maintenance, machine failure is inherently a **severe class imbalance problem** (failures represent only 3.39% of operating records). A naive model that predicts "no failure" 100% of the time achieves 96.61% accuracy while failing completely at its core objective. Consequently, this project focuses on optimizing **Precision, Recall, F1-Score, and False Alarm rates** rather than accuracy alone.

### Experimental Progression

The project was executed in two sequential phases:

- **Task 1 (Baseline & Initial Modeling):**
  - Evaluated unweighted and class-weighted **Logistic Regression** baselines.
  - Implemented an initial **Random Forest** classifier on 6 raw operational features.
  - Explored hyperparameter optimization via `GridSearchCV` and conducted a decision threshold tuning experiment.
- **Task 2 (Advanced Feature Engineering & Selection):**
  - Formulated 5 domain-specific physical and thermal interaction features.
  - Evaluated feature importance and isolated the top 8 predictive inputs.
  - Trained and tuned an improved **Feature-Selected Random Forest**, achieving **88.19% F1-score** and reducing false alarms to 3 out of 1,932 healthy machines.

> **Note:** Task 2 represents the stronger, final machine learning experiment in this repository.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** (`data/ai4i2020.csv`), an industrial predictive-maintenance benchmark dataset.

### Summary Statistics

| Property | Description |
|---|---|
| **Total Records** | 10,000 observations |
| **Total Columns** | 14 variables (features, identifiers, failure modes) |
| **Target Variable** | `Machine failure` (`0` = No Failure, `1` = Failure) |
| **Class Distribution** | 9,661 Non-Failures (**96.61%**) vs. 339 Failures (**3.39%**) |
| **Data Quality** | 0 missing values, 0 duplicate rows |

### Input Features

The predictive feature matrix uses the following raw sensor measurements:
- **`Type`**: Machine quality variant (`L` = Low: 6,000, `M` = Medium: 2,997, `H` = High: 1,003).
- **`Air temperature [K]`**: Ambient operating temperature (295.3 – 304.5 K).
- **`Process temperature [K]`**: Internal process temperature (305.7 – 313.8 K).
- **`Rotational speed [rpm]`**: Spindle rotational speed (1,168 – 2,886 rpm).
- **`Torque [Nm]`**: Torque applied during operation (3.8 – 76.6 Nm).
- **`Tool wear [min]`**: Cumulative tool usage time (0 – 253 minutes).

### Leakage Prevention & Excluded Variables
To prevent target leakage and ensure generalizability, the following columns were excluded from model features:
- **Identifiers**: `UDI` (row index) and `Product ID` (serial string).
- **Post-Hoc Failure Modes**: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure). These columns are diagnostic breakdown classifications available only after a failure occurs.

---

## Methodology

### Data Preparation
- **Splitting Strategy**: 80% Train ($N=8,000$) / 20% Test ($N=2,000$) using `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
- **Stratification**: Guaranteed the ~3.39% failure rate was preserved in both the training set (271 failures / 7,729 healthy) and the test set (68 failures / 1,932 healthy).
- **Pipeline Preprocessing**:
  - Continuous numerical features transformed via `StandardScaler()`.
  - Categorical feature (`Type`) encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`.
  - Encapsulated inside `ColumnTransformer` and `Pipeline` objects to ensure preprocessing parameters were fitted strictly on the training partition.

---

### Task 1: Baseline and Threshold Experiment

In Task 1, models were trained on the 6 raw operational features:

1. **Logistic Regression (Unweighted)**: Achieved 96.75% accuracy, but a recall of only **10.29%** (caught only 7 of 68 failures). The model defaulted to predicting the majority class.
2. **Logistic Regression (`class_weight='balanced'`)**: Improved recall to **82.35%**, but precision collapsed to **14.18%**, generating over 300 false alarms across 2,000 machines.
3. **Random Forest (Default, `class_weight='balanced'`)**: Outperformed linear models by capturing non-linear interactions (Accuracy: 98.10%, Precision: 73.44%, Recall: 69.12%, F1: 0.7121).
4. **Random Forest (Tuned via 5-Fold GridSearchCV)**: Best parameters (`n_estimators=200, max_depth=None, min_samples_leaf=1`) produced 98.20% accuracy, 75.00% precision, 70.59% recall, and 0.7273 F1.
5. **Threshold-Adjusted Random Forest (Threshold = 0.525)**: Adjusted cutoff from default 0.50 to 0.525 based on the test set precision-recall curve, yielding 98.30% accuracy, 77.42% precision, 70.59% recall, and **0.7385 F1** (48 of 68 failures caught, 14 false alarms).

> **Methodological Note on Task 1 Threshold:** The decision threshold of `0.525` was identified by maximizing F1 on the test-set precision-recall curve. Because the test set was utilized during threshold selection, the 0.7385 F1 score should be understood as an exploratory threshold optimization experiment rather than an untouched holdout benchmark.

---

### Task 2: Feature Engineering and Feature Selection

Task 2 investigated whether domain-informed physical and thermal interaction features could improve detection performance without compromising false alarm rates.

#### 1. Engineered Features

Five domain variables were formulated based on exploratory data analysis:

1. **`Temperature Difference [K]`**: `Process temperature [K] - Air temperature [K]`. Measures the thermal gradient and heat dissipation efficiency across the machine.
2. **`Mechanical Power Proxy`**: `Torque [Nm] * Rotational speed [rpm]`. Directly proportional to the mechanical power loading on the drive system.
3. **`Temperature Ratio`**: `Process temperature [K] / Air temperature [K]`. Captures relative thermal strain.
4. **`Torque Speed Ratio`**: `Torque [Nm] / Rotational speed [rpm]`. Characterizes the torque-to-speed operational regime.
5. **`Tool Wear Category`**: Discretized `Tool wear [min]` into bins (`Low` $\le 50$, `Medium` $50-150$, `High` $>150$).

#### 2. Feature Importance & Selection

A Random Forest was fitted on the 11-feature training set (`X_train_fe`). Analysis of Gini feature importances showed that mechanical loading and thermal interaction variables provided the strongest discriminative signals:
- `Mechanical Power Proxy` (~20.5%)
- `Rotational speed [rpm]` (~15.6%)
- `Torque [Nm]` (~14.1%)
- `Torque Speed Ratio` (~12.9%)
- `Tool wear [min]` (~11.9%)
- `Temperature Ratio` (~8.1%)
- `Temperature Difference [K]` (~7.0%)

Based on these rankings, the input space was refined to **8 selected features**:
`['Type', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Mechanical Power Proxy', 'Torque Speed Ratio', 'Temperature Ratio', 'Temperature Difference [K]']`.

---

## Results

All models were evaluated on the held-out test partition ($N=2,000$, containing 1,932 healthy machines and 68 failures).

### Comparative Performance Table

| Phase | Model Configuration | Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positives | False Negatives |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Task 1** | Logistic Regression (Unweighted) | 6 Raw | 96.75% | 63.64% | 10.29% | 0.1772 | — | — | 4 | 61 |
| **Task 1** | Logistic Regression (Balanced) | 6 Raw | 82.45% | 14.18% | 82.35% | 0.2419 | — | — | 340 | 12 |
| **Task 1** | Random Forest (Default) | 6 Raw | 98.10% | 73.44% | 69.12% | 0.7121 | — | — | 17 | 21 |
| **Task 1** | Random Forest (Tuned, Cutoff 0.50) | 6 Raw | 98.20% | 75.00% | 70.59% | 0.7273 | — | — | 16 | 20 |
| **Task 1** | Random Forest (Tuned, Cutoff 0.525)* | 6 Raw | 98.30% | 77.42% | 70.59% | 0.7385 | 0.9638 | — | 14 | 20 |
| **Task 2** | Random Forest (All Engineered) | 11 Feat. | 99.10% | 94.64% | 77.94% | 0.8548 | — | — | 3 | 15 |
| **Task 2** | **Random Forest (8 Selected Features)** | **8 Feat.** | **99.25%** | **94.92%** | **82.35%** | **0.8819** | **0.9679** | **0.8887** | **3** | **12** |
| **Task 2** | Random Forest (Tuned on 8 Features) | 8 Feat. | 99.20% | 94.83% | 80.88% | 0.8730 | — | — | 3 | 13 |

*\*Task 1 threshold tuned directly on test-set PR curve.*

### Final Model Confusion Matrix (Task 2 Feature-Selected RF)

```text
                             Predicted Normal (0)    Predicted Failure (1)
Actual Normal (0) [1,932]            1,929                     3  (False Positives)
Actual Failure (1)   [68]               12                    56  (True Positives)
```

### Key Engineering Insights
1. **Dramatic False Alarm Reduction**: Compared to the Task 1 tuned model, Task 2 reduced false positives from **14 down to 3** across 1,932 non-failure cases (a **78.6% reduction in false alarms**).
2. **Improved Failure Detection**: Caught **56 out of 68 real failures (82.35% recall)** compared to 48 in Task 1, while elevating precision to **94.92%**.
3. **Imbalance Metrics vs. Accuracy**: While accuracy only shifted from 98.30% to 99.25%, the **F1-score increased from 73.85% to 88.19%** (+14.34 points), demonstrating why precision/recall/F1 metrics are essential for imbalanced anomaly detection.

---

## Model Artifacts

The `models/` directory contains three serialized model artifacts:

- **`models/final_model.pkl`**: The Task 1 trained `Pipeline` (ColumnTransformer on 6 raw features + Random Forest classifier).
- **`models/final_threshold.pkl`**: The Task 1 optimal decision threshold scalar (`0.525`).
- **`models/final_machine_failure_model.pkl`**: The Task 2 final `Pipeline` (`ColumnTransformer` scaling/encoding the 8 selected features + Random Forest classifier).

### Current Task 2 Inference Limitation
The saved Task 2 pipeline (`final_machine_failure_model.pkl`) expects a DataFrame that **already contains the engineered feature columns** (`Mechanical Power Proxy`, `Temperature Difference [K]`, etc.). Because feature engineering was executed in pandas prior to calling the pipeline in the notebook, `final_machine_failure_model.pkl` does not currently transform a completely raw 6-feature sensor record directly.

---

## Evaluation and Leakage Checks

The project workflow was verified against data leakage risks:

- **Split Isolation**: An 80/20 stratified split (`random_state=42`) separated training and test data.
- **Training-Only Feature Importance**: Feature importance rankings used to select the 8 final features were extracted from an estimator fitted strictly on `X_train_fe`.
- **Training-Only Preprocessing**: `StandardScaler` and `OneHotEncoder` parameters were fitted strictly on `X_train_sel` inside the pipeline.
- **Training-Only Tuning**: `GridSearchCV` evaluated hyperparameter configurations using 5-fold cross-validation solely on the training partition.
- **Methodological Transparency**: The decision threshold tuning in Task 1 was explicitly identified as having utilized the test-set PR curve, whereas Task 2 evaluated its final selected model with the default 0.50 threshold on the holdout test set.

---

## Limitations

- **Dataset Nature**: The AI4I 2020 dataset is an industrial benchmark/synthetic dataset; performance may vary under complex factory conditions with unmodeled physical variables (vibration, acoustics, humidity).
- **Class Imbalance Sensitivity**: Failures represent 3.39% of historical records. Operational deployment would require ongoing monitoring for concept drift and shifts in failure distribution.
- **Offline Batch Pipeline**: The current codebase operates in an offline notebook environment without live sensor stream integration or telemetry ingestion.
- **Inference Preprocessing Dependency**: The Task 2 pipeline requires engineered features to be computed upstream before invoking `predict()`.

---

## Project Structure

```text
Machine_Failure_Prediction/
├── data/
│   └── ai4i2020.csv                      # Canonical dataset (10,000 records, 14 columns)
├── models/
│   ├── final_model.pkl                   # Task 1 Pipeline (6 raw features, class-weighted RF)
│   ├── final_threshold.pkl               # Task 1 decision threshold scalar (0.525)
│   └── final_machine_failure_model.pkl   # Task 2 Pipeline (8 selected features, tuned RF)
├── Notebook/
│   └── 01_data_exploration.ipynb        # Complete 87-cell workflow (EDA, Task 1, Task 2)
├── .gitignore                            # Environment and cache ignore rules
├── Readme.md                             # Applied ML case study documentation
├── requirements.txt                      # Project dependency list
├── Task2_Technical_report.md             # Technical documentation for Task 2 (Feature Engineering)
└── Technical_Report.md                   # Technical documentation for Task 1 (Baseline & Threshold)
```

---

## Reproducing the Work

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Quantumdata66/Machine_Failure_Prediction.git
cd Machine_Failure_Prediction
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Workflow Notebook
```bash
jupyter notebook Notebook/01_data_exploration.ipynb
```
Run all cells sequentially from top to bottom to reproduce the exploratory data analysis, Task 1 baseline modeling, Task 2 feature engineering, cross-validation, and metric reports.

---

## Key Takeaways

This project illustrates core applied machine learning principles:
- **Navigating Class Imbalance**: Demonstrates why unweighted linear models fail on rare failure classes and how tree-based ensembles provide superior precision-recall trade-offs.
- **Physics-Informed Feature Engineering**: Shows how deriving mechanical power and thermal gradient features improved failure class F1 from 71.21% to 88.19%.
- **Objective Evaluation**: Prioritizes False Alarm reduction and Minority-Class Recall over misleading raw accuracy metrics.
- **Methodological Discipline**: Maintains clear separation between training partitions and test holdouts, identifying and documenting threshold-tuning trade-offs.

---

## Future Improvements

- **Custom Scikit-Learn Transformers**: Encapsulate the 5 domain feature calculations into a `BaseEstimator` / `TransformerMixin` class to allow `Pipeline` to accept raw 6-feature telemetry records directly during inference.
- **Modular Python Codebase**: Refactor notebook logic into modular scripts (`train.py`, `evaluate.py`, `predict.py`) for clean separation of experimentation and serving.
- **Cost-Weighted Threshold Optimization**: Calibrate operational thresholds based on concrete industrial monetary costs (cost of 1 hour of unplanned downtime vs. cost of 1 preemptive inspection).
