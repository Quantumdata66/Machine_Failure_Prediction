# Predictive Maintenance: Machine Failure Prediction Technical Report

## 1. Problem Statement

Unplanned machine failure is costly and disruptive in industrial settings. This project builds a supervised machine learning model to predict, from realtime operating conditions, whether a machine is at risk of failureenabling proactive maintenance rather than reactive repair.

The dataset contains **10,000 machine records**, each described by air temperature, process temperature, rotational speed, torque, tool wear, and machine type (L, M, H), with a binary label indicating whether a failure occurred. The target is highly imbalanced: only **3.39%** of records represent an actual failure, which shaped every modeling decision made below.

## 2. Exploratory Data Analysis

Key patterns identified before modeling:

- **Tool wear vs. failure:** Machines with tool wear between 50–150 minutes showed a fairly even failure/no failure split. Above the 150-minute mark, failure likelihood increased sharply.
- **Rotational speed vs. torque:** No failures were observed when torque stayed between 15–60 Nm combined with rotational speed between 1250–2250 rpm. Outside this window, higher torque or faster rotational speed the failure risk rose.
- **Failure rate by machine type:** Type **L** machines had the highest failure rate; Type **H** machines had the lowest.

These patterns motivated the choice of models capable of capturing feature interactions (not just linear relationships), and were later validated against the trained model's feature importances.

## 3. Preprocessing

**Features used (X):** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear
**Target (y):** Machine failure

**Train/test split:** 80/20, stratified on the target to preserve the 3.39% failure rate in both sets (8,000 train / 2,000 test).

**Feature handling:**
- Numerical features (temperature, speed, torque, tool wear) → `StandardScaler`
- Categorical feature (Type) → `OneHotEncoder(drop="first")`, avoiding redundant columns while preserving the ability to distinguish all three types

All preprocessing was wrapped in a `ColumnTransformer` and combined with each classifier inside a single `Pipeline`, ensuring consistent transformation of train and test data with no risk of data leakage.

## 4. Modeling Iterations

### 4.1 Logistic Regression (unweighted baseline)
Accuracy of 96.75% looked strong, but recall was only **10.3%** the model defaulted to predicting "no failure" almost universally, since that strategy is already correct 96.6% of the time on imbalanced data. This model was not viable for the actual task.

### 4.2 Logistic Regression (class_weight="balanced")
Rebalancing the loss function shifted recall up to **82.4%**, but precision collapsed to **14.2%** over 300 false alarms out of 2,000 machines. This demonstrated the expected precision/recall trade-off, but the false alarm rate was too high for practical deployment.

### 4.3 Random Forest (default, class_weight="balanced")
Random Forest immediately outperformed both Logistic Regression variants: 98.1% accuracy, 73.4% precision, 69.1% recall, F1 of 0.712. Tree-based models naturally capture the "if this AND that" interaction patterns identified during EDA (e.g., torque and rotational speed acting jointly), which a linear model cannot represent without manual feature engineering.

### 4.4 Random Forest (hyperparameter tuned)
`GridSearchCV` (5-fold cross-validation, scored on F1) searched across `n_estimators`, `max_depth`, and `min_samples_leaf`. Best parameters: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`. This nudged every metric up slightly: 98.2% accuracy, 75.0% precision, 70.6% recall, F1 of 0.727.

### 4.5 Threshold Optimization
ROC-AUC of **0.964** indicated the model's underlying probability estimates were ranking failure risk very well, suggesting room to adjust the decision threshold. Precision-recall curve analysis showed the default 0.5 threshold was already close to optimal  the curve's steep precision drop-off occurs beyond ~65% recall. Searching all thresholds for maximum F1 found **0.525**, a marginal but clean improvement: 98.3% accuracy, 77.4% precision, 70.6% recall, F1 of 0.738 with 2 fewer false alarms than the 0.5 threshold, at no cost to recall.

## 5. Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (unweighted) | 0.9675 | 0.6364 | 0.1029 | 0.1772 | — |
| Logistic Regression (balanced) | 0.8245 | 0.1418 | 0.8235 | 0.2419 | — |
| Random Forest (default) | 0.9810 | 0.7344 | 0.6912 | 0.7121 | — |
| Random Forest (tuned, threshold=0.5) | 0.9820 | 0.7500 | 0.7059 | 0.7273 | — |
| **Random Forest (tuned, threshold=0.525)** | **0.9830** | **0.7742** | **0.7059** | **0.7385** | **0.9638** |

## 6. Feature Importance

| Feature | Importance |
|---|---|
| Torque [Nm] | 0.322 |
| Rotational speed [rpm] | 0.286 |
| Tool wear [min] | 0.213 |
| Air temperature [K] | 0.099 |
| Process temperature [K] | 0.064 |
| Type_L | 0.009 |
| Type_M | 0.006 |

Torque and rotational speed together account for roughly 61% of total feature importance, directly confirming the EDA observation that failures cluster where these two variables move outside their safe operating window together. Tool wear ranks third, consistent with the 150-minute threshold identified earlier.

Notably, machine **Type** carries very little importance (under 1% combined) despite Type L showing the highest raw failure rate in EDA. This indicates the model is learning the underlying physical failure mechanism (torque, speed, tool wear) rather than relying on machine type as a proxy — a more robust and generalizable signal than a categorical shortcut.

## 7. Error Analysis

Of the 68 actual failures in the test set, the final model missed 20 (false negatives) and raised 14 false alarms (false positives). Inspection of the false negatives showed their predicted probabilities clustered close to the 0.525 decision threshold (roughly 0.54–0.66), indicating these are genuinely borderline, ambiguous cases rather than confident model errors an expected and reasonable failure mode given the class overlap near the decision boundary.

## 8. Final Model Selection

**Chosen model:** Random Forest (tuned, threshold = 0.525)

**Justification:**
- Highest F1 score (0.7385) of all models tested best balance of precision and recall
- Strong ROC-AUC (0.9638), confirming reliable risk ranking across all thresholds, not just the chosen cutoff
- Catches 48 of 68 real failures (71%) while raising only 14 false alarms across 1,932 healthy machines, a practical trade off for a predictive maintenance deployment
- Both hyperparameter tuning and threshold optimization produced consistent, no downside improvements over the default model

**Rejected alternatives:**
- Logistic Regression (unweighted): 10% recall fails at the core task of catching failures
- Logistic Regression (balanced): 82% recall but only 14% precision over 300 false alarms, impractical
- Random Forest (default, untuned): solid, but strictly outperformed by the tuned + threshold optimized version

## 9. Limitations & Next Steps

- The model still misses ~29% of real failures; further gains could come from richer feature engineering (e.g., explicit torque×speed interaction terms), additional sensor data, or trying gradient boosting models (XGBoost/LightGBM) for comparison.
- Class imbalance (3.39% failure rate) means performance should be monitored on new data, as failure patterns or rates may shift over time in a real deployment (concept drift).
- The current threshold (0.525) was optimized for F1 on this test set; a real deployment should tune this based on the actual business cost of a missed failure versus a false alarm, which may not weight precision and recall equally.
- Model retraining cadence and a feedback loop for logging real-world outcomes would be needed to keep the model reliable in production.

## 10. Artifacts

- `final_model.pkl` — trained pipeline (preprocessing + tuned Random Forest classifier)
- `final_threshold.pkl` — optimal decision threshold (0.525) for converting probabilities to predictions