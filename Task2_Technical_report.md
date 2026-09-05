TECHNICAL REPORT
Advanced Machine Learning and Model Optimization
AI4I 2020 Predictive Maintenance Dataset Task 2
1. Executive Summary
This report documents Week 2 (Task 2) of the predictive-maintenance project using the AI4I 2020 dataset. The objective was to improve the machine-failure classifier through feature engineering, investigation of class imbalance, feature selection, hyperparameter optimization, and model explainability. The baseline Random Forest achieved 98.10% accuracy, 73.44% precision, 69.12% recall, and a 71.21% F1 score for the failure class on the held-out test set. Feature engineering produced physically meaningful variables, followed by feature-importance-based selection of eight inputs. The improved Random Forest achieved 99.25% accuracy, 94.92% precision, 82.35% recall, and an 88.19% F1 score on the same 2,000-sample test set. A 5fold Grid Search evaluated 27 parameter combinations (135 fits). The best configuration was 100 trees, maximum depth 10, and minimum samples per leaf 1, with a cross-validation F1 score of 0.8217.
2. Dataset and Problem Definition
The dataset contains 10,000 observations and 14 original columns. The target, Machine failure, is a binary classification variable where 0 denotes no failure and 1 denotes failure. There were no duplicate rows and no missing values in the original columns. The target was strongly imbalanced: 9,661 observations (96.61%) were non-failures and 339 (3.39%) were failures. Consequently, accuracy was treated as a secondary metric, precision, recall, and F1 for the failure class were emphasized. The modeling variables initially consisted of Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], and Tool wear [min]. An 80/20 stratified train test split with random state = 42 produced 8,000 training observations and 2,000 test observations.
3. Feature Engineering
Feature engineering expanded the dataset from 14 to 19 columns. The engineered variables were designed to represent relationships between operating measurements and to provide the model with more informative physical signals.


4. Feature Engineering Rationale
Temperature Difference measures the separation between process and ambient temperature. Temperature Ratio provides a relative rather than absolute temperature relationship. The Mechanical Power Proxy combines torque and rotational speed to represent mechanical loading/power behavior. Torque Speed Ratio represents the interaction between torque and speed. Tool Wear Category converts continuous wear into interpretable degradation groups. These transformations were selected because predictive maintenance is fundamentally concerned with interactions among operating conditions, loading, thermal state, and accumulated wear.
Feature	Purpose 
Temperature Difference [K]	Difference between process and air temperature; captures the thermal operating relationship.
Mechanical Power Proxy	Combines torque and rotational speed to represent machine mechanical loading power behavior.
Temperature Ratio	Represents the relative relationship between process and air temperature.
Torque Speed Ratio	Captures the interaction between torque and rotational speed.
Tool Wear Category	Groups tool wear observations into interpretable degradation stages.
5. Handling Class Imbalance
The initial Logistic Regression model exposed the effect of class imbalance: its unbalanced version achieved 10.29% recall and an F1 score of 0.1772 for failures, despite 96.75% accuracy. Applying class weight = 'balanced' increased recall to 82.35%, but precision fell to 14.18% and F1 to 0.2419. This created a large number of false alarms. The Random Forest approach produced a substantially better precision recall balance. This justified evaluating the minority class metrics directly instead of optimizing accuracy alone.
6. Baseline Random Forest
The baseline Random Forest achieved:
Accuracy:98.10%
Precision:73.44%
Recall:69.12%
F1 Score: 71.21%
Confusion matrix: [[1915, 17], [21, 47]]
The baseline correctly detected 47 of the 68 failure cases, missed 21 failures, and produced 17 false positives.
7. Feature Importance and Selection
Random Forest feature importance showed that the strongest predictive signals came mainly from mechanical operating variables and engineered interactions. The ranked importance values were: Mechanical Power Proxy (0.205157), Rotational speed (0.156218), Torque (0.141474), Torque Speed Ratio (0.129246), Tool wear (0.119456), Temperature Ratio (0.080667), Temperature Difference (0.070416), Air temperature (0.034395), and Process temperature (0.031393). The encoded Type and Tool Wear Category variables had substantially smaller importance values. The feature selection stage reduced the transformed model input to eight selected features. The selected matrix had shape (10,000, 8), with corresponding training and test shapes of (8,000, 8) and (2,000, 8). This concentrated the model on the most influential predictors while reducing the number of inputs.
8. Hyperparameter Optimization
Grid Search with 5-fold cross validation was performed on the Random Forest using F1 as the scoring metric. The search evaluated 27 parameter combinations, resulting in 135 fits. F1 was selected as the optimization objective because the failure class is rare and the project requires a balance between detecting failures and limiting false alarms.
9. Final Model Evaluation
The feature selected Random Forest produced the following held-out test performance:
Accuracy:99.25%
Precision:94.92%
Recall:82.35%
F1 Score: 88.19%
Confusion matrix: [[1929, 3], [12, 56]]
The model correctly classified 1,929 of 1,932 non-failure cases and 56 of 68 failure cases. It generated only three false positives and twelve false negatives.



10. Before and After Performance
The improvement is substantial. Compared with the baseline Random Forest, accuracy increased by 1.15 percentage points, precision increased by 21.48 percentage points, recall increased by 13.24 percentage points, and failure-class F1 increased by 16.98 percentage points. The largest practical improvement is the reduction in false positives from 17 to 3 while simultaneously reducing missed failures from 21 to 12. This indicates a much better balance between sensitivity and false-alarm control on the held-out test set.
Metric	Baseline RF	Feature-selected RF
Accuracy	98.10%	99.25%
Precision (failure)	73.44%	94.92%
Recall (failure)	69.12%	82.35%
F1 (failure)	71.21%	88.19%
False positives	17	3
False negatives	21	12
11. Model Explainability
Feature importance was used to explain which variables contributed most to the Random Forest's predictions. Mechanical Power Proxy was the dominant feature, followed by rotational speed, torque, torque-speed ratio, and tool wear. The results suggest that machine loading, speed, their interaction, and accumulated tool wear are the most influential signals in the learned decision process. Temperature-derived features also contributed, but less strongly than the principal mechanical variables. Machine Type contributed relatively little compared with the direct operating measurements.
Feature importance indicates association within the trained model; it does not establish that a feature independently causes failure. Correlated variables can also share or redistribute importance.
12. Generalization and Trade offs
The held out test set was kept separate from training and hyperparameter selection, providing an independent evaluation of the final workflow. The best cross-validation F1 of 0.8217 and test F1 of 0.8819 are not expected to be identical because they measure performance on different samples.
The final model is more complex than a simple linear classifier, but the improvement in failure detection and false-alarm control justifies the added complexity for this project. The main remaining concern is the limited number of failure observations, so additional validation on genuinely unseen operational data would strengthen confidence before deployment.
13. Conclusion
Week 2 achieved the objective of improving the predictive-maintenance solution beyond the baseline model. The workflow introduced physically meaningful features, explicitly investigated class imbalance, selected the strongest model inputs, optimized Random Forest hyperparameters using cross-validation, and interpreted the model using feature importance. The final test result of 99.25% accuracy and 88.19% failure-class F1 represents a clear improvement over the baseline Random Forest's 98.10% accuracy and 71.21% F1. The final confusion matrix also shows fewer false positives and fewer missed failures. The project is now ready to move from model development into final validation, model packaging, and preparation of the deployment-oriented documentation.
