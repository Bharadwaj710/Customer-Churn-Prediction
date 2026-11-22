# Technical Project Walkthrough & Academic Defense Guide

## Customer Churn Prediction – Capstone Technical Viva / Demonstration Guide

This document is structured as an interactive technical demonstration and oral defense guide. It addresses the 12 fundamental questions evaluated during academic project reviews, viva voce examinations, and senior technical interviews.

---

### 1. What problem does this project solve?
In subscription telecom businesses, acquiring new customers costs $5\times$ to $7\times$ more than retaining existing subscribers. This project solves the problem of **subscriber attrition** by transforming raw customer operational and billing records into:
1. Early warning churn risk probabilities.
2. Segment-level behavioral insights explaining why customers leave.
3. Prescriptive, cost-effective retention playbooks tailored to individual subscriber profiles.

---

### 2. Why is churn prediction important?
Unplanned customer churn directly degrades Monthly Recurring Revenue (MRR) and customer lifetime value (CLV). A telecom operator experiencing a $26.5\%$ annual churn rate loses over a quarter of its subscriber base each year. Predicting churn allows proactive, automated intervention *before* the customer reaches out to cancel or switches to a competitor.

---

### 3. Why was the authentic IBM Telco dataset selected?
The IBM Telco Customer Churn dataset is the industry benchmark for subscription modeling because it captures the complete multi-dimensional customer lifecycle:
- **Demographics:** Age, gender, family structure.
- **Contractual terms:** Month-to-month, 1-year, 2-year commitments.
- **Service infrastructure:** DSL vs. high-speed Fiber optic.
- **Value-added bundles:** Online security, tech support, streaming media.
- **Billing friction:** Paperless billing, automated bank debit vs. electronic checks.
- **Financial metrics:** Monthly charges and cumulative lifetime total charges.

Its authentic SHA-256 integrity was verified (`16320c9c...`), accounting for all 7,043 customer accounts.

---

### 4. What preprocessing was performed, and how was data leakage eliminated?
1. **Domain Imputation:** 11 records in `TotalCharges` contained whitespace strings. An audit confirmed that all 11 records had `tenure == 0` (new accounts prior to their first monthly billing cycle). Rather than discarding data or imputing a misleading mean, they were deterministically imputed with `0.0`.
2. **Key Exclusion:** `customerID` was removed to prevent memorization of arbitrary keys.
3. **Target Encoding:** `Churn` was mapped to binary integers (`No` $\rightarrow 0$, `Yes` $\rightarrow 1$).
4. **Stratified Split:** Data was partitioned into an 80:20 stratified split (`random_state=42`), preserving class balance (73.46% / 26.54%) across training ($N=5,634$) and held-out test sets ($N=1,409$).
5. **Zero Data Leakage:** A Scikit-learn `ColumnTransformer` (`StandardScaler` for continuous features, `OneHotEncoder(drop='first')` for categoricals) was fitted **exclusively on the training data**. The fitted preprocessor was serialized to disk (`preprocessor.joblib`) and applied to test and live inference data without ever recalculating scaling or encoding statistics.

---

### 5. Why were three distinct model paradigms selected?
Comparing diverse architectural families is required to ensure academic rigor:
1. **Logistic Regression:** A linear parametric model providing auditable log-odds coefficients ($\beta_i$) and odds ratios.
2. **Random Forest Classifier:** A non-linear decision tree ensemble utilizing bootstrap aggregation (bagging) and feature subspace sampling to model complex feature interactions.
3. **Deep Learning MLP (TensorFlow/Keras):** A multi-layer artificial neural network (64-relu / 32-relu / 1-sigmoid) with dropout regularization to evaluate hierarchical representation learning.

---

### 6. Why was Random Forest selected as the recommended production model?
On the unseen held-out test cohort ($N = 1,409$):
- **Highest Accuracy:** $75.73\%$ ($+1.92\%$ over Logistic Regression).
- **Lowest False Positive Rate:** Generated only 263 false alarms compared to 288 for Logistic Regression, saving substantial operational outreach costs.
- **Highest Average Precision (AP):** Attained $0.6590$ (a $+0.0259$ advantage under class imbalance).
- **Operational Balance:** Achieved $78.88\%$ recall (identifying 295 out of 374 actual churners) while maintaining $74.59\%$ specificity.
- **Inherent Regularization:** Handled tabular interaction thresholds naturally without requiring extensive feature engineering.

---

### 7. Why are Accuracy, Precision, Recall, F1, ROC-AUC, and Average Precision all necessary?
- **Accuracy ($75.73\%$):** Measures overall correctness but is insufficient alone because a naive model predicting "No Churn" achieves $73.46\%$ accuracy while catching zero churners.
- **Recall ($78.88\%$):** Measures sensitivity—the percentage of true churners caught. Essential to avoid lost revenue.
- **Precision ($52.87\%$):** Measures positive predictive value—when the model sounds an alarm, how often is it right? Controls marketing budget waste.
- **F1-Score ($0.6330$):** Harmonic mean of precision and recall.
- **ROC-AUC ($0.8434$):** Evaluates ranking concordance across all thresholds.
- **Average Precision ($0.6590$):** Focuses specifically on the minority churn class across thresholds. Critical for evaluating imbalanced datasets where true negatives outnumber positive churn cases $3:1$.

---

### 8. How does the real-time Prediction Simulator work?
1. The stakeholder inputs raw customer parameters through the Plotly Dash interface.
2. `src/dashboard/model_inference.py` formats the inputs into a 1-row DataFrame.
3. The loaded `preprocessor.joblib` transforms the inputs into the exact 30-feature vector expected by the models.
4. The chosen model evaluates the vector and outputs raw posterior probability $P(\text{Churn} = 1)$.
5. The gauge meter updates dynamically, assigning the account to a Risk Tier (**Low**, **Moderate**, or **High**).
6. Model-specific feature influences (log-odds for LR, feature importance for RF) are displayed alongside the result.

---

### 9. What does the Classification Decision Threshold ($\tau$) control?
The default threshold is $\tau = 0.50$. Modifying $\tau$ enables operational flexibility:
- **Low Threshold ($\tau = 0.35$):** Recall increases to $96.6\%$, capturing almost every at-risk customer at the cost of more false positives. Ideal for low-cost, automated email campaigns.
- **High Threshold ($\tau = 0.65$):** Precision increases to $68.5\%$, minimizing false alarms. Ideal when retention incentives (e.g., $\$100$ bill credits or subsidized hardware) carry significant direct costs.

---

### 10. How does the Retention Playbook generate recommendations?
`src/dashboard/recommendations.py` applies domain-justified heuristic rules linked to empirical EDA findings:
- **Month-to-Month Contract:** Triggers an annual contract transition incentive with a rate lock.
- **Tenure $\le 12$ Months:** Triggers a first-year onboarding check-in to mitigate early customer drop-off.
- **Fiber Optic without Tech Support / Security:** Triggers a complimentary 6-month trial of technical support and security add-ons to eliminate service friction.
- **Electronic Check Payment:** Triggers an autopay bill credit incentive to eliminate manual billing friction.

*Methodological Distinction:* Recommendations are clearly labeled as heuristic interventions based on historical statistical associations, not definitive causal claims.

---

### 11. What are the limitations of this project?
1. **Static Snapshot Data:** The dataset is cross-sectional; longitudinal time-series logs (e.g., daily network outages or call center transcripts) were unavailable.
2. **Observational Inference:** Statistical associations (e.g., month-to-month contracts correlating with churn) do not guarantee that offering a 1-year contract will unilaterally prevent churn. Real-world validation requires randomized A/B trials.
3. **Tabular Neural Network Limits:** The TensorFlow MLP matched but did not exceed the Random Forest ensemble, reflecting known properties of deep learning on moderate tabular datasets.

---

### 12. What would be improved in a production deployment?
1. **Real-Time Data Streaming:** Connect Apache Kafka to stream daily customer usage metrics.
2. **Automated ML Pipelines (CI/CD/CT):** Deploy continuous training triggers via MLflow or Kubeflow to retrain models when data drift is detected.
3. **Causal Inference / Uplift Modeling:** Implement Two-Model Uplift or CausalML to identify "persuadable" customers rather than customers who would churn regardless of intervention.
4. **Explainable AI:** Integrate SHAP (SHapley Additive exPlanations) or TreeSHAP for game-theoretic local attribution.

