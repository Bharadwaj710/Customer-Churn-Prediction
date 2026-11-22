# Exploratory Data Analysis (EDA) Academic Findings Report

**Project:** Customer Churn Prediction Using Machine Learning  
**Dataset:** Authentic IBM Telco Customer Churn Dataset (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Records Analyzed:** 7,043 rows, 21 attributes  
**Analysis Date:** 2026-11-22  

---

## 1. Executive Summary & Class Distribution

- **Total Sample Size:** 7,043 customers.
- **Retained Customers (`Churn = No`):** 5,174 (73.46%).
- **Churned Customers (`Churn = Yes`):** 1,869 (26.54%).
- **Class Imbalance Ratio:** ~2.77 : 1 (Majority to Minority).
- **Academic Significance:** Because the minority class represents 26.5% of instances, a naive baseline classifier predicting "No" would achieve 73.5% accuracy while failing completely to identify at-risk customers. Precision, Recall, F1-score, and ROC-AUC are mandatory metrics.

---

## 2. Key Empirical Findings by Domain Segment

### A. Contract Structure & Commitment
- **Month-to-month Contract Churn Rate:** **42.71%** (1,655 out of 3,875 customers).
- **One-Year Contract Churn Rate:** **11.27%** (166 out of 1,473 customers).
- **Two-Year Contract Churn Rate:** **2.83%** (48 out of 1,695 customers).
- **Finding:** Customers on month-to-month contracts are **15.1x more likely to churn** than customers on two-year contracts. Long-term contractual commitment serves as the single strongest protective factor against customer attrition.

### B. Customer Tenure & Cohort Dynamics
- **0–12 Months Tenure Cohort Churn Rate:** **47.44%**.
- **61–72 Months Tenure Cohort Churn Rate:** **6.61%**.
- **Median Tenure Comparison:**
  - Retained customers: **38 months** (Mean: 37.6 months).
  - Churned customers: **10 months** (Mean: 18.0 months).
- **Finding:** Churn risk is front-loaded. Nearly half of all first-year subscribers churn, after which churn rate systematically declines as brand lock-in and customer habituation increase.

### C. Financial & Billing Factors
- **Payment Method Churn Rates:**
  - **Electronic Check:** **45.29%** (1,071 churned).
  - **Mailed Check:** **19.11%**.
  - **Bank Transfer (Automatic):** **16.71%**.
  - **Credit Card (Automatic):** **15.24%**.
- **Monthly Charges Impact:**
  - Churned Customers Median: **$79.65** (Mean: $74.44).
  - Retained Customers Median: **$64.43** (Mean: $61.27).
  - Difference: Churned customers pay **$15.22 more per month** at the median.
- **Total Charges Paradox:**
  - Retained Customers Median Total Charges: **$1,683.60**.
  - Churned Customers Median Total Charges: **$703.55**.
  - Explanation: Because churned customers leave early in their lifecycle (median tenure of only 10 months), their cumulative lifetime spend is lower despite paying higher monthly rates.

### D. Service Configuration & Add-on Utilities
- **Internet Service Type:**
  - **Fiber Optic:** **41.89%** churn rate.
  - **DSL:** **18.96%** churn rate.
  - **No Internet:** **7.40%** churn rate.
  - Fiber optic users churn at more than double the rate of DSL users, primarily because fiber plans carry significantly higher monthly fees ($70–$100+/mo).
- **Protective Add-ons:**
  - Customers without Tech Support churn at **41.64%** vs **15.17%** with Tech Support.
  - Customers without Online Security churn at **41.77%** vs **14.61%** with Online Security.
  - Subscribers with value-added support and security services exhibit strong retention.

### E. Demographics
- **Senior Citizens:** Churn rate of **41.68%** vs **23.61%** for non-seniors.
- **Gender:** Female churn rate is **26.92%** vs Male churn rate of **26.16%**. Gender is virtually non-predictive (difference < 1%).

---

## 3. Numerical Correlation Analysis & Academic Limitations

### Pearson / Point-Biserial Correlation with `Churn_Binary`:
- `tenure` vs `Churn_Binary`: **$r = -0.352$** (Moderate negative correlation).
- `MonthlyCharges` vs `Churn_Binary`: **$r = +0.193$** (Moderate positive correlation).
- `TotalCharges` vs `Churn_Binary`: **$r = -0.199$** (Weak negative correlation).

### Crucial Academic Methodological Note:
> [!WARNING]
> **Limitations of Interpreting Correlations with a Binary Target:**
> 1. **Point-Biserial Nature:** Pearson's correlation coefficient assumes continuous bivariate normality. When calculated against a dichotomous binary target ($Y \in \{0, 1\}$), Pearson's formula simplifies to the point-biserial correlation ($r_{pb}$), whose theoretical maximum is strictly bounded below $1.0$ (depending on class distribution skewness).
> 2. **Non-Linear Relationships:** Linear correlation fails to capture threshold or non-linear effects (e.g., rapid churn spike in months 1–3 followed by plateau).
> 3. **Confounding Variables:** `TotalCharges` is the mathematical product of `tenure` $\times$ `MonthlyCharges`. Its negative correlation with churn ($r = -0.199$) is confounded by tenure and should not be misinterpreted as lower total cost causing churn.
> 4. **Identifier Exclusion:** `customerID` was strictly excluded from correlation analysis as it is an arbitrary primary key with no predictive or statistical meaning.

---

## 4. Summary Table of Generated Figures

| File Name | Analysis Type | Description |
| :--- | :--- | :--- |
| `01_univariate_churn.png` | Univariate | Class balance count and percentage breakdown |
| `02_univariate_demographics.png` | Univariate | Gender and Senior Citizen distributions |
| `03_univariate_contract_payment.png` | Univariate | Contract types and Payment method counts |
| `04_univariate_internet_service.png` | Univariate | DSL, Fiber optic, and No Internet proportions |
| `05_univariate_numerical_distributions.png` | Univariate | Histograms with KDE for Tenure, MonthlyCharges, TotalCharges |
| `06_bivariate_contract_churn.png` | Bivariate | Counts and empirical churn rates across contract types |
| `07_bivariate_tenure_churn.png` | Bivariate | Boxplots and churn percentages across 6 tenure cohorts |
| `08_bivariate_charges_churn.png` | Bivariate | MonthlyCharges density and TotalCharges distributions |
| `09_bivariate_services_churn.png` | Bivariate | Churn rates across Internet, TechSupport, OnlineSecurity |
| `10_bivariate_payment_churn.png` | Bivariate | Churn rates across all 4 payment methods |
| `11_bivariate_demographics_churn.png` | Bivariate | Churn rates across Senior Citizen status and Gender |
| `12_correlation_heatmap.png` | Multivariate | Correlation matrix across numerical variables and target |

---
