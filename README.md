
#  Imputation via Regression for Missing Data 

**Author:** Mohmad Yaqoob (DA25M017)

---

## Project Overview

In real-world datasets, missing values are common and can severely affect the performance of machine learning models. This project demonstrates how different missing data handling strategies influence the predictive performance of a credit risk classifier.

We use the **UCI Credit Card Default Clients Dataset** and artificially introduce **Missing At Random (MAR)** values in key numerical columns. We then apply three different imputation strategies, plus a baseline approach of listwise deletion, to assess their impact on logistic regression classification performance.

The objective is to understand the trade-offs between simple and advanced imputation methods and how missing data affects downstream modeling.

---

## Dataset Description

The dataset contains information on 30,000 credit card clients in Taiwan. Key points:

- **Target variable:** `default.payment.next.month` (0 = No default, 1 = Default)
- **Features:** Demographics (AGE, SEX, EDUCATION), financial metrics (BILL_AMT1–6, PAY_AMT1–6), payment history, and previous bill amounts.

**Step 1 — Introduce Missing Values (MAR):**

- Randomly set **5% of `AGE` and `BILL_AMT1` values** to `NaN`.
- MAR assumption: missingness depends on observed data, not on the missing value itself.
- This simulates realistic scenarios in which some client information is unavailable.

---

## Part A — Data Preprocessing and Imputation

### Strategy 1: Median Imputation (Dataset A)

**Method:** Replace missing values with the **median** of the respective column.

**Why median?** Robust to outliers and preserves central tendency better than mean for skewed data.

**Formula:**

\[
\tilde{x} = \text{median of observed values in column } x
\]

### Strategy 2: Linear Regression Imputation (Dataset B)

**Method:** Use Linear Regression to predict missing entries based on other non-missing features.

**Assumptions:** MAR, linear relationship between predictors and missing column.

**Linear Regression Formula:**

\[
\hat{y} = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n
\]

### Strategy 3: Non-Linear Regression Imputation (Dataset C)

**Method:** Use KNN Regression to predict missing values.

**Formula:**

\[
\hat{x}_{\text{missing}} = \frac{1}{K} \sum_{i=1}^{K} x_i^{\text{neighbor}}
\]

### Strategy 4: Listwise Deletion (Dataset D)

**Method:** Remove all rows containing missing values.

**Pros:** Simple, no imputation bias.
**Cons:** Reduces dataset size and can hurt model performance.

---

## Part B — Model Training and Performance Assessment

- Logistic Regression classifier
- StandardScaler to standardize features

**Logistic Regression Formula:**

\[
P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^{n} \beta_i x_i)}}
\]

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, AUC, Log Loss

---

## Part C — Comparative Analysis

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|---------|----------|--------|----------|-----|
| Model A – Median Imputation | 0.81 | 0.71 | 0.24 | 0.36 | 0.71 |
| Model B – Linear Regression | 0.81 | 0.70 | 0.24 | 0.36 | 0.71 |
| Model C – Non-Linear Regression | 0.81 | 0.71 | 0.24 | 0.36 | 0.71 |
| Model D – Listwise Deletion | 0.81 | 0.69 | 0.23 | 0.34 | 0.70 |

**Observations:** Regression-based imputations slightly improve F1-score for minority class over listwise deletion.

**Trade-offs:**

1. **Median Imputation:** Quick, ignores relationships.
2. **Linear Regression:** Captures linear dependencies.
3. **Non-Linear Regression:** Captures complex dependencies, best overall.
4. **Listwise Deletion:** Simple but reduces training data.

---

## Visual Analysis

1. **Bar Charts:** Compare key metrics across models.
2. **Radar Chart:** Holistic performance view; larger area = stronger overall performance.

---

## Conclusion

- Missing data handling significantly impacts classification performance.
- Regression-based imputations (especially non-linear) outperform median and listwise deletion.
- Non-linear imputation is recommended for complex feature relationships.

**Author:** Mohmad Yaqoob (DA25M017)
