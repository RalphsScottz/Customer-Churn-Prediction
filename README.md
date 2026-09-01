# E-Commerce Customer Churn Analysis

A beginner-friendly, end-to-end data science project that analyzes an e-commerce customer dataset to answer four real business questions — from raw data cleaning through exploratory analysis to machine learning modeling and business recommendations.

Built as a single, self-contained Jupyter notebook: upload the dataset, run all cells, get answers.

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)
- [License](#license)

## Business Problem

An e-commerce platform wants to understand and reduce customer churn. The analysis answers four specific business questions:

1. **Churn risk** — Which active customers are at the highest risk of abandoning the platform in the next 30 days, and what behaviours indicate they are pulling away?
2. **Discount dependency** — Which customer segments buy mainly because of cashback and coupons, and which customers stay loyal without continuous discounting?
3. **Complaint prediction** — Can we predict whether a customer is about to file a complaint based on recent app activity and ordering friction, *before* the complaint happens?
4. **Delivery friction** — At what delivery-distance threshold does friction start driving up complaints and churn, and does this vary by region?

## Dataset

- **5,630 customers**, 20 original columns (tabular, one row per customer)
- Includes: tenure, city tier, delivery distance, login device, payment mode, app usage hours, order category, satisfaction score, marital status, number of addresses, complaint flag, order amount change vs. last year, coupons used, order count, days since last order, cashback amount, and the churn label
- Two binary targets: `Churn` (~17% positive) and `Complain` (~28% positive) — both meaningfully imbalanced

> The raw file is not included in this repository. Provide your own `.xlsx` export with the same column schema (see notebook Step 1 for the exact expected columns).

## Project Structure

```
.
├── ecommerce_churn_analysis_colab.ipynb   # Main notebook — run this
└── README.md
```

Everything — cleaning, EDA, modeling, evaluation, and the written business answers — lives in the one notebook, organized into clearly labeled sections with markdown explanations before every code cell.

## Methodology

The project follows a standard **CRISP-DM** style workflow:

### 1. Data Understanding
Inspect data types, missing-value patterns, class balance for both targets, inconsistent category labels (e.g. `"COD"` vs `"Cash on Delivery"`), and outlier detection (e.g. two implausible 126–127 km delivery-distance values).

### 2. Data Cleaning & Preprocessing
- Merge inconsistent category spellings across `PreferredPaymentMode`, `PreferedOrderCat`, `PreferredLoginDevice`
- Treat extreme delivery-distance outliers as missing, then **median-impute** all numeric columns with missing values (median chosen for robustness to skew/outliers)
- Feature engineering: `CouponUsageRate` (coupons used per order), `CashbackPerOrder`, `IsNewCustomer` (tenure ≤ 3 months)
- One-hot encode categorical variables for modeling

### 3. Exploratory Data Analysis
Churn/complaint rates broken down by tenure, satisfaction score, complaint history, order category, and city tier; a full numeric correlation heatmap.

### 4. Modeling — one approach per business question

| Question | Technique | Target | Notes |
|---|---|---|---|
| Q1 — Churn risk | Logistic Regression (baseline) + **Random Forest** (final model) | `Churn` | Class-weighted to handle imbalance; scored on a held-out 20% test set; active customers ranked by predicted churn probability |
| Q2 — Discount dependency | **K-Means clustering** (unsupervised) | n/a | 4 segments profiled on coupon usage rate, cashback per order, spend growth, tenure, satisfaction, order count; k chosen via silhouette score comparison |
| Q3 — Complaint prediction | **Random Forest** classifier | `Complain` | `Churn` deliberately excluded from features to avoid leaking a downstream outcome into a "before it happens" prediction |
| Q4 — Delivery friction threshold | Binned rate analysis + **decision tree stump** (max depth 2) | `Churn`, `Complain` | City tier used as the closest available proxy for "region"; tree stump lets the data pick the break point rather than manual binning alone |

### 5. Evaluation
Classification models are judged with **precision, recall, F1-score, ROC-AUC, and a confusion matrix** — not raw accuracy, since both targets are imbalanced and accuracy alone would be misleading. Clustering is judged with **silhouette score** plus business-interpretability of the resulting segment profiles.

### 6. Business Translation
Each of the four sections ends with a plain-language answer to the original business question and a concrete recommended action, followed by a project-level summary table and an honest limitations section.

## Results Summary

| # | Question | Key Finding | Model Performance |
|---|---|---|---|
| 1 | Churn risk | New customers (0–3 month tenure) with low cashback, a recent complaint, and no recent orders are highest risk | Random Forest, ROC-AUC ≈ 0.98 |
| 2 | Discount dependency | Short-tenure "deal chasers" churn most; long-tenure heavy-discount users are actually the most loyal segment — tenure, not discount usage, is what separates healthy from risky discount reliance | 4-segment K-Means |
| 3 | Complaint prediction | Complaints can be anticipated from pre-complaint behaviour alone (low cashback, short tenure, long delivery distance, recent spend spikes) | Random Forest, ROC-AUC ≈ 0.82 |
| 4 | Delivery friction | Risk rises noticeably past ~13–14 km and again past ~27 km; the effect is strongest in City Tier 3 | Binned rates + decision tree split |

## How to Run

**Google Colab (recommended):**
1. Open `ecommerce_churn_analysis_colab.ipynb` in Colab
2. `Runtime → Run all`
3. When prompted, upload your e-commerce `.xlsx` dataset
4. That's the only manual step — cleaning, charts, models, and results run automatically, entirely in memory (no files written to disk unless you explicitly opt into the one optional CSV-export cell)

**Local Jupyter:**
1. Place your `.xlsx` file in the same folder as the notebook
2. Install dependencies (see below)
3. Run all cells — the notebook auto-detects it isn't running in Colab and loads the local file instead

## Tech Stack

- Python 3
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualization
- scikit-learn — Logistic Regression, Random Forest, K-Means, Decision Tree, preprocessing, and evaluation metrics

```
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Limitations

- Single-snapshot data (one row per customer, not full event-level history) — churn "risk" is a current score, not a literal date forecast
- Metrics come from one train/test split; a production system should use cross-validation and periodic retraining
- Clustering silhouette scores are modest (~0.2–0.25), typical for real behavioral data — segments are directionally useful, not perfectly separated
- City Tier is used as a proxy for "region" since the dataset has no explicit geography column

## License

Add a license of your choice (e.g. MIT) if you plan to make this repository public.
