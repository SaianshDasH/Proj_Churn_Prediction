# 🛒 E-Commerce Customer Churn Prediction & Revenue Impact Analysis

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189FDD)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

> An e-commerce company is losing **~25% of its customers every quarter**. The business team needs a system that predicts which customers are likely to churn so they can launch **targeted retention campaigns** and reduce revenue leakage.

This project tackles this real-world problem end-to-end — from **SQL-based data engineering** to **deep learning** — and quantifies the **business impact** in monetary terms.

---

## 🧠 Project Highlights

| Aspect | Details |
|--------|---------|
| **Dataset** | ~5,600 customers, 20 behavioral features |
| **SQL** | Normalized schema, RFM segmentation, cohort analysis, window functions |
| **Python** | EDA with 12+ visualizations, robust preprocessing pipeline |
| **Machine Learning** | Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, LightGBM |
| **Deep Learning** | ANN (Keras), Entity Embeddings, Autoencoder-based anomaly detection |
| **Interpretability** | SHAP values, feature importance, permutation importance |
| **Business** | CLV-based ROI analysis, cost-benefit matrix, retention campaign simulation |

---

## 📁 Project Structure

```
ecommerce-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                    # Original dataset (gitignored)
│   └── processed/              # Cleaned & transformed data
│
├── notebooks/
│   ├── Day01_SQL_Database_Setup.py
│   ├── Day02_SQL_Advanced_Analysis.py
│   ├── Day03_EDA_Python.py
│   ├── Day04_Feature_Engineering.py
│   ├── Day05_ML_Baseline_Models.py
│   ├── Day06_ML_Advanced_Tuning.py
│   ├── Day07_Deep_Learning_ANN.py
│   ├── Day08_DL_Advanced_Architectures.py
│   ├── Day09_Model_Comparison_Business.py
│   └── Day10_Final_Pipeline.py
│
├── sql/
│   ├── 01_create_tables.sql
│   ├── 02_data_profiling.sql
│   ├── 03_rfm_analysis.sql
│   ├── 04_cohort_analysis.sql
│   └── 05_business_kpis.sql
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── deep_learning_model.py
│
├── models/                     # Saved model artifacts
│
└── reports/
    ├── eda_summary.md
    ├── model_comparison.md
    └── business_recommendations.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ecommerce-churn-prediction.git
cd ecommerce-churn-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place the CSV file in data/raw/ directory
# Dataset: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

# 5. Run notebooks day by day
python notebooks/Day01_SQL_Database_Setup.py
```

---

## 📊 Key Findings

> ⚠️ *Results will be populated as the project progresses*

### Churn Drivers (Top Features)
1. **Complain** — Customers who complained are X× more likely to churn
2. **Tenure** — New customers (< 6 months) have the highest churn rate
3. **DaySinceLastOrder** — Longer gaps = higher churn probability
4. **CashbackAmount** — Lower cashback correlates with higher churn
5. **SatisfactionScore** — Inverse relationship with churn

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| LightGBM | — | — | — | — | — |
| ANN (Keras) | — | — | — | — | — |
| Wide & Deep | — | — | — | — | — |

### Business Impact
- **Revenue at Risk**: ₹ — lakhs/quarter from potential churners
- **Model Savings**: Catching X% of churners → ₹ — lakhs saved
- **Recommended Action**: Top 3 retention strategies based on analysis

---

## 📅 10-Day Development Log

| Day | Focus | Status |
|-----|-------|--------|
| 1 | SQL: Database Design & Data Ingestion | ⬜ |
| 2 | SQL: Advanced Analysis (RFM, Cohorts, KPIs) | ⬜ |
| 3 | Python: Exploratory Data Analysis | ⬜ |
| 4 | Python: Feature Engineering & Preprocessing | ⬜ |
| 5 | ML: Baseline Models (LR, DT, RF, SVM) | ⬜ |
| 6 | ML: Advanced Models & Hyperparameter Tuning | ⬜ |
| 7 | DL: Artificial Neural Network (ANN) | ⬜ |
| 8 | DL: Advanced Architectures & Experiments | ⬜ |
| 9 | Model Comparison & Business Impact | ⬜ |
| 10 | Final Pipeline & Documentation | ⬜ |

---

## 🛠️ Tech Stack

- **Languages**: Python 3.10+, SQL
- **Data**: Pandas, NumPy, SQLite
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: TensorFlow / Keras
- **Interpretability**: SHAP
- **Imbalanced Learning**: imbalanced-learn (SMOTE)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Connect

If you found this project helpful, give it a ⭐ and feel free to connect!
