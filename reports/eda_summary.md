# EDA Summary Report -- Day 3

## Dataset Overview
- **Total customers**: 5630
- **Churn rate**: 16.5% (928 churned, 4702 retained)
- **Features**: 20 columns (15 numeric, 5 categorical)
- **Missing values**: 7 columns have nulls

## Top Churn Drivers (Ranked by Impact)

### 1. Complaint Status (STRONGEST)
- Complainers: **28.8% churn** vs Non-complainers: **10.1% churn**
- Complainers are **2.85x more likely** to leave
- Action: Prioritize complaint resolution to reduce churn

### 2. Satisfaction Score (STRONG)
- Score 1: **22.9% churn** -> Score 5: **9.1% churn**
- Each 1-point increase reduces churn by ~3-5%
- Action: Improve customer experience for low-score customers

### 3. Tenure (STRONG)
- New customers (0-6 months): highest churn
- Veteran customers (24+ months): near-zero churn
- Action: Focus retention on first-year customers

### 4. Product Category (MODERATE)
- Laptop & Accessory: highest churn rate
- Fashion: lowest churn rate
- Action: Investigate delivery/quality issues in electronics

### 5. City Tier (WEAK)
- Small cities slightly higher churn than metros
- Difference is marginal (~1-2%)
- Action: Not a primary lever

## Class Imbalance
- 83.5% retained vs 16.5% churned
- Must use SMOTE or class_weight='balanced' during modeling

## Missing Data Strategy (For Day 4)
- Tenure: median imputation (60.8% missing)
- Order features: median imputation + missing indicator flag
- Hours on app: median imputation (94.1% missing -- very sparse)

## Charts Generated (12 total)
All saved in `reports/figures/` directory.
