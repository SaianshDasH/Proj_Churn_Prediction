"""
============================================================
Day 04: Feature Engineering & Preprocessing
E-Commerce Customer Churn Prediction
============================================================

WHAT THIS SCRIPT DOES (in plain English):
    1. Loads the clean data from Day 1
    2. Fills in missing values (so the model doesn't crash)
    3. Creates brand-new features from existing columns
    4. Converts text categories into numbers (models need numbers)
    5. Scales everything to the same range
    6. Splits data into training set and test set
    7. Balances the churned vs retained classes (SMOTE)
    8. Saves everything for Day 5 modeling

WHY THIS MATTERS:
    Raw data is messy. Models need clean, numeric, balanced data.
    This step is 70% of a data scientist's real job.
============================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import joblib
warnings.filterwarnings('ignore')

# scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section_header, ensure_directories


# ============================================================
# STEP 1: Load the data
# ============================================================
print_section_header("STEP 1: Load Data from Day 1")
ensure_directories()

df = pd.read_csv("data/processed/churn_data_clean.csv")
print(f"[OK] Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"     Churn rate: {df['churn'].mean()*100:.1f}%")
print(f"     Missing values in {df.isnull().any().sum()} columns")


# ============================================================
# STEP 2: Handle Missing Values
# ============================================================
# WHY: Machine learning models cannot handle NaN/null values.
#      We need to fill them with reasonable substitutes.
#
# STRATEGY:
#   - For numeric columns with < 70% missing: fill with MEDIAN
#     (median is safer than mean because outliers don't skew it)
#   - For columns with > 90% missing: DROP the column entirely
#     (too much missing = unreliable signal)
#   - Create "was_missing" flag columns for important features
#     (the FACT that data is missing can itself be a signal)
# ============================================================
print_section_header("STEP 2: Handle Missing Values")

# Show what we're working with
print("\nMissing value summary BEFORE fixing:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    pct = count / len(df) * 100
    print(f"  {col:40s} -> {count:5d} missing ({pct:.1f}%)")

# --- Step 2a: Drop columns that are too sparse (> 90% missing) ---
# These columns have so little real data that imputing them would
# basically be making up numbers. Better to drop them.

DROP_THRESHOLD = 0.90
cols_to_drop = []

for col in df.columns:
    if df[col].isnull().mean() > DROP_THRESHOLD:
        cols_to_drop.append(col)

if cols_to_drop:
    print(f"\n[!] Dropping {len(cols_to_drop)} columns (>{DROP_THRESHOLD*100:.0f}% missing):")
    for col in cols_to_drop:
        print(f"    - {col} ({df[col].isnull().mean()*100:.1f}% missing)")
    df = df.drop(columns=cols_to_drop)

# --- Step 2b: Create "was_missing" flags BEFORE imputing ---
# WHY: If a customer's tenure is unknown, that itself tells us something
#      (maybe they're a brand new customer whose data wasn't recorded yet).
#      We capture this signal before filling in the blanks.

flag_columns = ['tenure', 'warehouse_to_home', 'day_since_last_order']

for col in flag_columns:
    if col in df.columns and df[col].isnull().any():
        new_col = f"{col}_was_missing"
        df[new_col] = df[col].isnull().astype(int)  # 1 = was missing, 0 = had data
        print(f"[OK] Created flag: {new_col}")

# --- Step 2c: Fill remaining missing values with MEDIAN ---
# WHY median and not mean?
#   Example: Salaries = [10k, 12k, 11k, 15k, 1 Crore]
#   Mean = 20 Lakhs (pulled up by the outlier)
#   Median = 12k (the middle value, more representative)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'customer_id' and c != 'churn']

imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print(f"\n[OK] Filled all remaining nulls with median values")
print(f"     Remaining missing values: {df.isnull().sum().sum()}")


# ============================================================
# STEP 3: Feature Engineering (Create New Columns)
# ============================================================
# WHY: Raw features are what the company recorded.
#      Engineered features capture PATTERNS that the raw data doesn't.
#      This is where domain knowledge meets data science.
#
# Example: A customer with tenure=2 months and complaint=1 is
# more dangerous than tenure=24 months with complaint=1.
# But the model sees these as separate columns. By COMBINING them,
# we help the model understand the relationship.
# ============================================================
print_section_header("STEP 3: Feature Engineering")

# --- Feature 1: Cashback per Order ---
# Higher cashback per order = company is investing in this customer
if 'order_count' in df.columns and 'cashback_amount' in df.columns:
    df['cashback_per_order'] = np.where(
        df['order_count'] > 0,
        df['cashback_amount'] / df['order_count'],
        0
    )
    print("[OK] Created: cashback_per_order (cashback / orders)")

# --- Feature 2: Tenure Group (bucketized) ---
# Instead of raw months, group into meaningful business segments
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[-1, 6, 12, 24, 36, 100],
    labels=['0-6m', '7-12m', '13-24m', '25-36m', '37m+']
)
print("[OK] Created: tenure_group (bucketed tenure)")

# --- Feature 3: Is New Customer? ---
# Customers under 6 months are at highest risk (from Day 2 cohort analysis)
df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
print("[OK] Created: is_new_customer (tenure <= 6 months)")

# --- Feature 4: Risk Score (hand-crafted from Day 2 insights) ---
# Combines the top churn signals into a single score
df['risk_score'] = (
    (df['complain'] * 3) +                             # complaint = +3 risk
    ((df['satisfaction_score'] <= 2).astype(int) * 2) + # low satisfaction = +2
    (df['is_new_customer'] * 1)                         # new customer = +1
)
print("[OK] Created: risk_score (composite: complain + satisfaction + tenure)")

# --- Feature 5: High Value Customer ---
# Identifies customers spending above average
cashback_median = df['cashback_amount'].median()
df['is_high_value'] = (df['cashback_amount'] > cashback_median).astype(int)
print("[OK] Created: is_high_value (cashback > median)")

# --- Feature 6: Satisfaction x Complain Interaction ---
# Low satisfaction + complaint = double trouble
df['unhappy_complainer'] = (
    (df['satisfaction_score'] <= 2) & (df['complain'] == 1)
).astype(int)
print("[OK] Created: unhappy_complainer (satisfaction<=2 AND complained)")

# Show all new features
new_features = ['cashback_per_order', 'tenure_group', 'is_new_customer',
                'risk_score', 'is_high_value', 'unhappy_complainer']
existing_new = [f for f in new_features if f in df.columns]
print(f"\n     Total new features created: {len(existing_new)}")
print(f"     Dataset now has {len(df.columns)} columns")


# ============================================================
# STEP 4: Encode Categorical Variables
# ============================================================
# WHY: Models only understand numbers, not text like "Male" or "UPI".
#
# TWO APPROACHES:
#   1. Label Encoding: Male=0, Female=1 (for 2 categories)
#   2. One-Hot Encoding: Creates a new 0/1 column per category
#      (for columns with 3+ categories)
#
# We use Label Encoding for binary columns and One-Hot for the rest.
# ============================================================
print_section_header("STEP 4: Encode Categories -> Numbers")

# Identify text columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns to encode: {cat_cols}")

# Store encoders so we can decode predictions later
encoders = {}

for col in cat_cols:
    unique_vals = df[col].nunique()
    
    if unique_vals == 2:
        # Binary column: use simple Label Encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  [Label Encoded] {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    elif unique_vals > 2:
        # Multi-category: use One-Hot Encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        print(f"  [One-Hot Encoded] {col} -> {list(dummies.columns)}")

print(f"\n[OK] All categories converted to numbers")
print(f"     Dataset shape: {df.shape}")


# ============================================================
# STEP 5: Drop Unnecessary Columns
# ============================================================
print_section_header("STEP 5: Clean Up Columns")

# customer_id is just an identifier, not a feature
# tenure_group was used for feature creation, now redundant
drop_cols = ['customer_id']
if 'tenure_group' in df.columns:
    drop_cols.append('tenure_group')

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"  Dropped: {col}")

print(f"\n  Final feature count: {len(df.columns) - 1} features + 1 target")
print(f"  Columns: {list(df.columns)}")


# ============================================================
# STEP 6: Split into Features (X) and Target (y)
# ============================================================
# X = all columns EXCEPT churn (what the model uses to predict)
# y = the churn column (what the model tries to predict)
# ============================================================
print_section_header("STEP 6: Split Features and Target")

X = df.drop(columns=['churn'])
y = df['churn']

print(f"  X (features): {X.shape} -> {X.shape[1]} features, {X.shape[0]} samples")
print(f"  y (target):   {y.shape} -> {y.value_counts().to_dict()}")


# ============================================================
# STEP 7: Train/Test Split
# ============================================================
# WHY: We need to test the model on data it has NEVER seen.
#      If we train and test on the same data, the model just
#      memorizes the answers (like studying the answer key).
#
# SPLIT: 80% training, 20% testing
# STRATIFY: Keeps the churn ratio the same in both sets
#           (so test set isn't accidentally all retained)
# RANDOM_STATE: Makes the split reproducible (same split every time)
# ============================================================
print_section_header("STEP 7: Train/Test Split (80/20)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,       # 20% for testing
    random_state=42,      # reproducible
    stratify=y            # keep churn ratio balanced in both sets
)

print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test set:     {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Train churn rate: {y_train.mean()*100:.1f}%")
print(f"  Test churn rate:  {y_test.mean()*100:.1f}%")
print(f"  (Both should be ~16.5% thanks to stratify)")


# ============================================================
# STEP 8: Feature Scaling
# ============================================================
# WHY: Some features are on very different scales.
#      tenure: 0 to 61
#      cashback_amount: 0 to 325
#      number_of_address: 1 to 22
#
# Models like Logistic Regression and Neural Networks are
# sensitive to scale. A feature with bigger numbers will
# dominate the model unfairly.
#
# StandardScaler: Converts each feature to mean=0, std=1
#   Formula: scaled_value = (value - mean) / std_deviation
#
# IMPORTANT: We fit the scaler on TRAINING data only, then
# use that same scaler on test data. If we scale on all data
# first, the test set "leaks" information into training.
# ============================================================
print_section_header("STEP 8: Feature Scaling (StandardScaler)")

scaler = StandardScaler()

# Fit on training data, transform both train and test
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"[OK] Scaled {X_train_scaled.shape[1]} features")
print(f"     Before scaling - tenure mean: {X_train['tenure'].mean():.1f}, std: {X_train['tenure'].std():.1f}")
print(f"     After scaling  - tenure mean: {X_train_scaled['tenure'].mean():.4f}, std: {X_train_scaled['tenure'].std():.4f}")
print(f"     (Should be close to mean=0, std=1)")


# ============================================================
# STEP 9: Handle Class Imbalance with SMOTE
# ============================================================
# PROBLEM: We have 83.5% retained vs 16.5% churned.
#          If we train a model on this, it will just predict
#          "not churned" for everyone and get 83.5% accuracy.
#          That's useless for catching actual churners.
#
# SOLUTION: SMOTE (Synthetic Minority Over-Sampling Technique)
#   - Looks at each churned customer
#   - Finds its nearest neighbors (similar churned customers)
#   - Creates NEW synthetic churned customers between them
#   - Result: 50/50 balanced classes for training
#
# IMPORTANT: Only apply SMOTE to TRAINING data, never test data.
# Test data must remain untouched to simulate real-world conditions.
# ============================================================
print_section_header("STEP 9: Balance Classes with SMOTE")

print(f"  BEFORE SMOTE:")
print(f"    Retained (0): {(y_train == 0).sum()}")
print(f"    Churned  (1): {(y_train == 1).sum()}")
print(f"    Ratio: 1:{(y_train == 0).sum() / (y_train == 1).sum():.1f}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n  AFTER SMOTE:")
print(f"    Retained (0): {(y_train_balanced == 0).sum()}")
print(f"    Churned  (1): {(y_train_balanced == 1).sum()}")
print(f"    Ratio: 1:{(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.1f}")
print(f"    New training samples: {len(y_train_balanced)} (was {len(y_train)})")


# ============================================================
# STEP 10: Save Everything for Day 5
# ============================================================
print_section_header("STEP 10: Save Preprocessed Data")

# Save as CSV for easy inspection
X_train_balanced.to_csv("data/processed/X_train.csv", index=False)
y_train_balanced.to_csv("data/processed/y_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("[OK] Saved train/test CSVs to data/processed/")

# Save the scaler and encoders (needed to preprocess new data later)
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
print("[OK] Saved scaler and encoders to models/")

# Save feature names (important for model interpretation)
feature_names = list(X_train_balanced.columns)
with open("data/processed/feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(name + "\n")
print(f"[OK] Saved {len(feature_names)} feature names")

# Quick summary of what's saved
print(f"\nFiles saved:")
print(f"  data/processed/X_train.csv      -> {X_train_balanced.shape} (balanced training features)")
print(f"  data/processed/y_train.csv      -> {y_train_balanced.shape} (balanced training labels)")
print(f"  data/processed/X_test.csv       -> {X_test_scaled.shape} (test features)")
print(f"  data/processed/y_test.csv       -> {y_test.shape} (test labels)")
print(f"  models/scaler.pkl               -> fitted StandardScaler")
print(f"  models/label_encoders.pkl       -> category encoders")
print(f"  data/processed/feature_names.txt -> column names for model input")


# ============================================================
# SUMMARY
# ============================================================
print_section_header("DAY 4 -- COMPLETE")
print(f"""
What we accomplished today:
-----------------------------------------
1. [DONE] Missing Value Handling
     -> Dropped {len(cols_to_drop)} columns with >90% missing
     -> Created 'was_missing' flags for important columns
     -> Filled remaining nulls with median

2. [DONE] Feature Engineering ({len(existing_new)} new features)
     -> cashback_per_order (spending efficiency)
     -> is_new_customer (tenure <= 6 months flag)
     -> risk_score (composite: complain + satisfaction + tenure)
     -> is_high_value (above-median spender)
     -> unhappy_complainer (low satisfaction + complained)

3. [DONE] Categorical Encoding
     -> Binary columns: Label Encoded (0/1)
     -> Multi-category columns: One-Hot Encoded

4. [DONE] Train/Test Split (80/20, stratified)
     -> Train: {X_train.shape[0]} samples
     -> Test: {X_test.shape[0]} samples

5. [DONE] Feature Scaling (StandardScaler)
     -> All features normalized to mean=0, std=1
     -> Fitted on train only (no data leakage)

6. [DONE] Class Balancing (SMOTE)
     -> Balanced training set: {len(y_train_balanced)} samples
     -> 50/50 churn ratio for fair model training

7. [DONE] Saved all preprocessed data + artifacts

Skills Demonstrated:
-----------------------------------------
  Imputation, Feature Engineering, Label/One-Hot Encoding
  StandardScaler, SMOTE, Train/Test Split, Data Leakage Prevention
  joblib serialization, Pipeline architecture

Tomorrow (Day 5):
-----------------------------------------
  -> Train 4 ML models: Logistic Regression, Random Forest,
     Decision Tree, SVM
  -> Evaluate with Accuracy, Precision, Recall, F1, AUC-ROC
  -> Compare model performance head-to-head
""")
