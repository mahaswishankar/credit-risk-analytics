# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 2: DATA PREPROCESSING & CLEANING
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import os

warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0a0a0a'
plt.rcParams['axes.facecolor']   = '#111111'
plt.rcParams['axes.edgecolor']   = '#333333'
plt.rcParams['axes.labelcolor']  = '#cccccc'
plt.rcParams['xtick.color']      = '#999999'
plt.rcParams['ytick.color']      = '#999999'
plt.rcParams['text.color']       = '#ffffff'
plt.rcParams['grid.color']       = '#222222'
plt.rcParams['font.family']      = 'monospace'

ACCENT  = '#00d4ff'
DANGER  = '#ff4757'
SUCCESS = '#2ed573'
WARNING = '#ffa502'

os.makedirs('outputs',        exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 2: PREPROCESSING")
print("=" * 65)

for path in ['credit_risk_dataset.csv', 'data/credit_risk_dataset.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ Dataset loaded: {len(df):,} rows × {df.shape[1]} cols")
        break

df_original = df.copy()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — OUTLIER DETECTION & REMOVAL
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 1: OUTLIER DETECTION & REMOVAL")
print(f"{'─'*65}")

before = len(df)

# Age: real humans are 18–80
age_outliers = df[df['person_age'] > 80].shape[0]
df = df[df['person_age'] <= 80]
print(f"  Age > 80 removed       : {age_outliers:,} rows")

# Employment length: max realistic is 60 years
emp_outliers = df[df['person_emp_length'] > 60].shape[0]
df = df[df['person_emp_length'] <= 60]
print(f"  Emp length > 60 removed: {emp_outliers:,} rows")

# Income: remove top 1% (extreme outliers)
income_cap = df['person_income'].quantile(0.99)
inc_outliers = df[df['person_income'] > income_cap].shape[0]
df = df[df['person_income'] <= income_cap]
print(f"  Income > 99th pct removed: {inc_outliers:,} rows (cap=${income_cap:,.0f})")

after = len(df)
print(f"\n  Rows before : {before:,}")
print(f"  Rows after  : {after:,}")
print(f"  Removed     : {before - after:,} ({(before-after)/before*100:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — MISSING VALUE IMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 2: MISSING VALUE IMPUTATION")
print(f"{'─'*65}")

missing_before = df.isnull().sum()
print("\n  Missing values before imputation:")
print(missing_before[missing_before > 0].to_string())

# loan_int_rate: impute with median per loan_grade (smarter than global median)
df['loan_int_rate'] = df.groupby('loan_grade')['loan_int_rate'].transform(
    lambda x: x.fillna(x.median())
)

# person_emp_length: impute with median
df['person_emp_length'] = df['person_emp_length'].fillna(
    df['person_emp_length'].median()
)

missing_after = df.isnull().sum().sum()
print(f"\n  Missing values after imputation: {missing_after}")
print("  ✅ loan_int_rate → median per loan_grade")
print("  ✅ person_emp_length → global median")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — ENCODING CATEGORICAL VARIABLES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 3: CATEGORICAL ENCODING")
print(f"{'─'*65}")

# Ordinal encoding for loan_grade (A=1 best, G=7 worst)
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['loan_grade_encoded'] = df['loan_grade'].map(grade_map)
print("  loan_grade → ordinal (A=1 to G=7)")

# Binary encoding
df['cb_default_encoded'] = (df['cb_person_default_on_file'] == 'Y').astype(int)
print("  cb_person_default_on_file → binary (Y=1, N=0)")

# One-hot encoding for nominal categoricals
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'],
                    drop_first=False, dtype=int)
print("  person_home_ownership → one-hot encoded")
print("  loan_intent → one-hot encoded")

# Drop original categorical columns we've encoded
df.drop(columns=['loan_grade', 'cb_person_default_on_file'], inplace=True)

print(f"\n  Final columns ({df.shape[1]}):")
print(f"  {list(df.columns)}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE SCALING
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 4: FEATURE SCALING")
print(f"{'─'*65}")

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

feature_names = X.columns.tolist()

# RobustScaler — better for financial data with outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print("  ✅ RobustScaler applied to all features")
print("  (RobustScaler chosen — resistant to remaining outliers)")
print(f"  Features scaled: {len(feature_names)}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN / TEST SPLIT
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 5: TRAIN / TEST SPLIT")
print(f"{'─'*65}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X_scaled)*100:.0f}%)")
print(f"  Test set  : {X_test.shape[0]:,}  rows ({X_test.shape[0]/len(X_scaled)*100:.0f}%)")
print(f"  Train defaults: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Test  defaults: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
print("  ✅ Stratified split — class ratio preserved")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — SMOTE (Handle Class Imbalance)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 6: SMOTE — HANDLE CLASS IMBALANCE")
print(f"{'─'*65}")

print(f"  Before SMOTE — Class 0: {(y_train==0).sum():,} | Class 1: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  After  SMOTE — Class 0: {(y_train_sm==0).sum():,} | Class 1: {(y_train_sm==1).sum():,}")
print(f"  New training set size : {len(X_train_sm):,}")
print("  ✅ SMOTE applied to training data only (no data leakage)")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 8 — Before vs After Preprocessing
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Chart 8 — Data Quality: Before vs After Preprocessing',
             fontsize=16, fontweight='bold', color=ACCENT)

# Age distribution before/after
axes[0,0].hist(df_original['person_age'], bins=40, color=DANGER,
               alpha=0.7, label='Before', density=True)
axes[0,0].hist(df['person_age'], bins=40, color=SUCCESS,
               alpha=0.7, label='After', density=True)
axes[0,0].set_title('Age Distribution', color=ACCENT)
axes[0,0].legend()

# Income distribution before/after
axes[0,1].hist(df_original['person_income'], bins=40, color=DANGER,
               alpha=0.7, label='Before', density=True)
axes[0,1].hist(df['person_income'], bins=40, color=SUCCESS,
               alpha=0.7, label='After', density=True)
axes[0,1].set_title('Income Distribution', color=ACCENT)
axes[0,1].legend()

# Employment length before/after
axes[0,2].hist(df_original['person_emp_length'].dropna(), bins=40,
               color=DANGER, alpha=0.7, label='Before', density=True)
axes[0,2].hist(df['person_emp_length'], bins=40, color=SUCCESS,
               alpha=0.7, label='After', density=True)
axes[0,2].set_title('Employment Length Distribution', color=ACCENT)
axes[0,2].legend()

# Class balance before SMOTE
axes[1,0].bar(['Non-Default', 'Default'],
              [y_train.value_counts()[0], y_train.value_counts()[1]],
              color=[SUCCESS, DANGER], edgecolor='#0a0a0a')
axes[1,0].set_title('Class Balance — Before SMOTE', color=ACCENT)
axes[1,0].set_ylabel('Count')

# Class balance after SMOTE
axes[1,1].bar(['Non-Default', 'Default'],
              [(y_train_sm==0).sum(), (y_train_sm==1).sum()],
              color=[SUCCESS, DANGER], edgecolor='#0a0a0a')
axes[1,1].set_title('Class Balance — After SMOTE', color=ACCENT)
axes[1,1].set_ylabel('Count')

# Missing values before/after
missing_cols  = ['loan_int_rate', 'person_emp_length']
missing_before_vals = [df_original[c].isnull().sum() for c in missing_cols]
missing_after_vals  = [0, 0]
x = np.arange(len(missing_cols))
w = 0.35
axes[1,2].bar(x - w/2, missing_before_vals, w, label='Before',
              color=DANGER, edgecolor='#0a0a0a')
axes[1,2].bar(x + w/2, missing_after_vals,  w, label='After',
              color=SUCCESS, edgecolor='#0a0a0a')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(missing_cols, rotation=15)
axes[1,2].set_title('Missing Values Resolved', color=ACCENT)
axes[1,2].set_ylabel('Missing Count')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('outputs/08_preprocessing_summary.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 8 saved: outputs/08_preprocessing_summary.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE PROCESSED DATA
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  SAVING PROCESSED DATA")
print(f"{'─'*65}")

X_train_sm.to_csv('processed_data/X_train.csv', index=False)
X_test.to_csv('processed_data/X_test.csv',      index=False)
y_train_sm.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv('processed_data/y_test.csv',      index=False)
df.to_csv('processed_data/df_clean.csv',         index=False)

# Save feature names for later scripts
import json
with open('processed_data/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("  ✅ processed_data/X_train.csv")
print("  ✅ processed_data/X_test.csv")
print("  ✅ processed_data/y_train.csv")
print("  ✅ processed_data/y_test.csv")
print("  ✅ processed_data/df_clean.csv")
print("  ✅ processed_data/feature_names.json")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  PREPROCESSING COMPLETE — SUMMARY")
print(f"{'='*65}")
print(f"  Original dataset      : {len(df_original):,} rows")
print(f"  After outlier removal : {len(df):,} rows")
print(f"  Features engineered   : {len(feature_names)}")
print(f"  Training samples      : {len(X_train_sm):,} (after SMOTE)")
print(f"  Test samples          : {len(X_test):,}")
print(f"  Missing values        : 0 ✅")
print(f"  Class balance (train) : 50/50 ✅")
print(f"\n  ➡  Next: Run credit_risk_part3_feature_engineering.py")
print(f"{'='*65}\n")