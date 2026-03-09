# =============================================================================
# TRAIN ON STARTUP - Runs when Render server boots
# Trains XGBoost model from scratch using the dataset
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import json
import os

print("=" * 60)
print("  TRAINING MODEL ON STARTUP...")
print("=" * 60)

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv('data/credit_risk_dataset.csv')
print(f"✅ Dataset loaded: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────
df = df[df['person_age'] <= 80]
df = df[df['person_emp_length'] <= 60]
income_99 = df['person_income'].quantile(0.99)
df = df[df['person_income'] <= income_99]

# Impute missing
for grade in df['loan_grade'].unique():
    mask = (df['loan_grade'] == grade) & (df['loan_int_rate'].isna())
    median_rate = df[df['loan_grade'] == grade]['loan_int_rate'].median()
    df.loc[mask, 'loan_int_rate'] = median_rate
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# ── Encode ────────────────────────────────────────────────────
grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
df['loan_grade_encoded']   = df['loan_grade'].map(grade_map)
df['cb_default_encoded']   = (df['cb_person_default_on_file'] == 'Y').astype(int)
df = pd.get_dummies(df, columns=['person_home_ownership','loan_intent'])

# Drop originals
df.drop(['loan_grade','cb_person_default_on_file'], axis=1, inplace=True)

# ── Feature Engineering ───────────────────────────────────────
df['dti_ratio']               = df['loan_amnt'] / (df['person_income'] + 1)
df['monthly_payment']         = df['loan_amnt'] * (df['loan_int_rate'] / 100) / 12
df['payment_to_income']       = df['monthly_payment'] / (df['person_income'] / 12 + 1)
df['total_interest_burden']   = df['loan_amnt'] * (df['loan_int_rate'] / 100) * 3
df['financial_stress_index']  = (df['dti_ratio'] * 0.4 +
                                  df['payment_to_income'] * 0.4 +
                                  df['loan_percent_income'] * 0.2)
df['rate_premium']            = df['loan_int_rate'] - 5.42
df['grade_risk_score']        = df['loan_grade_encoded'] / 7.0
df['credit_risk_signal']      = (df['grade_risk_score'] * 0.5 +
                                  df['cb_default_encoded'] * 0.3 +
                                  (df['loan_int_rate'] / 23.22) * 0.2)
df['high_interest_flag']      = (df['loan_int_rate'] > 13.5).astype(int)
df['income_per_emp_year']     = df['person_income'] / (df['person_emp_length'] + 1)
df['is_young_borrower']       = (df['person_age'] < 25).astype(int)
df['credit_history_ratio']    = df['cb_person_cred_hist_length'] / df['person_age']
df['income_stability']        = np.log1p(df['person_income']) * df['person_emp_length'] / 100
df['loan_per_age']            = df['loan_amnt'] / df['person_age']
df['grade_rate_interaction']  = df['loan_grade_encoded'] * df['loan_int_rate']
df['income_credit_interaction']= np.log1p(df['person_income']) * df['cb_person_cred_hist_length']
df['debt_default_interaction'] = df['dti_ratio'] * df['cb_default_encoded']

df['loan_size_bin'] = pd.cut(df['loan_amnt'],
    bins=[0,5000,10000,20000,float('inf')], labels=[1,2,3,4]).astype(int)
df['income_bin'] = pd.cut(df['person_income'],
    bins=[0,30000,60000,100000,float('inf')], labels=[1,2,3,4]).astype(int)
df['rate_bin'] = pd.cut(df['loan_int_rate'],
    bins=[0,8,12,16,float('inf')], labels=[1,2,3,4]).astype(int)

# ── Split ─────────────────────────────────────────────────────
target       = 'loan_status'
feature_cols = [c for c in df.columns if c != target]

X = df[feature_cols]
y = df[target]

# Ensure consistent column order
feature_names = list(X.columns)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ── Train XGBoost ─────────────────────────────────────────────
print("Training XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_sm, y_train_sm)
print("✅ XGBoost trained!")

# ── Save ──────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

with open('models/XGBoost.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('processed_data/all_features.json', 'w') as f:
    json.dump(feature_names, f)

scoring_config = {
    'grade_thresholds': {
        'A': [0.00, 0.05], 'B': [0.05, 0.15],
        'C': [0.15, 0.30], 'D': [0.30, 0.50], 'E': [0.50, 1.00]
    },
    'decision_thresholds': {
        'APPROVE': [0.00, 0.15],
        'REVIEW' : [0.15, 0.35],
        'REJECT' : [0.35, 1.00]
    },
    'score_range': [300, 850]
}
with open('processed_data/scoring_config.json', 'w') as f:
    json.dump(scoring_config, f)

shap_summary = {
    'top_features': feature_names[:15],
    'feature_names': feature_names,
    'expected_value': 0.0
}
with open('processed_data/shap_summary.json', 'w') as f:
    json.dump(shap_summary, f)

print("✅ All files saved!")
print("=" * 60)
print("  TRAINING COMPLETE — SERVER READY")
print("=" * 60)
