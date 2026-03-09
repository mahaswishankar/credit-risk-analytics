# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 3: FEATURE ENGINEERING
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import json
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
PURPLE  = '#a55eea'

os.makedirs('outputs',        exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

# ── Load Clean Data ───────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 3: FEATURE ENGINEERING")
print("=" * 65)

df = pd.read_csv('processed_data/df_clean.csv')
print(f"\n✅ Clean dataset loaded: {len(df):,} rows × {df.shape[1]} cols")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 1 — DEBT & FINANCIAL STRESS FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  FEATURE GROUP 1: DEBT & FINANCIAL STRESS")
print(f"{'─'*65}")

# Debt-to-Income Ratio (DTI) — classic banking metric
df['dti_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
print("  ✅ dti_ratio          = loan_amnt / person_income")

# Monthly payment estimate (simple interest approximation)
df['monthly_payment'] = (df['loan_amnt'] * (df['loan_int_rate'] / 100)) / 12
print("  ✅ monthly_payment    = loan_amnt × interest_rate / 12")

# Payment-to-income ratio
df['payment_to_income'] = df['monthly_payment'] / (df['person_income'] / 12 + 1)
print("  ✅ payment_to_income  = monthly_payment / monthly_income")

# Total interest burden over loan life (assume 3yr avg)
df['total_interest_burden'] = df['loan_amnt'] * (df['loan_int_rate'] / 100) * 3
print("  ✅ total_interest_burden = loan_amnt × rate × 3 years")

# Financial stress index — combined debt pressure
df['financial_stress_index'] = (
    df['dti_ratio'] * 0.4 +
    df['payment_to_income'] * 0.4 +
    df['loan_percent_income'] * 0.2
)
print("  ✅ financial_stress_index = weighted(dti + pti + pct_income)")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 2 — CREDIT RISK FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  FEATURE GROUP 2: CREDIT RISK INDICATORS")
print(f"{'─'*65}")

# Interest rate premium — how much above minimum is their rate
min_rate = df['loan_int_rate'].min()
df['rate_premium'] = df['loan_int_rate'] - min_rate
print("  ✅ rate_premium       = loan_int_rate - minimum_rate")

# Risk score from grade (higher grade = higher risk)
df['grade_risk_score'] = df['loan_grade_encoded'] / 7.0
print("  ✅ grade_risk_score   = loan_grade_encoded / 7 (normalized)")

# Combined credit risk signal
df['credit_risk_signal'] = (
    df['grade_risk_score'] * 0.5 +
    df['cb_default_encoded'] * 0.3 +
    (df['loan_int_rate'] / df['loan_int_rate'].max()) * 0.2
)
print("  ✅ credit_risk_signal = weighted(grade + history + rate)")

# High interest flag (above 75th percentile)
int_rate_75 = df['loan_int_rate'].quantile(0.75)
df['high_interest_flag'] = (df['loan_int_rate'] > int_rate_75).astype(int)
print(f"  ✅ high_interest_flag = 1 if rate > {int_rate_75:.1f}% (75th pct)")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 3 — BORROWER PROFILE FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  FEATURE GROUP 3: BORROWER PROFILE")
print(f"{'─'*65}")

# Income per year of experience
df['income_per_emp_year'] = df['person_income'] / (df['person_emp_length'] + 1)
print("  ✅ income_per_emp_year = income / (emp_length + 1)")

# Age group risk (young borrowers tend to default more)
df['is_young_borrower'] = (df['person_age'] < 25).astype(int)
print("  ✅ is_young_borrower  = 1 if age < 25")

# Credit history per age (how long they've had credit relative to age)
df['credit_history_ratio'] = df['cb_person_cred_hist_length'] / df['person_age']
print("  ✅ credit_history_ratio = cred_hist_length / age")

# Income stability score
df['income_stability'] = np.log1p(df['person_income']) * df['person_emp_length'] / 100
print("  ✅ income_stability   = log(income) × emp_length / 100")

# Loan size relative to age (older = can handle bigger loans)
df['loan_per_age'] = df['loan_amnt'] / df['person_age']
print("  ✅ loan_per_age       = loan_amnt / person_age")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 4 — INTERACTION FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  FEATURE GROUP 4: INTERACTION FEATURES")
print(f"{'─'*65}")

# Grade × Interest rate interaction
df['grade_rate_interaction'] = df['loan_grade_encoded'] * df['loan_int_rate']
print("  ✅ grade_rate_interaction = grade × interest_rate")

# Income × Credit history interaction
df['income_credit_interaction'] = (
    np.log1p(df['person_income']) * df['cb_person_cred_hist_length']
)
print("  ✅ income_credit_interaction = log(income) × cred_hist")

# Debt burden × Prior default interaction (risky combo!)
df['debt_default_interaction'] = df['dti_ratio'] * df['cb_default_encoded']
print("  ✅ debt_default_interaction  = dti_ratio × prior_default")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP 5 — BINNED / CATEGORICAL FEATURES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  FEATURE GROUP 5: BINNED FEATURES")
print(f"{'─'*65}")

# Loan amount bins
df['loan_size_bin'] = pd.cut(df['loan_amnt'],
                              bins=[0, 5000, 10000, 20000, 35001],
                              labels=[1, 2, 3, 4])
df['loan_size_bin'] = df['loan_size_bin'].astype(float)
print("  ✅ loan_size_bin      = Small/Medium/Large/XLarge (1-4)")

# Income bins
df['income_bin'] = pd.cut(df['person_income'],
                           bins=[0, 30000, 60000, 100000, 999999],
                           labels=[1, 2, 3, 4])
df['income_bin'] = df['income_bin'].astype(float)
print("  ✅ income_bin         = Low/Mid/High/VeryHigh (1-4)")

# Interest rate bins
df['rate_bin'] = pd.cut(df['loan_int_rate'],
                         bins=[0, 8, 12, 16, 25],
                         labels=[1, 2, 3, 4])
df['rate_bin'] = df['rate_bin'].astype(float)
print("  ✅ rate_bin           = Low/Mid/High/VeryHigh (1-4)")

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
new_features = [
    'dti_ratio', 'monthly_payment', 'payment_to_income',
    'total_interest_burden', 'financial_stress_index',
    'rate_premium', 'grade_risk_score', 'credit_risk_signal',
    'high_interest_flag', 'income_per_emp_year', 'is_young_borrower',
    'credit_history_ratio', 'income_stability', 'loan_per_age',
    'grade_rate_interaction', 'income_credit_interaction',
    'debt_default_interaction', 'loan_size_bin', 'income_bin', 'rate_bin'
]

print(f"\n{'─'*65}")
print(f"  TOTAL NEW FEATURES CREATED: {len(new_features)}")
print(f"{'─'*65}")
for i, f in enumerate(new_features, 1):
    print(f"  {i:2d}. {f}")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 9 — New Feature Correlations with Target
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  COMPUTING FEATURE CORRELATIONS WITH TARGET")
print(f"{'─'*65}")

correlations = df[new_features + ['loan_status']].corr()['loan_status'].drop('loan_status')
correlations_sorted = correlations.abs().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Chart 9 — Engineered Feature Correlations with Default',
             fontsize=16, fontweight='bold', color=ACCENT)

# Absolute correlation bar chart
colors = [DANGER if correlations[f] > 0 else SUCCESS for f in correlations_sorted.index]
bars = axes[0].barh(correlations_sorted.index,
                    correlations_sorted.values,
                    color=colors, edgecolor='#0a0a0a', alpha=0.85)
axes[0].set_title('|Correlation| with loan_status', color=ACCENT, fontsize=12)
axes[0].set_xlabel('Absolute Correlation')
axes[0].axvline(x=0.1, color=WARNING, linestyle='--', alpha=0.7, label='0.1 threshold')
axes[0].legend()
for bar, val in zip(bars, correlations_sorted.values):
    axes[0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=8, color='white')

# Signed correlation
corr_signed = correlations.sort_values()
colors2 = [DANGER if v > 0 else SUCCESS for v in corr_signed.values]
axes[1].barh(corr_signed.index, corr_signed.values,
             color=colors2, edgecolor='#0a0a0a', alpha=0.85)
axes[1].set_title('Signed Correlation (Red=+Default, Green=-Default)',
                   color=ACCENT, fontsize=12)
axes[1].set_xlabel('Correlation Coefficient')
axes[1].axvline(x=0, color='white', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/09_feature_correlations.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 9 saved: outputs/09_feature_correlations.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 10 — Top Engineered Features Distribution
# ═════════════════════════════════════════════════════════════════════════════
top_features = correlations_sorted.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Chart 10 — Top 6 Engineered Features vs Default',
             fontsize=16, fontweight='bold', color=ACCENT)
axes = axes.flatten()

for i, feat in enumerate(top_features):
    non_def = df[df['loan_status'] == 0][feat].dropna()
    default = df[df['loan_status'] == 1][feat].dropna()
    axes[i].hist(non_def, bins=40, alpha=0.6, color=SUCCESS,
                 label='Non-Default', density=True)
    axes[i].hist(default, bins=40, alpha=0.6, color=DANGER,
                 label='Default', density=True)
    corr_val = correlations[feat]
    axes[i].set_title(f'{feat}\ncorr={corr_val:.3f}', color=ACCENT, fontsize=10)
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/10_top_features_distribution.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 10 saved: outputs/10_top_features_distribution.png")

# ═════════════════════════════════════════════════════════════════════════════
# REBUILD TRAIN/TEST WITH ALL FEATURES & SAVE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  REBUILDING TRAIN/TEST WITH ENGINEERED FEATURES")
print(f"{'─'*65}")

# Fill any NaN from feature engineering
df = df.fillna(df.median(numeric_only=True))

X = df.drop('loan_status', axis=1)
y = df['loan_status']

all_features = X.columns.tolist()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=all_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  Total features (original + engineered): {len(all_features)}")
print(f"  Training samples (after SMOTE)         : {len(X_train_sm):,}")
print(f"  Test samples                           : {len(X_test):,}")

# Save updated processed data
X_train_sm.to_csv('processed_data/X_train_engineered.csv', index=False)
X_test.to_csv('processed_data/X_test_engineered.csv',      index=False)
y_train_sm.to_csv('processed_data/y_train_engineered.csv', index=False)
y_test.to_csv('processed_data/y_test_engineered.csv',      index=False)

with open('processed_data/all_features.json', 'w') as f:
    json.dump(all_features, f)

print("\n  ✅ processed_data/X_train_engineered.csv")
print("  ✅ processed_data/X_test_engineered.csv")
print("  ✅ processed_data/y_train_engineered.csv")
print("  ✅ processed_data/y_test_engineered.csv")
print("  ✅ processed_data/all_features.json")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  FEATURE ENGINEERING COMPLETE — SUMMARY")
print(f"{'='*65}")
print(f"  Original features     : 19")
print(f"  New features created  : {len(new_features)}")
print(f"  Total features        : {len(all_features)}")
print(f"\n  Top 5 features by correlation with default:")
for feat in correlations_sorted.head(5).index:
    print(f"    {feat:<35} {correlations[feat]:+.4f}")
print(f"\n  ➡  Next: Run credit_risk_part4_models.py")
print(f"{'='*65}\n")