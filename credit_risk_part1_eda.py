# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 1: EXPLORATORY DATA ANALYSIS
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
plt.rcParams['grid.linewidth']   = 0.5
plt.rcParams['font.family']      = 'monospace'

ACCENT   = '#00d4ff'
DANGER   = '#ff4757'
SUCCESS  = '#2ed573'
WARNING  = '#ffa502'
PURPLE   = '#a55eea'
PALETTE  = [ACCENT, DANGER, SUCCESS, WARNING, PURPLE, '#ff6b81', '#eccc68']

os.makedirs('outputs', exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 1: EDA")
print("=" * 65)

for path in ['credit_risk_dataset.csv',
             'data/credit_risk_dataset.csv',
             '../data/credit_risk_dataset.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ Dataset loaded from: {path}")
        break
else:
    raise FileNotFoundError("❌ credit_risk_dataset.csv not found. "
                            "Place it in the project or data/ folder.")

# ── 1. Basic Overview ─────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  DATASET OVERVIEW")
print(f"{'─'*65}")
print(f"  Rows         : {df.shape[0]:,}")
print(f"  Columns      : {df.shape[1]}")
print(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n  Columns:\n  {list(df.columns)}")

print(f"\n{'─'*65}")
print("  DATA TYPES & MISSING VALUES")
print(f"{'─'*65}")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
dtypes_df = pd.DataFrame({
    'dtype'      : df.dtypes,
    'missing'    : missing,
    'missing_%'  : missing_pct,
    'unique'     : df.nunique()
})
print(dtypes_df.to_string())

# ── 2. Target Variable Analysis ───────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  TARGET VARIABLE: loan_status")
print(f"{'─'*65}")
target_counts = df['loan_status'].value_counts()
target_pct    = df['loan_status'].value_counts(normalize=True) * 100
print(f"  0 = Non-Default : {target_counts[0]:,}  ({target_pct[0]:.1f}%)")
print(f"  1 = Default     : {target_counts[1]:,}  ({target_pct[1]:.1f}%)")
imbalance = target_counts[0] / target_counts[1]
print(f"  Imbalance ratio : {imbalance:.1f}:1")

# ── 3. Descriptive Statistics ─────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("  DESCRIPTIVE STATISTICS (Numerical)")
print(f"{'─'*65}")
print(df.describe().round(2).to_string())

# ═════════════════════════════════════════════════════════════════════════════
# CHART 1 — Target Distribution
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Chart 1 — Loan Default Distribution',
             fontsize=16, fontweight='bold', color=ACCENT, y=1.01)

labels = ['Non-Default', 'Default']
colors = [SUCCESS, DANGER]
sizes  = [target_counts[0], target_counts[1]]

# Pie
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'color': 'white', 'fontsize': 12},
            wedgeprops={'edgecolor': '#0a0a0a', 'linewidth': 2})
axes[0].set_title('Proportion of Defaults', color=ACCENT, fontsize=13)

# Bar
bars = axes[1].bar(labels, sizes, color=colors, edgecolor='#0a0a0a', linewidth=1.5)
for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{val:,}', ha='center', va='bottom', color='white', fontsize=11)
axes[1].set_title('Count of Defaults', color=ACCENT, fontsize=13)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('outputs/01_target_distribution.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 1 saved: outputs/01_target_distribution.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 2 — Numerical Feature Distributions
# ═════════════════════════════════════════════════════════════════════════════
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'loan_status']

n = len(num_cols)
ncols = 3
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
fig.suptitle('Chart 2 — Numerical Feature Distributions',
             fontsize=16, fontweight='bold', color=ACCENT)
axes = axes.flatten()

for i, col in enumerate(num_cols):
    non_def = df[df['loan_status'] == 0][col].dropna()
    default = df[df['loan_status'] == 1][col].dropna()
    axes[i].hist(non_def, bins=40, alpha=0.6, color=SUCCESS,
                 label='Non-Default', density=True)
    axes[i].hist(default, bins=40, alpha=0.6, color=DANGER,
                 label='Default', density=True)
    axes[i].set_title(col, color=ACCENT, fontsize=11)
    axes[i].legend(fontsize=9)
    axes[i].set_xlabel(col, fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/02_numerical_distributions.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 2 saved: outputs/02_numerical_distributions.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 3 — Categorical Feature Analysis
# ═════════════════════════════════════════════════════════════════════════════
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n  Categorical columns: {cat_cols}")

if cat_cols:
    ncols = min(3, len(cat_cols))
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    fig.suptitle('Chart 3 — Categorical Feature vs Default Rate',
                 fontsize=16, fontweight='bold', color=ACCENT)
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten().tolist()

    for i, col in enumerate(cat_cols):
        default_rate = df.groupby(col)['loan_status'].mean().sort_values(ascending=False)
        counts       = df[col].value_counts()
        colors_bar   = [DANGER if r > 0.2 else WARNING if r > 0.1 else SUCCESS
                        for r in default_rate]
        bars = axes[i].bar(default_rate.index, default_rate.values * 100,
                           color=colors_bar, edgecolor='#0a0a0a')
        for bar, (cat, rate) in zip(bars, default_rate.items()):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.3,
                         f'{rate*100:.1f}%\nn={counts[cat]:,}',
                         ha='center', va='bottom', fontsize=8, color='white')
        axes[i].set_title(f'{col} — Default Rate (%)', color=ACCENT, fontsize=12)
        axes[i].set_ylabel('Default Rate (%)')
        axes[i].tick_params(axis='x', rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('outputs/03_categorical_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    print("✅ Chart 3 saved: outputs/03_categorical_analysis.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 4 — Correlation Heatmap
# ═════════════════════════════════════════════════════════════════════════════
corr_df = df[num_cols + ['loan_status']].corr()

fig, ax = plt.subplots(figsize=(14, 11))
fig.suptitle('Chart 4 — Feature Correlation Heatmap',
             fontsize=16, fontweight='bold', color=ACCENT)

mask = np.zeros_like(corr_df, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_df, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 9},
            linewidths=0.5, linecolor='#0a0a0a',
            cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Pearson Correlation — All Features vs loan_status',
             color='#cccccc', fontsize=11, pad=10)

plt.tight_layout()
plt.savefig('outputs/04_correlation_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 4 saved: outputs/04_correlation_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 5 — Loan Amount vs Income by Default Status
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Chart 5 — Loan Amount & Income Analysis',
             fontsize=16, fontweight='bold', color=ACCENT)

# Loan amount by default
for status, color, label in [(0, SUCCESS, 'Non-Default'), (1, DANGER, 'Default')]:
    subset = df[df['loan_status'] == status]['loan_amnt'].dropna()
    axes[0].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
axes[0].set_title('Loan Amount Distribution', color=ACCENT)
axes[0].set_xlabel('Loan Amount ($)')
axes[0].legend()

# Income by default
if 'person_income' in df.columns:
    for status, color, label in [(0, SUCCESS, 'Non-Default'), (1, DANGER, 'Default')]:
        subset = df[df['loan_status'] == status]['person_income'].dropna()
        subset = subset[subset < subset.quantile(0.99)]  # remove extreme outliers
        axes[1].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
    axes[1].set_title('Person Income Distribution', color=ACCENT)
    axes[1].set_xlabel('Annual Income ($)')
    axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/05_loan_income_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 5 saved: outputs/05_loan_income_analysis.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 6 — Interest Rate & Loan Grade vs Default
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Chart 6 — Interest Rate & Loan Grade vs Default',
             fontsize=16, fontweight='bold', color=ACCENT)

# Interest rate boxplot
if 'loan_int_rate' in df.columns:
    data_0 = df[df['loan_status'] == 0]['loan_int_rate'].dropna()
    data_1 = df[df['loan_status'] == 1]['loan_int_rate'].dropna()
    bp = axes[0].boxplot([data_0, data_1],
                         patch_artist=True,
                         labels=['Non-Default', 'Default'])
    bp['boxes'][0].set_facecolor(SUCCESS)
    bp['boxes'][1].set_facecolor(DANGER)
    for median in bp['medians']:
        median.set_color('white')
        median.set_linewidth(2)
    axes[0].set_title('Interest Rate by Default Status', color=ACCENT)
    axes[0].set_ylabel('Interest Rate (%)')

# Loan grade default rate
if 'loan_grade' in df.columns:
    grade_default = df.groupby('loan_grade')['loan_status'].mean().sort_index()
    grade_colors  = [DANGER if r > 0.3 else WARNING if r > 0.15 else SUCCESS
                     for r in grade_default]
    bars = axes[1].bar(grade_default.index, grade_default.values * 100,
                       color=grade_colors, edgecolor='#0a0a0a')
    for bar, val in zip(bars, grade_default.values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.3,
                     f'{val*100:.1f}%', ha='center', va='bottom',
                     color='white', fontsize=10)
    axes[1].set_title('Default Rate by Loan Grade', color=ACCENT)
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].set_xlabel('Loan Grade')

plt.tight_layout()
plt.savefig('outputs/06_interest_grade_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 6 saved: outputs/06_interest_grade_analysis.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 7 — Age & Employment Length vs Default
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Chart 7 — Age & Employment Length vs Default',
             fontsize=16, fontweight='bold', color=ACCENT)

if 'person_age' in df.columns:
    age_bins = [18, 25, 30, 35, 40, 50, 60, 100]
    age_labels = ['18-25', '26-30', '31-35', '36-40', '41-50', '51-60', '60+']
    df['age_group'] = pd.cut(df['person_age'], bins=age_bins, labels=age_labels)
    age_default = df.groupby('age_group', observed=True)['loan_status'].mean()
    bars = axes[0].bar(age_default.index, age_default.values * 100,
                       color=ACCENT, edgecolor='#0a0a0a', alpha=0.85)
    for bar, val in zip(bars, age_default.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.2,
                     f'{val*100:.1f}%', ha='center', va='bottom',
                     color='white', fontsize=9)
    axes[0].set_title('Default Rate by Age Group', color=ACCENT)
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].set_xlabel('Age Group')

if 'person_emp_length' in df.columns:
    emp_default = df.groupby('person_emp_length')['loan_status'].mean()
    axes[1].plot(emp_default.index, emp_default.values * 100,
                 color=WARNING, linewidth=2.5, marker='o',
                 markersize=6, markerfacecolor=DANGER)
    axes[1].fill_between(emp_default.index, emp_default.values * 100,
                         alpha=0.2, color=WARNING)
    axes[1].set_title('Default Rate by Employment Length', color=ACCENT)
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].set_xlabel('Employment Length (Years)')

plt.tight_layout()
plt.savefig('outputs/07_age_employment_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 7 saved: outputs/07_age_employment_analysis.png")

# ═════════════════════════════════════════════════════════════════════════════
# EDA SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  EDA COMPLETE — KEY FINDINGS")
print(f"{'='*65}")
print(f"  Total records      : {len(df):,}")
print(f"  Default rate       : {df['loan_status'].mean()*100:.1f}%")
print(f"  Class imbalance    : {imbalance:.1f}:1")
print(f"  Missing values     : {df.isnull().sum().sum():,} total")
print(f"  Numerical features : {len(num_cols)}")
print(f"  Categorical features: {len(cat_cols)}")

if 'loan_int_rate' in df.columns:
    avg_int_def    = df[df['loan_status']==1]['loan_int_rate'].mean()
    avg_int_nondef = df[df['loan_status']==0]['loan_int_rate'].mean()
    print(f"\n  Avg interest rate (Default)    : {avg_int_def:.1f}%")
    print(f"  Avg interest rate (Non-Default): {avg_int_nondef:.1f}%")

if 'loan_grade' in df.columns:
    worst_grade = df.groupby('loan_grade')['loan_status'].mean().idxmax()
    worst_rate  = df.groupby('loan_grade')['loan_status'].mean().max()
    print(f"\n  Highest risk grade : {worst_grade} ({worst_rate*100:.1f}% default rate)")

print(f"\n  Charts saved to outputs/ folder (01–07)")
print(f"\n  ➡  Next: Run credit_risk_part2_preprocessing.py")
print(f"{'='*65}\n")