# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 7: RISK SCORING SYSTEM
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
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
ORANGE  = '#ff6348'

os.makedirs('outputs', exist_ok=True)

# ── Load Data & Model ─────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 7: RISK SCORING")
print("=" * 65)

X_test = pd.read_csv('processed_data/X_test_engineered.csv')
y_test = pd.read_csv('processed_data/y_test_engineered.csv').squeeze()
df_clean = pd.read_csv('processed_data/df_clean.csv')

with open('processed_data/all_features.json') as f:
    feature_names = json.load(f)

with open('models/XGBoost.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"\n✅ Model and data loaded")
print(f"✅ Test set: {X_test.shape[0]:,} rows")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — PREDICT PROBABILITIES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 1: PREDICTING DEFAULT PROBABILITIES")
print(f"{'─'*65}")

probs = model.predict_proba(X_test)[:, 1]
print(f"  Min probability  : {probs.min():.4f}")
print(f"  Max probability  : {probs.max():.4f}")
print(f"  Mean probability : {probs.mean():.4f}")
print(f"  Median probability: {np.median(probs):.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CREDIT SCORE (300-850 scale like FICO)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 2: CREDIT SCORE CALCULATION (300-850 FICO Scale)")
print(f"{'─'*65}")

# Convert default probability to credit score
# Higher prob = lower score (more risky)
# FICO: 300 (worst) to 850 (best)
credit_scores = 850 - (probs * 550).astype(int)
credit_scores = np.clip(credit_scores, 300, 850)

print(f"  Score range: {credit_scores.min()} - {credit_scores.max()}")
print(f"  Mean score : {credit_scores.mean():.0f}")
print(f"  Median score: {np.median(credit_scores):.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — CREDIT GRADE ASSIGNMENT
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 3: CREDIT GRADE ASSIGNMENT")
print(f"{'─'*65}")

def assign_grade(prob):
    if prob < 0.05:   return 'A'   # Very Low Risk
    elif prob < 0.15: return 'B'   # Low Risk
    elif prob < 0.30: return 'C'   # Moderate Risk
    elif prob < 0.50: return 'D'   # High Risk
    else:             return 'E'   # Very High Risk

def assign_decision(prob):
    if prob < 0.15:   return 'APPROVE'
    elif prob < 0.35: return 'REVIEW'
    else:             return 'REJECT'

def assign_risk_level(prob):
    if prob < 0.05:   return 'Very Low'
    elif prob < 0.15: return 'Low'
    elif prob < 0.30: return 'Moderate'
    elif prob < 0.50: return 'High'
    else:             return 'Very High'

grades    = np.array([assign_grade(p)     for p in probs])
decisions = np.array([assign_decision(p)  for p in probs])
risk_lvls = np.array([assign_risk_level(p) for p in probs])

# Grade distribution
grade_counts = pd.Series(grades).value_counts().sort_index()
print(f"\n  Grade Distribution:")
grade_info = {
    'A': ('< 5%',   'Very Low Risk',  'APPROVE'),
    'B': ('5-15%',  'Low Risk',       'APPROVE'),
    'C': ('15-30%', 'Moderate Risk',  'REVIEW'),
    'D': ('30-50%', 'High Risk',      'REJECT'),
    'E': ('> 50%',  'Very High Risk', 'REJECT'),
}
for grade in ['A', 'B', 'C', 'D', 'E']:
    count = grade_counts.get(grade, 0)
    pct   = count / len(grades) * 100
    info  = grade_info[grade]
    print(f"  Grade {grade} ({info[0]:>8}) : {count:>5,} loans "
          f"({pct:>5.1f}%) — {info[1]:<20} → {info[2]}")

# Decision distribution
dec_counts = pd.Series(decisions).value_counts()
print(f"\n  Decision Distribution:")
for dec, count in dec_counts.items():
    pct = count / len(decisions) * 100
    print(f"  {dec:<10}: {count:>5,} ({pct:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD RISK SCORECARD
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  STEP 4: BUILDING RISK SCORECARD")
print(f"{'─'*65}")

results_df = pd.DataFrame({
    'default_probability' : probs,
    'credit_score'        : credit_scores,
    'grade'               : grades,
    'risk_level'          : risk_lvls,
    'decision'            : decisions,
    'actual_default'      : y_test.values
})

# Accuracy by grade
print(f"\n  Grade Performance Analysis:")
print(f"  {'Grade':<8} {'Count':>6} {'Avg Prob':>10} {'Avg Score':>10} "
      f"{'Act Default%':>13} {'Decision':>10}")
print(f"  {'─'*60}")
for grade in ['A', 'B', 'C', 'D', 'E']:
    subset = results_df[results_df['grade'] == grade]
    if len(subset) == 0:
        continue
    act_def = subset['actual_default'].mean() * 100
    avg_prob = subset['default_probability'].mean()
    avg_score = subset['credit_score'].mean()
    decision = grade_info[grade][2]
    print(f"  {grade:<8} {len(subset):>6,} {avg_prob:>10.3f} "
          f"{avg_score:>10.0f} {act_def:>12.1f}% {decision:>10}")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 22 — Risk Score Distribution
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Chart 22 — Credit Risk Score Distribution',
             fontsize=16, fontweight='bold', color=ACCENT)

# Credit score distribution
axes[0,0].hist(credit_scores[y_test == 0], bins=40, alpha=0.6,
               color=SUCCESS, label='Non-Default', density=True)
axes[0,0].hist(credit_scores[y_test == 1], bins=40, alpha=0.6,
               color=DANGER, label='Default', density=True)
axes[0,0].axvline(x=620, color=WARNING, linestyle='--', alpha=0.8, label='Approve threshold (620)')
axes[0,0].axvline(x=500, color=ORANGE,  linestyle='--', alpha=0.8, label='Review threshold (500)')
axes[0,0].set_title('Credit Score Distribution (FICO Scale)', color=ACCENT)
axes[0,0].set_xlabel('Credit Score')
axes[0,0].legend(fontsize=8)

# Grade distribution pie
grade_colors = [SUCCESS, '#7bed9f', WARNING, ORANGE, DANGER]
grade_labels = [f'Grade {g}\n({grade_info[g][1]})' for g in ['A','B','C','D','E']]
grade_vals   = [grade_counts.get(g, 0) for g in ['A','B','C','D','E']]
axes[0,1].pie(grade_vals, labels=grade_labels, colors=grade_colors,
              autopct='%1.1f%%', startangle=90,
              textprops={'color': 'white', 'fontsize': 9},
              wedgeprops={'edgecolor': '#0a0a0a', 'linewidth': 2})
axes[0,1].set_title('Credit Grade Distribution', color=ACCENT)

# Decision distribution
dec_colors = {
    'APPROVE': SUCCESS,
    'REVIEW' : WARNING,
    'REJECT' : DANGER
}
dec_order = ['APPROVE', 'REVIEW', 'REJECT']
dec_vals  = [dec_counts.get(d, 0) for d in dec_order]
dec_clrs  = [dec_colors[d] for d in dec_order]
bars = axes[1,0].bar(dec_order, dec_vals, color=dec_clrs, edgecolor='#0a0a0a')
for bar, val in zip(bars, dec_vals):
    pct = val / len(decisions) * 100
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 20,
                   f'{val:,}\n({pct:.1f}%)',
                   ha='center', va='bottom', color='white', fontsize=10)
axes[1,0].set_title('Loan Decision Distribution', color=ACCENT)
axes[1,0].set_ylabel('Count')

# Actual default rate by grade
grade_default_rates = []
for grade in ['A', 'B', 'C', 'D', 'E']:
    subset = results_df[results_df['grade'] == grade]
    rate   = subset['actual_default'].mean() * 100 if len(subset) > 0 else 0
    grade_default_rates.append(rate)

bar_colors = [SUCCESS, '#7bed9f', WARNING, ORANGE, DANGER]
bars = axes[1,1].bar(['A','B','C','D','E'], grade_default_rates,
                     color=bar_colors, edgecolor='#0a0a0a')
for bar, val in zip(bars, grade_default_rates):
    axes[1,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.3,
                   f'{val:.1f}%', ha='center', va='bottom',
                   color='white', fontsize=10, fontweight='bold')
axes[1,1].set_title('Actual Default Rate by Credit Grade\n(Validates our grading system)',
                    color=ACCENT)
axes[1,1].set_ylabel('Default Rate (%)')
axes[1,1].set_xlabel('Credit Grade')

plt.tight_layout()
plt.savefig('outputs/22_risk_score_distribution.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 22 saved: outputs/22_risk_score_distribution.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 23 — Risk Scorecard Heatmap
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle('Chart 23 — Credit Risk Scorecard Reference',
             fontsize=16, fontweight='bold', color=ACCENT)

scorecard_data = {
    'Grade' : ['A', 'B', 'C', 'D', 'E'],
    'Default Prob Range' : ['< 5%', '5-15%', '15-30%', '30-50%', '> 50%'],
    'Credit Score Range' : ['770-850', '685-770', '535-685', '300-535', '< 300'],
    'Risk Level'         : ['Very Low', 'Low', 'Moderate', 'High', 'Very High'],
    'Decision'           : ['APPROVE', 'APPROVE', 'REVIEW', 'REJECT', 'REJECT'],
    'Interest Premium'   : ['+0%', '+1-2%', '+3-5%', '+6-8%', 'N/A'],
    'Count in Test'      : [grade_counts.get(g, 0) for g in ['A','B','C','D','E']],
}

sc_df = pd.DataFrame(scorecard_data)

cell_colors = []
row_colors  = [
    ['#0a2a0a', '#0a2a0a', '#0a2a0a', '#0a2a0a', '#0a2a0a', '#0a2a0a', '#0a2a0a'],
    ['#1a2a0a', '#1a2a0a', '#1a2a0a', '#1a2a0a', '#1a2a0a', '#1a2a0a', '#1a2a0a'],
    ['#2a2a0a', '#2a2a0a', '#2a2a0a', '#2a2a0a', '#2a2a0a', '#2a2a0a', '#2a2a0a'],
    ['#2a1a0a', '#2a1a0a', '#2a1a0a', '#2a1a0a', '#2a1a0a', '#2a1a0a', '#2a1a0a'],
    ['#2a0a0a', '#2a0a0a', '#2a0a0a', '#2a0a0a', '#2a0a0a', '#2a0a0a', '#2a0a0a'],
]

table = ax.table(
    cellText  = sc_df.values,
    colLabels = sc_df.columns,
    cellLoc   = 'center',
    loc       = 'center',
    cellColours = row_colors
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#333333')
    cell.set_text_props(color='white')
    if row == 0:
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color=ACCENT, fontweight='bold')

ax.axis('off')
ax.set_title('JP Morgan Style Credit Risk Scorecard',
             color='#cccccc', fontsize=11, pad=20)

plt.tight_layout()
plt.savefig('outputs/23_risk_scorecard.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 23 saved: outputs/23_risk_scorecard.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE SCORING SYSTEM FOR WEB APP
# ═════════════════════════════════════════════════════════════════════════════
scoring_config = {
    'grade_thresholds': {
        'A': [0.00, 0.05],
        'B': [0.05, 0.15],
        'C': [0.15, 0.30],
        'D': [0.30, 0.50],
        'E': [0.50, 1.00]
    },
    'decision_thresholds': {
        'APPROVE': [0.00, 0.15],
        'REVIEW' : [0.15, 0.35],
        'REJECT' : [0.35, 1.00]
    },
    'score_range': [300, 850],
    'grade_descriptions': grade_info
}
with open('processed_data/scoring_config.json', 'w') as f:
    json.dump(scoring_config, f, indent=2)
print("\n✅ processed_data/scoring_config.json saved (for web app)")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
approve_count = dec_counts.get('APPROVE', 0)
review_count  = dec_counts.get('REVIEW',  0)
reject_count  = dec_counts.get('REJECT',  0)

print(f"\n{'='*65}")
print("  RISK SCORING COMPLETE — FINAL SUMMARY")
print(f"{'='*65}")
print(f"\n  Total loans assessed : {len(results_df):,}")
print(f"  APPROVE              : {approve_count:,} ({approve_count/len(results_df)*100:.1f}%)")
print(f"  REVIEW               : {review_count:,}  ({review_count/len(results_df)*100:.1f}%)")
print(f"  REJECT               : {reject_count:,}  ({reject_count/len(results_df)*100:.1f}%)")
print(f"\n  Grade breakdown:")
for grade in ['A', 'B', 'C', 'D', 'E']:
    count = grade_counts.get(grade, 0)
    print(f"  Grade {grade}: {count:,} loans")
print(f"\n  Charts saved → outputs/ (22-23)")
print(f"\n{'='*65}")
print("  🎉 ALL 7 PARTS COMPLETE!")
print(f"{'='*65}")
print(f"\n  Total charts generated : 23")
print(f"  Total Python scripts   : 7")
print(f"  Models trained         : 5 (LR, RF, XGB, GBM, Deep NN)")
print(f"  Best model             : XGBoost (AUC-ROC: 0.9348)")
print(f"  Features engineered    : 20 (39 total)")
print(f"\n  ➡  Next: Build React + Flask Web App!")
print(f"{'='*65}\n")