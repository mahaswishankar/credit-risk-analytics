# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 6: SHAP EXPLAINABILITY
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
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

os.makedirs('outputs', exist_ok=True)

# ── Load Data & Model ─────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 6: SHAP EXPLAINABILITY")
print("=" * 65)

X_test  = pd.read_csv('processed_data/X_test_engineered.csv')
y_test  = pd.read_csv('processed_data/y_test_engineered.csv').squeeze()

with open('processed_data/all_features.json') as f:
    feature_names = json.load(f)

with open('models/XGBoost.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"\n✅ XGBoost model loaded")
print(f"✅ Test set: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")

# ── Sample for SHAP (balanced) ────────────────────────────────────────────────
np.random.seed(42)
idx_default    = np.where(y_test == 1)[0]
idx_nondefault = np.where(y_test == 0)[0]
sample_idx = np.concatenate([
    np.random.choice(idx_default,    min(500, len(idx_default)),    replace=False),
    np.random.choice(idx_nondefault, min(500, len(idx_nondefault)), replace=False)
])
X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"\n✅ SHAP sample: {len(X_sample)} rows (500 default + 500 non-default)")

# ═════════════════════════════════════════════════════════════════════════════
# COMPUTE SHAP VALUES
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  COMPUTING SHAP VALUES...")
print(f"{'─'*65}")

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
expected_value = explainer.expected_value

print(f"✅ SHAP values computed")
print(f"   Shape       : {shap_values.shape}")
print(f"   Expected val: {expected_value:.4f}")

# Top features by mean absolute SHAP
mean_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_names
).sort_values(ascending=False)

print(f"\n  Top 10 features by SHAP importance:")
for i, (feat, val) in enumerate(mean_shap.head(10).items(), 1):
    print(f"  {i:2d}. {feat:<35} {val:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 18 — SHAP Global Feature Importance
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Chart 18 — SHAP Global Feature Importance',
             fontsize=16, fontweight='bold', color=ACCENT)

# Bar chart — mean |SHAP|
top15 = mean_shap.head(15).sort_values()
colors = [SUCCESS if 'ratio' in f or 'stress' in f or 'risk' in f or
          'interaction' in f or 'payment' in f else ACCENT
          for f in top15.index]
bars = axes[0].barh(top15.index, top15.values,
                    color=colors, edgecolor='#0a0a0a', alpha=0.85)
for bar, val in zip(bars, top15.values):
    axes[0].text(bar.get_width() + 0.001,
                 bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=8, color='white')
axes[0].set_title('Mean |SHAP Value| — Top 15 Features\n(Green = Engineered Features)',
                  color=ACCENT, fontsize=11)
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].axvline(x=mean_shap.mean(), color=WARNING, linestyle='--',
                alpha=0.7, label='Mean importance')
axes[0].legend()

# SHAP summary dot plot (manual implementation)
top10_feats = mean_shap.head(10).index.tolist()
for i, feat in enumerate(reversed(top10_feats)):
    feat_idx  = feature_names.index(feat)
    sv        = shap_values[:, feat_idx]
    fv        = X_sample[feat].values
    fv_norm   = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
    colors_scatter = plt.cm.RdYlGn(1 - fv_norm)
    axes[1].scatter(sv, np.full_like(sv, i) + np.random.uniform(-0.2, 0.2, len(sv)),
                    c=colors_scatter, alpha=0.5, s=12)

axes[1].set_yticks(range(len(top10_feats)))
axes[1].set_yticklabels(list(reversed(top10_feats)), fontsize=9)
axes[1].axvline(x=0, color='white', linewidth=0.8, alpha=0.5)
axes[1].set_title('SHAP Value Distribution (Red=High Feature Value, Green=Low)',
                  color=ACCENT, fontsize=11)
axes[1].set_xlabel('SHAP Value (impact on model output)')

plt.tight_layout()
plt.savefig('outputs/18_shap_global_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 18 saved: outputs/18_shap_global_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 19 — SHAP Dependence Plots (Top 4 Features)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Chart 19 — SHAP Dependence Plots (Top 4 Features)',
             fontsize=16, fontweight='bold', color=ACCENT)
axes = axes.flatten()

top4_feats = mean_shap.head(4).index.tolist()

for i, feat in enumerate(top4_feats):
    feat_idx = feature_names.index(feat)
    sv = shap_values[:, feat_idx]
    fv = X_sample[feat].values

    # Color by default status
    colors_dep = [DANGER if y == 1 else SUCCESS for y in y_sample]
    axes[i].scatter(fv, sv, c=colors_dep, alpha=0.5, s=15)
    axes[i].axhline(y=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

    # Trend line
    z = np.polyfit(fv, sv, 1)
    p = np.poly1d(z)
    x_line = np.linspace(fv.min(), fv.max(), 100)
    axes[i].plot(x_line, p(x_line), color=WARNING, linewidth=2, alpha=0.8)

    axes[i].set_title(f'{feat}\nSHAP Dependence (Red=Default, Green=Non-Default)',
                      color=ACCENT, fontsize=10)
    axes[i].set_xlabel(feat, fontsize=9)
    axes[i].set_ylabel('SHAP Value', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/19_shap_dependence_plots.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 19 saved: outputs/19_shap_dependence_plots.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 20 — SHAP Default vs Non-Default Comparison
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Chart 20 — SHAP: Default vs Non-Default Profiles',
             fontsize=16, fontweight='bold', color=ACCENT)

top10 = mean_shap.head(10).index.tolist()

# Mean SHAP for defaults
shap_default    = shap_values[y_sample == 1]
shap_nondefault = shap_values[y_sample == 0]

mean_shap_default    = pd.Series(shap_default.mean(axis=0),    index=feature_names)[top10]
mean_shap_nondefault = pd.Series(shap_nondefault.mean(axis=0), index=feature_names)[top10]

x   = np.arange(len(top10))
w   = 0.35
b1  = axes[0].barh(x - w/2, mean_shap_default.values,    w,
                   label='Default',     color=DANGER,  edgecolor='#0a0a0a', alpha=0.85)
b2  = axes[0].barh(x + w/2, mean_shap_nondefault.values, w,
                   label='Non-Default', color=SUCCESS, edgecolor='#0a0a0a', alpha=0.85)
axes[0].set_yticks(x)
axes[0].set_yticklabels(top10, fontsize=9)
axes[0].axvline(x=0, color='white', linewidth=0.8, alpha=0.5)
axes[0].set_title('Mean SHAP Values by Default Status', color=ACCENT, fontsize=12)
axes[0].set_xlabel('Mean SHAP Value')
axes[0].legend()

# SHAP value spread per feature
shap_std_default    = pd.Series(np.abs(shap_default).mean(axis=0),    index=feature_names)[top10]
shap_std_nondefault = pd.Series(np.abs(shap_nondefault).mean(axis=0), index=feature_names)[top10]

axes[1].barh(x - w/2, shap_std_default.values,    w,
             label='Default',     color=DANGER,  edgecolor='#0a0a0a', alpha=0.85)
axes[1].barh(x + w/2, shap_std_nondefault.values, w,
             label='Non-Default', color=SUCCESS, edgecolor='#0a0a0a', alpha=0.85)
axes[1].set_yticks(x)
axes[1].set_yticklabels(top10, fontsize=9)
axes[1].set_title('Mean |SHAP| Magnitude by Default Status', color=ACCENT, fontsize=12)
axes[1].set_xlabel('Mean |SHAP Value|')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/20_shap_default_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 20 saved: outputs/20_shap_default_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 21 — Individual Prediction Explanations
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Chart 21 — Individual Loan Application Explanations',
             fontsize=16, fontweight='bold', color=ACCENT)
axes = axes.flatten()

# Pick 2 high-risk and 2 low-risk examples
probs = model.predict_proba(X_sample)[:, 1]
high_risk_idx = np.argsort(probs)[-2:][::-1]
low_risk_idx  = np.argsort(probs)[:2]
examples = list(high_risk_idx) + list(low_risk_idx)
labels   = ['High Risk #1', 'High Risk #2', 'Low Risk #1', 'Low Risk #2']
colors_e = [DANGER, DANGER, SUCCESS, SUCCESS]

for i, (idx, label, color) in enumerate(zip(examples, labels, colors_e)):
    sv   = shap_values[idx]
    top8 = np.argsort(np.abs(sv))[-8:]
    feat_labels = [feature_names[j] for j in top8]
    feat_shap   = sv[top8]
    bar_colors  = [DANGER if v > 0 else SUCCESS for v in feat_shap]

    axes[i].barh(feat_labels, feat_shap, color=bar_colors, edgecolor='#0a0a0a')
    axes[i].axvline(x=0, color='white', linewidth=1, alpha=0.7)
    prob = probs[idx]
    actual = 'DEFAULT' if y_sample.iloc[idx] == 1 else 'NON-DEFAULT'
    axes[i].set_title(f'{label} | P(default)={prob:.1%} | Actual: {actual}',
                      color=color, fontsize=11)
    axes[i].set_xlabel('SHAP Value (Red=↑Risk, Green=↓Risk)')

plt.tight_layout()
plt.savefig('outputs/21_individual_explanations.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 21 saved: outputs/21_individual_explanations.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE SHAP DATA FOR WEB APP
# ═════════════════════════════════════════════════════════════════════════════
shap_summary = {
    'top_features'       : mean_shap.head(15).index.tolist(),
    'top_shap_values'    : mean_shap.head(15).values.tolist(),
    'expected_value'     : float(expected_value),
    'feature_names'      : feature_names
}
with open('processed_data/shap_summary.json', 'w') as f:
    json.dump(shap_summary, f, indent=2)
print("\n✅ processed_data/shap_summary.json saved (for web app)")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  SHAP EXPLAINABILITY COMPLETE — KEY FINDINGS")
print(f"{'='*65}")
print(f"\n  Top 5 Risk Drivers (SHAP):")
for i, (feat, val) in enumerate(mean_shap.head(5).items(), 1):
    tag = '⭐ (engineered)' if feat in [
        'payment_to_income', 'financial_stress_index', 'dti_ratio',
        'grade_rate_interaction', 'grade_risk_score', 'credit_risk_signal',
        'monthly_payment', 'total_interest_burden'
    ] else ''
    print(f"  {i}. {feat:<35} {val:.4f} {tag}")

print(f"\n  Charts saved → outputs/ (18-21)")
print(f"  ➡  Next: Run credit_risk_part7_risk_scoring.py")
print(f"{'='*65}\n")