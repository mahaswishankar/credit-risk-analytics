# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 4: MACHINE LEARNING MODELS
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import pickle
import json
import warnings
import os
import time

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
COLORS  = [ACCENT, SUCCESS, DANGER, WARNING, PURPLE]

os.makedirs('outputs', exist_ok=True)
os.makedirs('models',  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 4: ML MODELS")
print("=" * 65)

X_train = pd.read_csv('processed_data/X_train_engineered.csv')
X_test  = pd.read_csv('processed_data/X_test_engineered.csv')
y_train = pd.read_csv('processed_data/y_train_engineered.csv').squeeze()
y_test  = pd.read_csv('processed_data/y_test_engineered.csv').squeeze()

with open('processed_data/all_features.json') as f:
    feature_names = json.load(f)

print(f"\n✅ Training set : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
print(f"✅ Test set     : {X_test.shape[0]:,} rows × {X_test.shape[1]} features")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=0.1,
        random_state=42,
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
}

# ═════════════════════════════════════════════════════════════════════════════
# TRAIN & EVALUATE ALL MODELS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  TRAINING & EVALUATING MODELS")
print(f"{'─'*65}")

results   = {}
trained   = {}
cv_scores = {}

for name, model in models.items():
    print(f"\n  [{name}]")
    start = time.time()

    # Train
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time : {train_time:.1f}s")

    # Predict
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc_roc  = roc_auc_score(y_test, y_pred_prob)
    auc_pr   = average_precision_score(y_test, y_pred_prob)
    f1       = f1_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred)
    rec      = recall_score(y_test, y_pred)
    acc      = accuracy_score(y_test, y_pred)

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_test, y_test,
                              cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = cv_auc

    results[name] = {
        'AUC-ROC'  : auc_roc,
        'AUC-PR'   : auc_pr,
        'F1'       : f1,
        'Precision': prec,
        'Recall'   : rec,
        'Accuracy' : acc,
        'y_pred'   : y_pred,
        'y_prob'   : y_pred_prob
    }
    trained[name] = model

    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  CV AUC    : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  MODEL COMPARISON TABLE")
print(f"{'─'*65}")
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items()
           if k not in ['y_pred', 'y_prob']}
    for name, res in results.items()
}).T.round(4)
print(metrics_df.to_string())

# Best model
best_model_name = metrics_df['AUC-ROC'].idxmax()
print(f"\n  🏆 Best Model: {best_model_name}")
print(f"     AUC-ROC  : {metrics_df.loc[best_model_name, 'AUC-ROC']:.4f}")
print(f"     F1 Score : {metrics_df.loc[best_model_name, 'F1']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 11 — Model Comparison Bar Chart
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Chart 11 — Model Performance Comparison',
             fontsize=16, fontweight='bold', color=ACCENT)
axes = axes.flatten()

metric_list = ['AUC-ROC', 'AUC-PR', 'F1', 'Precision', 'Recall', 'Accuracy']
model_names = list(results.keys())
short_names = ['LR', 'RF', 'XGB', 'GBM']

for i, metric in enumerate(metric_list):
    vals   = [results[m][metric] for m in model_names]
    colors = [SUCCESS if v == max(vals) else ACCENT for v in vals]
    bars   = axes[i].bar(short_names, vals, color=colors, edgecolor='#0a0a0a')
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom',
                     color='white', fontsize=10, fontweight='bold')
    axes[i].set_title(metric, color=ACCENT, fontsize=13)
    axes[i].set_ylim(0, 1.1)
    axes[i].set_ylabel(metric)
    best_idx = vals.index(max(vals))
    axes[i].get_children()[best_idx].set_edgecolor(WARNING)
    axes[i].get_children()[best_idx].set_linewidth(2)

plt.tight_layout()
plt.savefig('outputs/11_model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 11 saved: outputs/11_model_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 12 — Confusion Matrices
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Chart 12 — Confusion Matrices (All Models)',
             fontsize=16, fontweight='bold', color=ACCENT)

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=axes[i],
                cmap='RdYlGn', vmin=0, vmax=100,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'],
                linewidths=1, linecolor='#0a0a0a',
                annot_kws={'size': 11, 'weight': 'bold'})
    short = ['LR', 'RF', 'XGB', 'GBM'][i]
    axes[i].set_title(f'{short}\nAUC={res["AUC-ROC"]:.3f}',
                      color=ACCENT, fontsize=12)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/12_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 12 saved: outputs/12_confusion_matrices.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 13 — ROC Curves
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Chart 13 — ROC & Precision-Recall Curves',
             fontsize=16, fontweight='bold', color=ACCENT)

for i, (name, res) in enumerate(results.items()):
    color = COLORS[i]
    # ROC
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{['LR','RF','XGB','GBM'][i]} (AUC={res['AUC-ROC']:.3f})")
    # PR
    prec_c, rec_c, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec_c, prec_c, color=color, linewidth=2,
                 label=f"{['LR','RF','XGB','GBM'][i]} (AP={res['AUC-PR']:.3f})")

# ROC diagonal
axes[0].plot([0,1],[0,1], 'white', linewidth=1, linestyle='--', alpha=0.4)
axes[0].set_title('ROC Curves', color=ACCENT, fontsize=13)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=10)
axes[0].fill_between([0,1],[0,1], alpha=0.05, color='white')

# PR baseline
baseline = y_test.mean()
axes[1].axhline(y=baseline, color='white', linestyle='--',
                alpha=0.4, label=f'Baseline ({baseline:.2f})')
axes[1].set_title('Precision-Recall Curves', color=ACCENT, fontsize=13)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('outputs/13_roc_pr_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 13 saved: outputs/13_roc_pr_curves.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 14 — Feature Importance (Tree Models)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('Chart 14 — Feature Importance (Tree Models)',
             fontsize=16, fontweight='bold', color=ACCENT)

tree_models = {
    'Random Forest'     : trained['Random Forest'],
    'XGBoost'           : trained['XGBoost'],
    'Gradient Boosting' : trained['Gradient Boosting']
}

for i, (name, model) in enumerate(tree_models.items()):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
    colors = [SUCCESS if 'ratio' in f or 'stress' in f or 'risk' in f or 'interaction' in f
              else ACCENT for f in feat_imp.index]
    feat_imp.plot(kind='barh', ax=axes[i], color=colors, edgecolor='#0a0a0a')
    axes[i].set_title(f'{name}\nTop 15 Features', color=ACCENT, fontsize=11)
    axes[i].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('outputs/14_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 14 saved: outputs/14_feature_importance.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 15 — Cross Validation Scores
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Chart 15 — 5-Fold Cross Validation AUC-ROC',
             fontsize=16, fontweight='bold', color=ACCENT)

positions = range(len(cv_scores))
for i, (name, scores) in enumerate(cv_scores.items()):
    short = ['LR', 'RF', 'XGB', 'GBM'][i]
    ax.boxplot(scores, positions=[i], widths=0.5,
               patch_artist=True,
               boxprops=dict(facecolor=COLORS[i], alpha=0.7),
               medianprops=dict(color='white', linewidth=2),
               whiskerprops=dict(color=COLORS[i]),
               capprops=dict(color=COLORS[i]),
               flierprops=dict(markerfacecolor=COLORS[i]))
    ax.text(i, scores.mean() + 0.002, f'μ={scores.mean():.3f}',
            ha='center', color='white', fontsize=10)

ax.set_xticks(range(len(cv_scores)))
ax.set_xticklabels(['LR', 'RF', 'XGB', 'GBM'])
ax.set_ylabel('AUC-ROC Score')
ax.set_title('Distribution of CV Scores — Lower variance = More stable',
             color='#cccccc', fontsize=11)

plt.tight_layout()
plt.savefig('outputs/15_cross_validation.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 15 saved: outputs/15_cross_validation.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  SAVING MODELS")
print(f"{'─'*65}")

for name, model in trained.items():
    fname = name.replace(' ', '_')
    with open(f'models/{fname}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ models/{fname}.pkl")

# Save best model separately for web app
best_model = trained[best_model_name]
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✅ models/best_model.pkl ({best_model_name})")

# Save results
metrics_df.to_csv('processed_data/model_results.csv')
print("  ✅ processed_data/model_results.csv")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  ML MODELS COMPLETE — FINAL RESULTS")
print(f"{'='*65}")
print(f"\n  {'Model':<25} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print(f"  {'─'*62}")
for name in model_names:
    r = results[name]
    marker = ' 🏆' if name == best_model_name else ''
    print(f"  {name:<25} {r['AUC-ROC']:>8.4f} {r['F1']:>8.4f} "
          f"{r['Precision']:>10.4f} {r['Recall']:>8.4f}{marker}")

print(f"\n  Best model saved → models/best_model.pkl")
print(f"  Charts saved     → outputs/ (11–15)")
print(f"\n  ➡  Next: Run credit_risk_part5_neural_network.py")
print(f"{'='*65}\n")