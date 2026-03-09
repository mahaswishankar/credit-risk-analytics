# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - PART 5: NEURAL NETWORK (DEEP LEARNING)
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.optimizers import Adam
import pickle
import json
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
os.makedirs('models',  exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CREDIT RISK ANALYTICS ENGINE — PART 5: NEURAL NETWORK")
print("=" * 65)

X_train = pd.read_csv('processed_data/X_train_engineered.csv').values
X_test  = pd.read_csv('processed_data/X_test_engineered.csv').values
y_train = pd.read_csv('processed_data/y_train_engineered.csv').squeeze().values
y_test  = pd.read_csv('processed_data/y_test_engineered.csv').squeeze().values

with open('processed_data/all_features.json') as f:
    feature_names = json.load(f)

n_features = X_train.shape[1]
print(f"\n✅ Training set : {X_train.shape[0]:,} rows × {n_features} features")
print(f"✅ Test set     : {X_test.shape[0]:,} rows × {n_features} features")
print(f"✅ TensorFlow   : {tf.__version__}")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 — BASELINE NEURAL NETWORK
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  MODEL 1: BASELINE NEURAL NETWORK")
print(f"{'─'*65}")

def build_baseline(n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='baseline_nn')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

baseline_model = build_baseline(n_features)
baseline_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor='val_auc', patience=10, restore_best_weights=True, mode='max'
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
)

print("\n  Training baseline model...")
history_baseline = baseline_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=512,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

y_prob_baseline = baseline_model.predict(X_test, verbose=0).flatten()
y_pred_baseline = (y_prob_baseline >= 0.5).astype(int)

auc_baseline = roc_auc_score(y_test, y_prob_baseline)
f1_baseline  = f1_score(y_test, y_pred_baseline)
print(f"\n  Baseline NN — AUC-ROC: {auc_baseline:.4f} | F1: {f1_baseline:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 — DEEP NEURAL NETWORK WITH REGULARIZATION
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  MODEL 2: DEEP NEURAL NETWORK WITH REGULARIZATION")
print(f"{'─'*65}")

def build_deep_nn(n_features):
    inputs = keras.Input(shape=(n_features,), name='input')

    # Block 1
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),
                     name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Activation('relu', name='relu_1')(x)
    x = layers.Dropout(0.4, name='drop_1')(x)

    # Block 2
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001),
                     name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Activation('relu', name='relu_2')(x)
    x = layers.Dropout(0.3, name='drop_2')(x)

    # Block 3
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001),
                     name='dense_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.Activation('relu', name='relu_3')(x)
    x = layers.Dropout(0.2, name='drop_3')(x)

    # Block 4
    x = layers.Dense(32, name='dense_4')(x)
    x = layers.Activation('relu', name='relu_4')(x)

    # Output
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=output, name='deep_nn')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

deep_model = build_deep_nn(n_features)
deep_model.summary()

early_stop2 = callbacks.EarlyStopping(
    monitor='val_auc', patience=15, restore_best_weights=True, mode='max'
)
reduce_lr2 = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=0
)
checkpoint = callbacks.ModelCheckpoint(
    'models/best_nn.keras', monitor='val_auc',
    save_best_only=True, mode='max', verbose=0
)

print("\n  Training deep neural network...")
history_deep = deep_model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=256,
    validation_split=0.15,
    callbacks=[early_stop2, reduce_lr2, checkpoint],
    verbose=1
)

y_prob_deep = deep_model.predict(X_test, verbose=0).flatten()
y_pred_deep = (y_prob_deep >= 0.5).astype(int)

auc_deep  = roc_auc_score(y_test, y_prob_deep)
f1_deep   = f1_score(y_test, y_pred_deep)
prec_deep = precision_score(y_test, y_pred_deep)
rec_deep  = recall_score(y_test, y_pred_deep)
acc_deep  = accuracy_score(y_test, y_pred_deep)

print(f"\n  Deep NN — AUC-ROC: {auc_deep:.4f} | F1: {f1_deep:.4f}")
print(f"           Precision: {prec_deep:.4f} | Recall: {rec_deep:.4f}")
print(f"           Accuracy : {acc_deep:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 16 — Training History
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('Chart 16 — Neural Network Training History',
             fontsize=16, fontweight='bold', color=ACCENT)

# Baseline loss
axes[0,0].plot(history_baseline.history['loss'],
               color=ACCENT, linewidth=2, label='Train Loss')
axes[0,0].plot(history_baseline.history['val_loss'],
               color=DANGER, linewidth=2, label='Val Loss')
axes[0,0].set_title('Baseline NN — Loss', color=ACCENT)
axes[0,0].set_xlabel('Epoch')
axes[0,0].legend()

# Baseline AUC
axes[0,1].plot(history_baseline.history['auc'],
               color=SUCCESS, linewidth=2, label='Train AUC')
axes[0,1].plot(history_baseline.history['val_auc'],
               color=WARNING, linewidth=2, label='Val AUC')
axes[0,1].set_title('Baseline NN — AUC', color=ACCENT)
axes[0,1].set_xlabel('Epoch')
axes[0,1].legend()

# Baseline accuracy
axes[0,2].plot(history_baseline.history['accuracy'],
               color=PURPLE, linewidth=2, label='Train Acc')
axes[0,2].plot(history_baseline.history['val_accuracy'],
               color=WARNING, linewidth=2, label='Val Acc')
axes[0,2].set_title('Baseline NN — Accuracy', color=ACCENT)
axes[0,2].set_xlabel('Epoch')
axes[0,2].legend()

# Deep NN loss
axes[1,0].plot(history_deep.history['loss'],
               color=ACCENT, linewidth=2, label='Train Loss')
axes[1,0].plot(history_deep.history['val_loss'],
               color=DANGER, linewidth=2, label='Val Loss')
axes[1,0].set_title('Deep NN — Loss', color=ACCENT)
axes[1,0].set_xlabel('Epoch')
axes[1,0].legend()

# Deep NN AUC
axes[1,1].plot(history_deep.history['auc'],
               color=SUCCESS, linewidth=2, label='Train AUC')
axes[1,1].plot(history_deep.history['val_auc'],
               color=WARNING, linewidth=2, label='Val AUC')
axes[1,1].set_title('Deep NN — AUC', color=ACCENT)
axes[1,1].set_xlabel('Epoch')
axes[1,1].legend()

# Deep NN precision & recall
axes[1,2].plot(history_deep.history['precision'],
               color=SUCCESS, linewidth=2, label='Train Precision')
axes[1,2].plot(history_deep.history['recall'],
               color=DANGER, linewidth=2, label='Train Recall')
axes[1,2].plot(history_deep.history['val_precision'],
               color=SUCCESS, linewidth=2, linestyle='--', label='Val Precision')
axes[1,2].plot(history_deep.history['val_recall'],
               color=DANGER, linewidth=2, linestyle='--', label='Val Recall')
axes[1,2].set_title('Deep NN — Precision & Recall', color=ACCENT)
axes[1,2].set_xlabel('Epoch')
axes[1,2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/16_nn_training_history.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("\n✅ Chart 16 saved: outputs/16_nn_training_history.png")

# ═════════════════════════════════════════════════════════════════════════════
# CHART 17 — NN vs XGBoost Comparison
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("  COMPARING NEURAL NETWORK vs XGBOOST")
print(f"{'─'*65}")

with open('models/XGBoost.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb_model.predict(X_test)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Chart 17 — Neural Network vs XGBoost',
             fontsize=16, fontweight='bold', color=ACCENT)

# ROC comparison
fpr_nn,  tpr_nn,  _ = roc_curve(y_test, y_prob_deep)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

axes[0].plot(fpr_nn,  tpr_nn,  color=ACCENT,  linewidth=2.5,
             label=f'Deep NN  (AUC={auc_deep:.3f})')
axes[0].plot(fpr_xgb, tpr_xgb, color=WARNING, linewidth=2.5,
             label=f'XGBoost  (AUC={auc_xgb:.3f})')
axes[0].plot([0,1],[0,1], 'white', linewidth=1, linestyle='--', alpha=0.4)
axes[0].set_title('ROC Curves', color=ACCENT, fontsize=13)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=11)

# Metric comparison
metrics  = ['AUC-ROC', 'F1', 'Precision', 'Recall', 'Accuracy']
nn_vals  = [auc_deep, f1_deep, prec_deep, rec_deep, acc_deep]
xgb_vals = [auc_xgb,
             f1_score(y_test, y_pred_xgb),
             precision_score(y_test, y_pred_xgb),
             recall_score(y_test, y_pred_xgb),
             accuracy_score(y_test, y_pred_xgb)]

x    = np.arange(len(metrics))
w    = 0.35
bars1 = axes[1].bar(x - w/2, nn_vals,  w, label='Deep NN',
                    color=ACCENT,   edgecolor='#0a0a0a', alpha=0.85)
bars2 = axes[1].bar(x + w/2, xgb_vals, w, label='XGBoost',
                    color=WARNING,  edgecolor='#0a0a0a', alpha=0.85)
for bar, val in zip(list(bars1) + list(bars2), nn_vals + xgb_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f'{val:.2f}', ha='center', va='bottom',
                 color='white', fontsize=8)
axes[1].set_title('Metric Comparison', color=ACCENT, fontsize=13)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics, rotation=20)
axes[1].set_ylim(0, 1.15)
axes[1].legend()

# Confusion matrix — Deep NN
cm_nn  = confusion_matrix(y_test, y_pred_deep)
cm_pct = cm_nn.astype(float) / cm_nn.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=axes[2],
            cmap='RdYlGn', vmin=0, vmax=100,
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'],
            linewidths=1, linecolor='#0a0a0a',
            annot_kws={'size': 12, 'weight': 'bold'})
axes[2].set_title(f'Deep NN Confusion Matrix\nAUC={auc_deep:.3f}',
                  color=ACCENT, fontsize=12)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/17_nn_vs_xgboost.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("✅ Chart 17 saved: outputs/17_nn_vs_xgboost.png")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE NEURAL NETWORK
# ═════════════════════════════════════════════════════════════════════════════
deep_model.save('models/deep_neural_network.keras')
print("✅ models/deep_neural_network.keras saved")

# Save results for comparison
nn_results = {
    'model'    : 'Deep Neural Network',
    'auc_roc'  : float(auc_deep),
    'f1'       : float(f1_deep),
    'precision': float(prec_deep),
    'recall'   : float(rec_deep),
    'accuracy' : float(acc_deep)
}
with open('processed_data/nn_results.json', 'w') as f:
    json.dump(nn_results, f, indent=2)
print("✅ processed_data/nn_results.json saved")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  NEURAL NETWORK COMPLETE — RESULTS")
print(f"{'='*65}")
print(f"\n  {'Model':<25} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print(f"  {'─'*60}")
print(f"  {'Baseline NN':<25} {auc_baseline:>8.4f} {f1_baseline:>8.4f}")
print(f"  {'Deep NN':<25} {auc_deep:>8.4f} {f1_deep:>8.4f} "
      f"{prec_deep:>10.4f} {rec_deep:>8.4f}")
print(f"  {'XGBoost':<25} {auc_xgb:>8.4f} "
      f"{f1_score(y_test,y_pred_xgb):>8.4f} "
      f"{precision_score(y_test,y_pred_xgb):>10.4f} "
      f"{recall_score(y_test,y_pred_xgb):>8.4f}")

winner = 'Deep NN' if auc_deep > auc_xgb else 'XGBoost'
print(f"\n  🏆 Overall Winner: {winner}")
print(f"\n  Charts saved → outputs/ (16-17)")
print(f"  ➡  Next: Run credit_risk_part6_shap.py")
print(f"{'='*65}\n")