# =============================================================================
# CREDIT RISK ANALYTICS ENGINE - FLASK BACKEND API
# JP Morgan Chase & Co. - Risk Analytics Portfolio Project
# =============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import shap
import json
import os

app = Flask(__name__)
CORS(app)

# ── Load Model & Config ───────────────────────────────────────────────────────
print("Loading model and configuration...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'models', 'XGBoost.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'processed_data', 'all_features.json')) as f:
    feature_names = json.load(f)

with open(os.path.join(BASE_DIR, 'processed_data', 'scoring_config.json')) as f:
    scoring_config = json.load(f)

with open(os.path.join(BASE_DIR, 'processed_data', 'shap_summary.json')) as f:
    shap_summary = json.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)

print("✅ Model loaded successfully!")
print(f"✅ Features: {len(feature_names)}")

# ── Helper Functions ──────────────────────────────────────────────────────────

def assign_grade(prob):
    if prob < 0.05:   return 'A'
    elif prob < 0.15: return 'B'
    elif prob < 0.30: return 'C'
    elif prob < 0.50: return 'D'
    else:             return 'E'

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

def calculate_credit_score(prob):
    score = int(850 - (prob * 550))
    return max(300, min(850, score))

def engineer_features(raw):
    """Recreate all engineered features from raw input"""
    f = {}

    # Original features
    f['person_age']                    = raw['person_age']
    f['person_income']                 = raw['person_income']
    f['person_emp_length']             = raw['person_emp_length']
    f['loan_amnt']                     = raw['loan_amnt']
    f['loan_int_rate']                 = raw['loan_int_rate']
    f['loan_percent_income']           = raw['loan_amnt'] / (raw['person_income'] + 1)
    f['cb_person_cred_hist_length']    = raw['cb_person_cred_hist_length']
    f['loan_grade_encoded']            = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}.get(raw['loan_grade'], 4)
    f['cb_default_encoded']            = 1 if raw['cb_person_default_on_file'] == 'Y' else 0

    # One-hot: home ownership
    for cat in ['MORTGAGE', 'OTHER', 'OWN', 'RENT']:
        f[f'person_home_ownership_{cat}'] = 1 if raw['person_home_ownership'] == cat else 0

    # One-hot: loan intent
    for cat in ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
        f[f'loan_intent_{cat}'] = 1 if raw['loan_intent'].upper().replace(' ','') == cat else 0

    # Engineered features
    f['dti_ratio']                  = raw['loan_amnt'] / (raw['person_income'] + 1)
    f['monthly_payment']            = raw['loan_amnt'] * (raw['loan_int_rate'] / 100) / 12
    f['payment_to_income']          = f['monthly_payment'] / (raw['person_income'] / 12 + 1)
    f['total_interest_burden']      = raw['loan_amnt'] * (raw['loan_int_rate'] / 100) * 3
    f['financial_stress_index']     = (f['dti_ratio'] * 0.4 +
                                       f['payment_to_income'] * 0.4 +
                                       f['loan_percent_income'] * 0.2)
    f['rate_premium']               = raw['loan_int_rate'] - 5.42
    f['grade_risk_score']           = f['loan_grade_encoded'] / 7.0
    f['credit_risk_signal']         = (f['grade_risk_score'] * 0.5 +
                                       f['cb_default_encoded'] * 0.3 +
                                       (raw['loan_int_rate'] / 23.22) * 0.2)
    f['high_interest_flag']         = 1 if raw['loan_int_rate'] > 13.5 else 0
    f['income_per_emp_year']        = raw['person_income'] / (raw['person_emp_length'] + 1)
    f['is_young_borrower']          = 1 if raw['person_age'] < 25 else 0
    f['credit_history_ratio']       = raw['cb_person_cred_hist_length'] / raw['person_age']
    f['income_stability']           = np.log1p(raw['person_income']) * raw['person_emp_length'] / 100
    f['loan_per_age']               = raw['loan_amnt'] / raw['person_age']
    f['grade_rate_interaction']     = f['loan_grade_encoded'] * raw['loan_int_rate']
    f['income_credit_interaction']  = np.log1p(raw['person_income']) * raw['cb_person_cred_hist_length']
    f['debt_default_interaction']   = f['dti_ratio'] * f['cb_default_encoded']

    # Binned features
    loan_amnt = raw['loan_amnt']
    f['loan_size_bin'] = 1 if loan_amnt <= 5000 else 2 if loan_amnt <= 10000 else 3 if loan_amnt <= 20000 else 4

    income = raw['person_income']
    f['income_bin'] = 1 if income <= 30000 else 2 if income <= 60000 else 3 if income <= 100000 else 4

    rate = raw['loan_int_rate']
    f['rate_bin'] = 1 if rate <= 8 else 2 if rate <= 12 else 3 if rate <= 16 else 4

    return f

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'XGBoost', 'features': len(feature_names)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Engineer all features
        features = engineer_features(data)

        # Build feature vector in correct order
        X = pd.DataFrame([features])[feature_names]

        # Predict
        prob      = float(model.predict_proba(X)[0][1])
        grade     = assign_grade(prob)
        decision  = assign_decision(prob)
        risk_level = assign_risk_level(prob)
        score     = calculate_credit_score(prob)

        # SHAP explanation
        shap_vals = explainer.shap_values(X)[0]
        shap_dict = dict(zip(feature_names, shap_vals.tolist()))

        # Top 8 risk factors
        top_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        risk_factors = [
            {
                'feature' : feat,
                'shap'    : round(val, 4),
                'impact'  : 'increases risk' if val > 0 else 'decreases risk',
                'value'   : round(float(features.get(feat, 0)), 4)
            }
            for feat, val in top_factors
        ]

        # Grade descriptions
        grade_desc = {
            'A': 'Excellent credit profile. Very low default risk.',
            'B': 'Good credit profile. Low default risk.',
            'C': 'Fair credit profile. Moderate default risk. Manual review recommended.',
            'D': 'Poor credit profile. High default risk. Likely rejection.',
            'E': 'Very poor credit profile. Very high default risk. Reject.'
        }

        response = {
            'default_probability' : round(prob * 100, 2),
            'credit_score'        : score,
            'grade'               : grade,
            'risk_level'          : risk_level,
            'decision'            : decision,
            'grade_description'   : grade_desc[grade],
            'risk_factors'        : risk_factors,
            'key_metrics'         : {
                'dti_ratio'            : round(features['dti_ratio'] * 100, 2),
                'monthly_payment'      : round(features['monthly_payment'], 2),
                'payment_to_income'    : round(features['payment_to_income'] * 100, 2),
                'financial_stress'     : round(features['financial_stress_index'], 4),
                'grade_risk_score'     : round(features['grade_risk_score'] * 100, 2),
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        'model_performance': {
            'auc_roc'  : 0.9348,
            'f1_score' : 0.8191,
            'precision': 0.9604,
            'recall'   : 0.7141,
            'accuracy' : 0.9318
        },
        'dataset': {
            'total_records'  : 32581,
            'default_rate'   : 21.8,
            'features_total' : 39,
            'features_engineered': 20
        },
        'top_features': shap_summary['top_features'][:5]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
