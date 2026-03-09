# 🏦 Credit Risk Analytics Engine
### JP Morgan Chase & Co. — Data Science Portfolio Project

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.9348-orange?style=for-the-badge)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow)
![Deployed](https://img.shields.io/badge/Deployed-Live-brightgreen?style=for-the-badge)

> A production-grade credit risk assessment system that predicts loan default probability using machine learning, featuring real-time SHAP explainability and a JP Morgan-styled web interface.

## 🌐 Live Demo
**[credit-risk-analytics.vercel.app](https://credit-risk-analytics.vercel.app)**

Backend API: [credit-risk-analytics.onrender.com/health](https://credit-risk-analytics.onrender.com/health)

> ⚠️ Note: Backend runs on Render free tier — first request may take 30-50 seconds to wake up.

---

## 📊 Model Performance

| Model | AUC-ROC | F1 Score | Precision | Recall | Accuracy |
|-------|---------|----------|-----------|--------|----------|
| Logistic Regression | 0.8774 | 0.6377 | 0.5347 | 0.7900 | 0.8058 |
| Random Forest | 0.9190 | 0.8074 | 0.9376 | 0.7089 | 0.9268 |
| Gradient Boosting | 0.9282 | 0.8077 | 0.9359 | 0.7104 | 0.9268 |
| Deep Neural Network | 0.8925 | 0.7012 | 0.6944 | 0.7082 | 0.8694 |
| **XGBoost** ✅ | **0.9348** | **0.8191** | **0.9604** | **0.7141** | **0.9318** |

---

## 🎯 Credit Scoring System

| Grade | Default Probability | Credit Score | Decision | Actual Default Rate |
|-------|-------------------|--------------|----------|-------------------|
| A | < 5% | 770–850 | ✅ APPROVE | 1.5% |
| B | 5–15% | 685–770 | ✅ APPROVE | 6.9% |
| C | 15–30% | 535–685 | ⚠️ REVIEW | 18.0% |
| D | 30–50% | 300–535 | ❌ REJECT | 28.4% |
| E | > 50% | < 300 | ❌ REJECT | 96.0% |

---

## 🏗️ Project Architecture

```
credit-risk-analytics/
├── 📊 credit_risk_part1_eda.py           # Exploratory Data Analysis (7 charts)
├── 🔧 credit_risk_part2_preprocessing.py  # Data Cleaning + SMOTE
├── ⚙️  credit_risk_part3_feature_engineering.py  # 20 engineered features
├── 🤖 credit_risk_part4_models.py         # 4 ML models comparison
├── 🧠 credit_risk_part5_neural_network.py # Deep Neural Network
├── 🔍 credit_risk_part6_shap.py           # SHAP Explainability (4 charts)
├── 📋 credit_risk_part7_risk_scoring.py   # Credit Scoring System
├── backend/
│   ├── app.py                             # Flask REST API
│   ├── train_on_startup.py                # Auto-training on Render
│   └── requirements.txt
├── frontend/
│   └── credit-risk-app/                   # React Web Application
├── outputs/                               # 23 generated charts
└── data/
    └── credit_risk_dataset.csv            # 32,581 loan records
```

---

## 🔬 Technical Pipeline

### 1️⃣ Data Preprocessing
- **Dataset**: 32,581 real loan applications from Kaggle
- **Outlier removal**: Age > 80 (7 rows), Employment > 60 years (2 rows), Income > 99th percentile (317 rows)
- **Missing value imputation**: Interest rate → median by loan grade, Employment length → global median
- **Class imbalance**: 21.8% default rate handled with **SMOTE** (3.6:1 → 1:1 ratio)
- **Scaling**: RobustScaler (handles outliers better than StandardScaler)

### 2️⃣ Feature Engineering (20 New Features)
```python
# Financial Stress Features
dti_ratio             = loan_amount / income
payment_to_income     = monthly_payment / monthly_income   # Top SHAP feature
financial_stress_index = dti * 0.4 + payment_to_income * 0.4 + pct_income * 0.2

# Credit Risk Features  
grade_risk_score      = loan_grade_encoded / 7.0
credit_risk_signal    = grade_risk * 0.5 + default_history * 0.3 + rate_norm * 0.2

# Interaction Features
grade_rate_interaction = loan_grade * interest_rate       # 4th highest SHAP
income_stability       = log(income) * employment_years / 100
```

### 3️⃣ Top 5 Risk Drivers (SHAP Analysis)
| Rank | Feature | Mean |SHAP| | Type |
|------|---------|----------|------|
| 1 | loan_grade_encoded | 0.7973 | Original |
| 2 | person_income | 0.5704 | Original |
| 3 | dti_ratio | 0.5545 | ⭐ Engineered |
| 4 | payment_to_income | 0.4369 | ⭐ Engineered |
| 5 | person_home_ownership_RENT | 0.3919 | Original |

### 4️⃣ Why XGBoost Beats Neural Networks on Tabular Data
> XGBoost achieved **0.9348 AUC-ROC** vs **0.8925** for the Deep Neural Network — a 4.2% improvement. This confirms the industry consensus that gradient boosting outperforms deep learning on structured tabular financial data, while neural networks excel on images and text. This is why JP Morgan's quantitative teams rely heavily on gradient boosting for credit risk models.

---

## 🌐 Web Application

### Architecture
```
React Frontend (Vercel) ──→ Flask API (Render) ──→ XGBoost Model
                                    ↓
                              SHAP Explainer
                                    ↓
                         Risk Grade + Decision
```

### API Endpoints
```
GET  /health    → Model status and feature count
POST /predict   → Full credit risk assessment
GET  /stats     → Model performance metrics
```

### Sample API Request
```json
POST /predict
{
  "person_age": 28,
  "person_income": 55000,
  "person_emp_length": 4,
  "person_home_ownership": "RENT",
  "loan_amnt": 10000,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_int_rate": 11.5,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 3
}
```

### Sample API Response
```json
{
  "default_probability": 34.5,
  "credit_score": 660,
  "grade": "D",
  "decision": "REVIEW",
  "risk_level": "High",
  "risk_factors": [
    {"feature": "loan_grade_encoded", "shap": 1.512, "impact": "increases risk"},
    {"feature": "person_income", "shap": -0.716, "impact": "decreases risk"}
  ],
  "key_metrics": {
    "dti_ratio": 18.18,
    "monthly_payment": 96,
    "payment_to_income": 2.09
  }
}
```

---

## 🚀 Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
# Flask running on http://127.0.0.1:5000
```

### Frontend
```bash
cd frontend/credit-risk-app
npm install
npm start
# React running on http://localhost:3000
```

---

## 📈 Visualizations (23 Charts)

| Chart | Description |
|-------|-------------|
| 01–07 | Exploratory Data Analysis |
| 08 | Preprocessing Summary |
| 09–10 | Feature Engineering Correlations |
| 11 | Model Comparison |
| 12 | Confusion Matrices |
| 13 | ROC & PR Curves |
| 14 | Feature Importance |
| 15 | Cross Validation |
| 16 | Neural Network Training History |
| 17 | Neural Network vs XGBoost |
| 18 | SHAP Global Importance |
| 19 | SHAP Dependence Plots |
| 20 | SHAP Default vs Non-Default |
| 21 | Individual Loan Explanations |
| 22 | Risk Score Distribution |
| 23 | Credit Risk Scorecard |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| ML/AI | XGBoost, Scikit-learn, TensorFlow/Keras, SHAP |
| Data | Pandas, NumPy, imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn |
| Backend | Flask, Flask-CORS, Gunicorn |
| Frontend | React 18, Axios, CSS3 |
| Deployment | Render (Backend), Vercel (Frontend) |
| Version Control | Git, GitHub |

---

## 📁 Dataset

- **Source**: [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Size**: 32,581 loan applications
- **Features**: 12 original → 39 after engineering
- **Default Rate**: 21.8% (7,108 defaults)
- **Class Imbalance**: 3.6:1 (handled with SMOTE)

---

## 👨‍💻 Author

**Mahaswi Shankar**
B.Tech CSE — 3rd Year (Graduating 2027)

[![GitHub](https://img.shields.io/badge/GitHub-mahaswishankar-black?style=flat&logo=github)](https://github.com/mahaswishankar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mahaswishankar1-blue?style=flat&logo=linkedin)](https://linkedin.com/in/mahaswishankar1)
[![Email](https://img.shields.io/badge/Email-mahaswiwork1%40gmail.com-red?style=flat&logo=gmail)](mailto:mahaswiwork1@gmail.com)

---

*Built as part of a JP Morgan Chase & Co. Data Science Portfolio*
