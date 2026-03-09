import { useState } from "react";
import axios from "axios";
import "./App.css";

const GRADE_COLORS = {
  A: "#2ed573", B: "#7bed9f", C: "#ffa502", D: "#ff6348", E: "#ff4757"
};
const DECISION_COLORS = {
  APPROVE: "#2ed573", REVIEW: "#ffa502", REJECT: "#ff4757"
};
const DECISION_ICONS = {
  APPROVE: "✅", REVIEW: "⚠️", REJECT: "❌"
};

const defaultForm = {
  person_age: 28,
  person_income: 55000,
  person_emp_length: 4,
  person_home_ownership: "RENT",
  loan_amnt: 10000,
  loan_intent: "PERSONAL",
  loan_grade: "B",
  loan_int_rate: 11.5,
  cb_person_default_on_file: "N",
  cb_person_cred_hist_length: 3,
};

// function GaugeChart({ probability }) {
//   const pct   = Math.min(Math.max(probability, 0), 100);
//   const angle = (pct / 100) * 180 - 90;
//   const r     = 80;
//   const cx    = 100, cy = 100;
//   const toRad = (deg) => (deg * Math.PI) / 180;
//   const arcX  = (deg) => cx + r * Math.cos(toRad(deg - 90));
//   const arcY  = (cy) => cy + r * Math.sin(toRad(cy - 90));

//   const startAngle = -90;
//   const endAngle   = 90;
//   const midAngle   = startAngle + (pct / 100) * 180;

//   const describeArc = (start, end) => {
//     const x1 = cx + r * Math.cos(toRad(start));
//     const y1 = cy + r * Math.sin(toRad(start));
//     const x2 = cx + r * Math.cos(toRad(end));
//     const y2 = cy + r * Math.sin(toRad(end));
//     return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
//   };

//   const needleX = cx + (r - 10) * Math.cos(toRad(midAngle));
//   const needleY = cy + (r - 10) * Math.sin(toRad(midAngle));

//   const color = pct < 15 ? "#2ed573" : pct < 35 ? "#ffa502" : "#ff4757";

//   return (
//     <svg viewBox="0 0 200 120" className="gauge-svg">
//       <path d={describeArc(-90, 90)} fill="none" stroke="#1a1a2e" strokeWidth="18" />
//       <path d={describeArc(-90, midAngle)} fill="none" stroke={color} strokeWidth="18"
//             strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
//       <line x1={cx} y1={cy} x2={needleX} y2={needleY}
//             stroke="white" strokeWidth="2.5" strokeLinecap="round" />
//       <circle cx={cx} cy={cy} r="5" fill="white" />
//       <text x={cx} y={cy + 22} textAnchor="middle" fill={color}
//             fontSize="22" fontWeight="bold" fontFamily="monospace">
//         {pct.toFixed(1)}%
//       </text>
//       <text x="22"  y="108" fill="#666" fontSize="9" fontFamily="monospace">LOW</text>
//       <text x="155" y="108" fill="#666" fontSize="9" fontFamily="monospace">HIGH</text>
//     </svg>
//   );
// }

function GaugeChart({ probability }) {
  const pct   = Math.min(Math.max(probability, 0), 100);
  const r     = 80;
  const cx    = 100, cy = 100;
  const toRad = (deg) => (deg * Math.PI) / 180;

  const startAngle = -90;
  const midAngle  = startAngle + (pct / 100) * 180;

  const describeArc = (start, end) => {
    const x1 = cx + r * Math.cos(toRad(start));
    const y1 = cy + r * Math.sin(toRad(start));
    const x2 = cx + r * Math.cos(toRad(end));
    const y2 = cy + r * Math.sin(toRad(end));
    return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
  };

  const needleX = cx + (r - 10) * Math.cos(toRad(midAngle));
  const needleY = cy + (r - 10) * Math.sin(toRad(midAngle));

  const color = pct < 15 ? "#2ed573" : pct < 35 ? "#ffa502" : "#ff4757";

  return (
    <svg viewBox="0 0 200 120" className="gauge-svg">
      <path d={describeArc(-90, 90)} fill="none" stroke="#1a1a2e" strokeWidth="18" />
      <path d={describeArc(-90, midAngle)} fill="none" stroke={color} strokeWidth="18"
            strokeLinecap="round" style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
      <line x1={cx} y1={cy} x2={needleX} y2={needleY}
            stroke="white" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill="white" />
      <text x={cx} y={cy + 22} textAnchor="middle" fill={color}
            fontSize="22" fontWeight="bold" fontFamily="monospace">
        {pct.toFixed(1)}%
      </text>
      <text x="22"  y="108" fill="#666" fontSize="9" fontFamily="monospace">LOW</text>
      <text x="155" y="108" fill="#666" fontSize="9" fontFamily="monospace">HIGH</text>
    </svg>
  );
}

function ShapBar({ factor }) {
  const maxVal = 1.2;
  const pct    = Math.min(Math.abs(factor.shap) / maxVal * 100, 100);
  const isRisk = factor.shap > 0;
  const color  = isRisk ? "#ff4757" : "#2ed573";
  const label  = factor.feature.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());

  return (
    <div className="shap-row">
      <span className="shap-label">{label}</span>
      <div className="shap-bar-track">
        <div className="shap-bar-fill"
          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 6px ${color}` }} />
      </div>
      <span className="shap-val" style={{ color }}>
        {isRisk ? "▲" : "▼"} {Math.abs(factor.shap).toFixed(3)}
      </span>
    </div>
  );
}

function MetricCard({ label, value, unit, color }) {
  return (
    <div className="metric-card">
      <div className="metric-val" style={{ color: color || "#00d4ff" }}>
        {value}{unit}
      </div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

export default function App() {
  const [form,    setForm]    = useState(defaultForm);
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(f => ({
      ...f,
      [name]: ["person_age","person_income","person_emp_length",
               "loan_amnt","loan_int_rate","cb_person_cred_hist_length"]
               .includes(name) ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      // const res = await axios.post("http://127.0.0.1:5000/predict", form);
      const res = await axios.post("https://credit-risk-analytics.onrender.com/predict", form);
      setResult(res.data);
    } catch (err) {
      setError("Failed to connect to backend. Make sure Flask is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  const gradeColor    = result ? GRADE_COLORS[result.grade]    : "#00d4ff";
  const decisionColor = result ? DECISION_COLORS[result.decision] : "#00d4ff";

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-jp">JP</span>
            <div className="logo-text">
              <span className="logo-title">Credit Risk Engine</span>
              <span className="logo-sub">XGBoost · AUC-ROC 0.9348 · 39 Features</span>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge badge-green">● LIVE</span>
            <span className="badge badge-blue">ML POWERED</span>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="grid">

          {/* ── LEFT: Input Form ── */}
          <section className="card form-card">
            <h2 className="card-title">
              <span className="card-title-icon">📋</span> Loan Application
            </h2>

            <div className="form-section-title">👤 Borrower Profile</div>
            <div className="form-grid">
              <div className="form-group">
                <label>Age</label>
                <input type="number" name="person_age"
                       value={form.person_age} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Annual Income ($)</label>
                <input type="number" name="person_income"
                       value={form.person_income} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Employment Length (yrs)</label>
                <input type="number" name="person_emp_length"
                       value={form.person_emp_length} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Home Ownership</label>
                <select name="person_home_ownership"
                        value={form.person_home_ownership} onChange={handleChange}>
                  <option value="RENT">Rent</option>
                  <option value="OWN">Own</option>
                  <option value="MORTGAGE">Mortgage</option>
                  <option value="OTHER">Other</option>
                </select>
              </div>
            </div>

            <div className="form-section-title">💰 Loan Details</div>
            <div className="form-grid">
              <div className="form-group">
                <label>Loan Amount ($)</label>
                <input type="number" name="loan_amnt"
                       value={form.loan_amnt} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Interest Rate (%)</label>
                <input type="number" step="0.1" name="loan_int_rate"
                       value={form.loan_int_rate} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Loan Intent</label>
                <select name="loan_intent" value={form.loan_intent} onChange={handleChange}>
                  <option value="PERSONAL">Personal</option>
                  <option value="EDUCATION">Education</option>
                  <option value="MEDICAL">Medical</option>
                  <option value="VENTURE">Venture</option>
                  <option value="HOMEIMPROVEMENT">Home Improvement</option>
                  <option value="DEBTCONSOLIDATION">Debt Consolidation</option>
                </select>
              </div>
              <div className="form-group">
                <label>Loan Grade</label>
                <select name="loan_grade" value={form.loan_grade} onChange={handleChange}>
                  {["A","B","C","D","E","F","G"].map(g =>
                    <option key={g} value={g}>Grade {g}</option>)}
                </select>
              </div>
            </div>

            <div className="form-section-title">📊 Credit History</div>
            <div className="form-grid">
              <div className="form-group">
                <label>Credit History (yrs)</label>
                <input type="number" name="cb_person_cred_hist_length"
                       value={form.cb_person_cred_hist_length} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>Prior Default</label>
                <select name="cb_person_default_on_file"
                        value={form.cb_person_default_on_file} onChange={handleChange}>
                  <option value="N">No</option>
                  <option value="Y">Yes</option>
                </select>
              </div>
            </div>

            <button className="submit-btn" onClick={handleSubmit} disabled={loading}>
              {loading ? (
                <span className="loading-text">
                  <span className="spinner" /> Analyzing Risk...
                </span>
              ) : "⚡ Analyze Credit Risk"}
            </button>

            {error && <div className="error-msg">{error}</div>}
          </section>

          {/* ── RIGHT: Results ── */}
          <section className="results-col">

            {!result && !loading && (
              <div className="card empty-card">
                <div className="empty-icon">🏦</div>
                <div className="empty-title">Ready for Analysis</div>
                <div className="empty-sub">Fill in the loan application and click Analyze</div>
                <div className="model-stats">
                  <div className="stat"><span className="stat-val">0.9348</span><span className="stat-lbl">AUC-ROC</span></div>
                  <div className="stat"><span className="stat-val">0.9604</span><span className="stat-lbl">Precision</span></div>
                  <div className="stat"><span className="stat-val">39</span><span className="stat-lbl">Features</span></div>
                  <div className="stat"><span className="stat-val">32K</span><span className="stat-lbl">Records</span></div>
                </div>
              </div>
            )}

            {loading && (
              <div className="card empty-card">
                <div className="loading-spinner-large" />
                <div className="empty-title">Running XGBoost Model...</div>
                <div className="empty-sub">Computing SHAP explanations</div>
              </div>
            )}

            {result && !loading && (
              <>
                {/* Decision Banner */}
                <div className="card decision-card"
                     style={{ borderColor: decisionColor, boxShadow: `0 0 24px ${decisionColor}33` }}>
                  <div className="decision-top">
                    <div>
                      <div className="decision-icon">{DECISION_ICONS[result.decision]}</div>
                      <div className="decision-label" style={{ color: decisionColor }}>
                        {result.decision}
                      </div>
                      <div className="decision-desc">{result.grade_description}</div>
                    </div>
                    <div className="grade-badge" style={{ background: gradeColor }}>
                      {result.grade}
                    </div>
                  </div>

                  {/* Gauge */}
                  <div className="gauge-wrap">
                    <GaugeChart probability={result.default_probability} />
                    <div className="gauge-label">Default Probability</div>
                  </div>

                  {/* Credit Score */}
                  <div className="score-row">
                    <div className="score-label">Credit Score</div>
                    <div className="score-bar-track">
                      <div className="score-bar-fill"
                           style={{
                             width: `${((result.credit_score - 300) / 550) * 100}%`,
                             background: gradeColor,
                             boxShadow: `0 0 10px ${gradeColor}`
                           }} />
                    </div>
                    <div className="score-val" style={{ color: gradeColor }}>
                      {result.credit_score}
                    </div>
                  </div>
                </div>

                {/* Key Metrics */}
                <div className="card metrics-card">
                  <h3 className="card-title">
                    <span className="card-title-icon">📈</span> Key Risk Metrics
                  </h3>
                  <div className="metrics-grid">
                    <MetricCard label="DTI Ratio"
                      value={result.key_metrics.dti_ratio} unit="%"
                      color={result.key_metrics.dti_ratio > 40 ? "#ff4757" : "#2ed573"} />
                    <MetricCard label="Monthly Payment"
                      value={`$${result.key_metrics.monthly_payment.toFixed(0)}`} unit=""
                      color="#00d4ff" />
                    <MetricCard label="Payment/Income"
                      value={result.key_metrics.payment_to_income} unit="%"
                      color={result.key_metrics.payment_to_income > 20 ? "#ff4757" : "#2ed573"} />
                    <MetricCard label="Risk Score"
                      value={result.key_metrics.grade_risk_score} unit="%"
                      color={result.key_metrics.grade_risk_score > 50 ? "#ff4757" : "#ffa502"} />
                  </div>
                </div>

                {/* SHAP */}
                <div className="card shap-card">
                  <h3 className="card-title">
                    <span className="card-title-icon">🔍</span> SHAP Risk Factors
                    <span className="shap-legend">
                      <span style={{color:"#ff4757"}}>▲ increases risk</span>
                      <span style={{color:"#2ed573"}}>▼ decreases risk</span>
                    </span>
                  </h3>
                  <div className="shap-list">
                    {result.risk_factors.map((f, i) => (
                      <ShapBar key={i} factor={f} />
                    ))}
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      </main>

      <footer className="footer">
        Built by Mahaswi Shankar · XGBoost Credit Risk Engine · Portfolio Project
      </footer>
    </div>
  );
}