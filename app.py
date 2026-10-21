import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic.pkl")
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with Glassmorphism & Modern Styling
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loan Eligibility Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(23, 32, 54, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.35);
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --input-bg: rgba(15, 23, 42, 0.6);
      --input-border: rgba(148, 163, 184, 0.15);
      --success: #10b981;
      --danger: #ef4444;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: 'Plus Jakarta Sans', sans-serif;
    }

    body {
      min-height: 100vh;
      background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.15), transparent 40%),
                  radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.12), transparent 40%),
                  var(--bg);
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 2.5rem 1rem;
      color: var(--text-primary);
    }

    .container {
      width: 100%;
      max-width: 820px;
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 24px;
      padding: 2.5rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    header {
      text-align: center;
      margin-bottom: 2rem;
    }

    header h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      background: linear-gradient(to right, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    header p {
      color: var(--text-secondary);
      font-size: 0.95rem;
      margin-top: 0.4rem;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    label {
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    input, select {
      background: var(--input-bg);
      border: 1px solid var(--input-border);
      border-radius: 12px;
      padding: 0.75rem 0.9rem;
      color: var(--text-primary);
      font-size: 0.95rem;
      outline: none;
      transition: all 0.2s ease;
    }

    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    select option {
      background: #0f172a;
      color: var(--text-primary);
    }

    .submit-btn {
      width: 100%;
      margin-top: 2rem;
      padding: 0.95rem 1.5rem;
      border: none;
      border-radius: 12px;
      background: var(--accent);
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      box-shadow: 0 4px 20px var(--accent-glow);
      transition: background 0.2s ease, transform 0.1s ease;
    }

    .submit-btn:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }

    .submit-btn:active {
      transform: translateY(1px);
    }

    #result-box {
      margin-top: 1.5rem;
      display: none;
      padding: 1.25rem;
      border-radius: 14px;
      text-align: center;
      animation: fadeIn 0.3s ease;
    }

    .result-approved {
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--success);
    }

    .result-rejected {
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: var(--danger);
    }

    .result-title {
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 0.25rem;
    }

    .result-subtitle {
      font-size: 0.85rem;
      color: var(--text-secondary);
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Loan Eligibility Evaluator</h1>
      <p>Powered by Scikit-Learn Logistic Regression</p>
    </header>

    <form id="prediction-form">
      <div class="form-grid">
        <div class="form-group">
          <label>Gender</label>
          <select name="Gender" required>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </div>

        <div class="form-group">
          <label>Married</label>
          <select name="Married" required>
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label>Dependents</label>
          <select name="Dependents" required>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3+">3+</option>
          </select>
        </div>

        <div class="form-group">
          <label>Education</label>
          <select name="Education" required>
            <option value="Graduate">Graduate</option>
            <option value="Not Graduate">Not Graduate</option>
          </select>
        </div>

        <div class="form-group">
          <label>Self Employed</label>
          <select name="Self_Employed" required>
            <option value="No">No</option>
            <option value="Yes">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label>Credit History</label>
          <select name="Credit_History" required>
            <option value="1.0">Meets guidelines (1.0)</option>
            <option value="0.0">Does not meet (0.0)</option>
          </select>
        </div>

        <div class="form-group">
          <label>Property Area</label>
          <select name="Property_Area" required>
            <option value="Urban">Urban</option>
            <option value="Semiurban">Semiurban</option>
            <option value="Rural">Rural</option>
          </select>
        </div>

        <div class="form-group">
          <label>Applicant Income ($)</label>
          <input type="number" step="any" name="ApplicantIncome" value="5000" required>
        </div>

        <div class="form-group">
          <label>Coapplicant Income ($)</label>
          <input type="number" step="any" name="CoapplicantIncome" value="1500" required>
        </div>

        <div class="form-group">
          <label>Loan Amount (k$)</label>
          <input type="number" step="any" name="LoanAmount" value="128" required>
        </div>

        <div class="form-group">
          <label>Term (Months)</label>
          <input type="number" step="any" name="Loan_Amount_Term" value="360" required>
        </div>
      </div>

      <button type="submit" class="submit-btn" id="submit-btn">Run Prediction</button>
    </form>

    <div id="result-box">
      <div class="result-title" id="result-title"></div>
      <div class="result-subtitle" id="result-subtitle"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('prediction-form');
    const resultBox = document.getElementById('result-box');
    const resultTitle = document.getElementById('result-title');
    const resultSubtitle = document.getElementById('result-subtitle');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(form);
      const payload = Object.fromEntries(formData.entries());

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        resultBox.style.display = 'block';

        if (data.status === 'success') {
          const isEligible = data.prediction === 'Y';
          resultBox.className = isEligible ? 'result-approved' : 'result-rejected';
          resultTitle.innerText = isEligible ? 'Loan Approved (Class: Y)' : 'Loan Not Approved (Class: N)';
          resultSubtitle.innerText = data.probability 
            ? `Confidence: ${(data.probability * 100).toFixed(1)}%`
            : 'Decision calculated successfully.';
        } else {
          resultBox.className = 'result-rejected';
          resultTitle.innerText = 'Evaluation Error';
          resultSubtitle.innerText = data.message;
        }
      } catch (err) {
        resultBox.style.display = 'block';
        resultBox.className = 'result-rejected';
        resultTitle.innerText = 'Server Error';
        resultSubtitle.innerText = 'Could not communicate with the prediction backend.';
      }
    });
  </script>
</body>
</html>
"""

def preprocess_input(data):
    """
    Transforms dictionary payload into formatted DataFrame matching
    the exact feature names stored in the pickle object:
    ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
     'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
     'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    """
    encoding_map = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    }

    # If the logistic regression expects numeric values, map them; otherwise keep original
    row = {
        'Gender': encoding_map['Gender'].get(data.get('Gender'), 1),
        'Married': encoding_map['Married'].get(data.get('Married'), 0),
        'Dependents': encoding_map['Dependents'].get(str(data.get('Dependents')), 0),
        'Education': encoding_map['Education'].get(data.get('Education'), 1),
        'Self_Employed': encoding_map['Self_Employed'].get(data.get('Self_Employed'), 0),
        'ApplicantIncome': float(data.get('ApplicantIncome', 0)),
        'CoapplicantIncome': float(data.get('CoapplicantIncome', 0)),
        'LoanAmount': float(data.get('LoanAmount', 0)),
        'Loan_Amount_Term': float(data.get('Loan_Amount_Term', 360)),
        'Credit_History': float(data.get('Credit_History', 1.0)),
        'Property_Area': encoding_map['Property_Area'].get(data.get('Property_Area'), 1)
    }

    expected_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area'
    ]

    return pd.DataFrame([row])[expected_cols]

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model file not found on server.'}), 500

    try:
        data = request.get_json(force=True)
        df_features = preprocess_input(data)

        # Attempt prediction directly or with numeric fallback
        try:
            pred = model.predict(df_features)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df_features)[0]
                prob = float(probs[list(model.classes_).index(pred)])
        except Exception:
            # Fallback in case model expected string labels directly
            raw_row = {col: data.get(col) for col in df_features.columns}
            raw_df = pd.DataFrame([raw_row])
            pred = model.predict(raw_df)[0]
            prob = None

        return jsonify({
            'status': 'success',
            'prediction': str(pred),
            'probability': prob
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
