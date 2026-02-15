from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained Isolation Forest model
model = joblib.load(os.path.join(os.path.dirname(__file__), "isolation_forest_model.pkl"))

def predict_anomaly(data):
    try:
        # Extract features from data
        annual_income = data.get('annual_income', 50000)
        registrations_per_aadhaar = data.get('registrations_per_aadhaar', 1)
        bank_shared_count = data.get('bank_shared_count', 1)
        phone_shared_count = data.get('phone_shared_count', 1)

        # Validate inputs (basic checks)
        if not isinstance(annual_income, (int, float)) or annual_income < 0:
            return {'error': 'Invalid annual_income'}
        if not isinstance(registrations_per_aadhaar, int) or not 1 <= registrations_per_aadhaar <= 10:
            return {'error': 'Invalid registrations_per_aadhaar (1-10)'}
        if not isinstance(bank_shared_count, int) or not 1 <= bank_shared_count <= 15:
            return {'error': 'Invalid bank_shared_count (1-15)'}
        if not isinstance(phone_shared_count, int) or not 1 <= phone_shared_count <= 15:
            return {'error': 'Invalid phone_shared_count (1-15)'}

        # Prepare model input
        model_input = pd.DataFrame([[
            annual_income,
            registrations_per_aadhaar,
            bank_shared_count,
            phone_shared_count
        ]], columns=[
            "annual_income",
            "registrations_per_aadhaar",
            "bank_shared_count",
            "phone_shared_count"
        ])

        # Make prediction
        score = -model.decision_function(model_input)[0]
        prediction = model.predict(model_input)[0]  # -1 for anomaly, 1 for normal

        # Prepare response
        if prediction == -1:
            result = {
                'prediction': 'High Fraud Risk Detected',
                'anomaly_score': round(score, 4),
                'details': [
                    'Possible reasons:',
                    '- Duplicate Aadhaar registrations',
                    '- Shared bank account among many beneficiaries',
                    '- Shared phone number indicating collusion',
                    '- Income inconsistent with scheme eligibility'
                ]
            }
        else:
            result = {
                'prediction': 'Beneficiary Appears Normal',
                'anomaly_score': round(score, 4),
                'details': [
                    'No strong anomaly patterns detected based on the trained AI model.'
                ]
            }

        return result

    except Exception as e:
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = predict_anomaly(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
