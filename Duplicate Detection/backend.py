from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from flask_cors import CORS

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset and trained XGBoost pipeline
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "duplicate_detection_50000_v4.csv"))

def load_pipeline():
    pipeline = joblib.load(os.path.join(os.path.dirname(__file__), "duplicate_detection_pipeline_xgb.pkl"))
    # Fix XGBoost compatibility issue
    if hasattr(pipeline, 'named_steps') and 'xgb' in pipeline.named_steps:
        model = pipeline.named_steps['xgb']
        if not hasattr(model, 'use_label_encoder'):
            model.use_label_encoder = True
    return pipeline

df = load_data()
pipeline = load_pipeline()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        aadhaar = data.get('aadhaar_like_id', '')
        name = data.get('name', '')
        household = data.get('household_id', '')
        phone = data.get('phone_number', '')
        bank = data.get('bank_account', '')
        district = data.get('district', 'District_1')

        # Convert to strings for consistent comparison
        aadhaar_str = str(aadhaar)
        phone_str = str(phone)
        bank_str = str(bank)
        household_str = str(household)

        # Compute linkage features
        aadhaar_count = df[df["aadhaar_like_id"].astype(str) == aadhaar_str].shape[0]
        phone_count = df[df["phone_number"].astype(str) == phone_str].shape[0]
        bank_count = df[df["bank_account"].astype(str) == bank_str].shape[0]
        household_size = df[df["household_id"].astype(str) == household_str].shape[0]

        # Validate inputs (basic checks)
        if not aadhaar_str:
            return jsonify({'error': 'Invalid aadhaar_like_id'}), 400
        if not phone_str:
            return jsonify({'error': 'Invalid phone_number'}), 400
        if not bank_str:
            return jsonify({'error': 'Invalid bank_account'}), 400
        if not household_str:
            return jsonify({'error': 'Invalid household_id'}), 400

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "aadhaar_like_id": aadhaar_str,
            "name": str(name),
            "household_id": household_str,
            "phone_number": phone_str,
            "bank_account": bank_str,
            "district": str(district),
            "aadhaar_count": float(aadhaar_count),
            "phone_count": float(phone_count),
            "bank_count": float(bank_count),
            "household_size": float(household_size)
        }])

        # Make prediction
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        duplicate_prob = prob * 100
        normal_prob = (1 - prob) * 100

        # Prepare response
        if pred == 1:
            result = {
                'prediction': 'Duplicate Detected',
                'duplicate_risk': round(duplicate_prob, 2),
                'details': [
                    'Possible Reasons',
                    '- Shared phone or bank across many beneficiaries',
                    '- Same Aadhaar appearing multiple times',
                    '- Large household cluster',
                    '- Cross-district duplication pattern'
                ]
            }
        else:
            result = {
                'prediction': 'Appears Genuine',
                'confidence_genuine': round(normal_prob, 2)
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
