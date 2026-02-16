from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATA_FILE = 'admin_decisions.json'

def load_decisions():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {'cases': [], 'audit': []}

def save_decisions(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/cases')
def get_cases():
    data = load_decisions()
    return jsonify(data['cases'])

@app.route('/decide', methods=['POST'])
def submit_decision():
    decision_data = request.json
    case_id = decision_data.get('case_id')
    decision = decision_data.get('decision')  # 'approve' or 'reject'
    notes = decision_data.get('notes', '')
    
    data = load_decisions()
    
    # Add to audit
    audit_entry = {
        'case_id': case_id,
        'decision': decision,
        'notes': notes,
        'timestamp': datetime.now().isoformat(),
        'admin': 'admin_user'  # Placeholder
    }
    data['audit'].append(audit_entry)
    
    # Update case status
    for case in data['cases']:
        if case['id'] == case_id:
            case['status'] = decision
            case['notes'] = notes
            break
    
    save_decisions(data)
    return jsonify({'status': 'success'})

@app.route('/audit')
def get_audit():
    data = load_decisions()
    return jsonify(data['audit'])

@app.route('/agentic-reasoning/analyze', methods=['POST'])
def agentic_analyze():
    data = request.json
    case_data = data.get('case_data', {})
    
    # Mock AI reasoning based on case data
    ml_pred = case_data.get('ml_prediction', 'Unknown')
    fraud_prob = case_data.get('fraud_probability', 0)
    normal_prob = case_data.get('normal_probability', 100 - fraud_prob)
    
    explanation = f"The analysis indicates a prediction of '{ml_pred}' with a fraud probability of {fraud_prob}%. "
    if ml_pred.lower() == 'fraud':
        explanation += "This suggests potential fraudulent activity. Key factors include high anomaly scores and suspicious network connections."
    else:
        explanation += f"The case appears legitimate with a confidence level of {normal_prob}%. No immediate red flags detected."
    
    audit_summary = f"Audit Recommendation: {'Immediate investigation required' if fraud_prob > 50 else 'Monitor for changes'}. "
    audit_summary += f"Case ID: {case_data.get('beneficiary_details', {}).get('beneficiary_id', 'N/A')}. "
    audit_summary += "Document all findings and follow up within 7 days."
    
    return jsonify({
        'explanation': explanation,
        'audit_summary': audit_summary
    })

# Mock some cases for demo
@app.route('/init-cases')
def init_cases():
    data = load_decisions()
    if not data['cases']:
        data['cases'] = [
            {'id': '1', 'beneficiary_id': 'BEN10458932', 'prediction': 'Anomaly Detected', 'status': 'pending', 'anomaly_score': 0.85},
            {'id': '2', 'beneficiary_id': 'BEN20567890', 'prediction': 'Duplicate Suspected', 'status': 'pending', 'duplicate_risk': 75}
        ]
        save_decisions(data)
    return jsonify({'status': 'initialized'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
