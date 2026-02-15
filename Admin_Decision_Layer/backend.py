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
