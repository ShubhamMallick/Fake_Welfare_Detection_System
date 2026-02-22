from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import re

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

# Load environment
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

DATA_FILE = 'admin_decisions.json'

def load_decisions():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {'cases': [], 'audit': []}

def save_decisions(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def final_risk_score(case_data):
    """Calculate risk score based on suspicious patterns"""
    risk_score = (
        case_data.get('bank_shared_count', 0) * 0.4 +
        case_data.get('phone_shared_count', 0) * 0.3 +
        case_data.get('registrations_per_aadhaar', 0) * 0.2 +
        case_data.get('agent_cluster_size', 0) * 0.1
    )
    
    return {
        'risk_score': min(risk_score, 10),
        'risk_level': 'High' if risk_score > 7 else 'Medium' if risk_score > 4 else 'Low',
        'factors': [
            f"Bank shared count: {case_data.get('bank_shared_count', 0)}",
            f"Phone shared count: {case_data.get('phone_shared_count', 0)}",
            f"Registrations per Aadhaar: {case_data.get('registrations_per_aadhaar', 0)}"
        ]
    }

def clean_json_response(response_text):
    """Clean AI response by removing markdown code blocks and extracting JSON"""
    # Remove markdown code blocks
    response_text = re.sub(r'```\w*\n?', '', response_text)
    response_text = re.sub(r'```', '', response_text)
    
    # Remove any leading/trailing whitespace
    response_text = response_text.strip()
    
    # Find JSON object boundaries (from first { to last })
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1
    
    if start_idx != -1 and end_idx > start_idx:
        response_text = response_text[start_idx:end_idx]
    
    return response_text

# LangChain prompts
explain_prompt = PromptTemplate(
    input_variables=["nlp", "anomaly", "duplicate", "fraud"],
    template="""
You are a government fraud detection analyst.

Analyze the following data from the fraud detection pipeline and explain why the case is suspicious.

NLP Extraction: {nlp}

Anomaly Detection: {anomaly}

Duplicate Detection: {duplicate}

Fraud Network Analysis: {fraud}

Provide:
- Bullet-point reasoning
- Mention duplicate identity, shared financial links, or abnormal registrations
- Keep explanation concise and professional

Output your response as a JSON object with keys 'summary' (string) and 'key_points' (array of strings).
"""
)

audit_prompt = PromptTemplate(
    input_variables=["nlp", "anomaly", "duplicate", "fraud", "explanation"],
    template="""
You are a senior government welfare fraud auditor.

Generate an official audit summary for the following suspicious beneficiary case based on the pipeline data.

NLP Extraction: {nlp}

Anomaly Detection: {anomaly}

Duplicate Detection: {duplicate}

Fraud Network Analysis: {fraud}

AI Explanation: {explanation}

Your report must include:

1. Case Overview
2. Key Fraud Indicators
3. Evidence Summary
4. Recommended Government Action

Write in clear, formal, audit-ready language.

Output your response as a JSON object with keys 'case_overview' (string), 'key_fraud_indicators' (array of strings), 'evidence_summary' (string), 'recommended_action' (string).
"""
)

explanation_chain = explain_prompt | llm

audit_chain = audit_prompt | llm

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
    
    if 'case_data' in data:
        # Standalone mode (from agentic page)
        nlp = "Not available (standalone mode)"
        anomaly = "Not available (standalone mode)"
        duplicate = "Not available (standalone mode)"
        fraud = json.dumps(data['case_data'])
    else:
        # Pipeline mode (from pipeline endpoint)
        nlp = json.dumps(data.get('nlp_extraction', {}))
        anomaly = json.dumps(data.get('anomaly_detection', {}))
        duplicate = json.dumps(data.get('duplicate_detection', {}))
        fraud = json.dumps(data.get('fraud_network_analysis', {}))
    
    # Generate explanation
    explanation_prompt_formatted = explain_prompt.format(nlp=nlp, anomaly=anomaly, duplicate=duplicate, fraud=fraud)
    explanation_raw = llm.invoke([{"role": "user", "content": explanation_prompt_formatted}]).content
    
    # Clean the response and parse JSON
    explanation_cleaned = clean_json_response(explanation_raw)
    try:
        explanation_data = json.loads(explanation_cleaned)
    except json.JSONDecodeError:
        explanation_data = {'summary': explanation_raw, 'key_points': []}
    
    # Generate audit summary
    audit_prompt_formatted = audit_prompt.format(nlp=nlp, anomaly=anomaly, duplicate=duplicate, fraud=fraud, explanation=json.dumps(explanation_data))
    audit_raw = llm.invoke([{"role": "user", "content": audit_prompt_formatted}]).content
    
    # Clean the response and parse JSON
    audit_cleaned = clean_json_response(audit_raw)
    try:
        audit_data = json.loads(audit_cleaned)
    except json.JSONDecodeError:
        audit_data = {'case_overview': audit_raw, 'key_fraud_indicators': [], 'evidence_summary': '', 'recommended_action': ''}
    
    return jsonify({
        'explanation': explanation_data,
        'audit_summary': audit_data
    })

# Mock some cases for demo
@app.route('/init-cases')
def init_cases():
    data = load_decisions()
    if not data['cases']:
        data['cases'] = [
            {'id': '1', 'beneficiary_id': 'BEN10458932', 'prediction': 'Anomaly Detected', 'status': 'pending', 'anomaly_score': 0.85, 'fraud_probability': 85},
            {'id': '2', 'beneficiary_id': 'BEN20567890', 'prediction': 'Duplicate Suspected', 'status': 'pending', 'duplicate_risk': 75, 'fraud_probability': 60}
        ]
        save_decisions(data)
    return jsonify({'status': 'initialized'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
