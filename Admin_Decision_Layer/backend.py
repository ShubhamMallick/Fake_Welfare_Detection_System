from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import time
import io
import gtts

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

def clean_json_response(text):
    """Remove markdown code blocks from JSON response"""
    if text.startswith('```json'):
        text = text[7:]  # Remove ```json
    if text.endswith('```'):
        text = text[:-3]  # Remove ```
    return text.strip()

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

def analyze_agentic(data):
    print("Agentic analyze called with data:", data)

    try:
        # Decide mode based on payload shape
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
        explanation_prompt_formatted = explain_prompt.format(
            nlp=nlp,
            anomaly=anomaly,
            duplicate=duplicate,
            fraud=fraud,
        )
        explanation_raw = llm.invoke([
            {"role": "user", "content": explanation_prompt_formatted}
        ]).content
        explanation_raw = clean_json_response(explanation_raw)

        try:
            explanation_data = json.loads(explanation_raw)
            print("Parsed explanation data:", explanation_data)
        except json.JSONDecodeError:
            explanation_data = {'summary': explanation_raw, 'key_points': []}
            print("Failed to parse explanation, using fallback:", explanation_data)

        # Generate audit summary
        audit_prompt_formatted = audit_prompt.format(
            nlp=nlp,
            anomaly=anomaly,
            duplicate=duplicate,
            fraud=fraud,
            explanation=json.dumps(explanation_data),
        )
        audit_raw = llm.invoke([
            {"role": "user", "content": audit_prompt_formatted}
        ]).content
        audit_raw = clean_json_response(audit_raw)

        try:
            audit_data = json.loads(audit_raw)
            print("Parsed audit data:", audit_data)
        except json.JSONDecodeError:
            audit_data = {
                'case_overview': audit_raw,
                'key_fraud_indicators': [],
                'evidence_summary': '',
                'recommended_action': ''
            }
            print("Failed to parse audit, using fallback:", audit_data)

        result = {
            'explanation': explanation_data,
            'audit_summary': audit_data
        }
        print("Returning data:", result)
        return result

    except Exception as e:
        # Catch any LLM / prompt / network errors so the pipeline
        # does not crash with a 500. Caller will turn this into 400.
        error_msg = f"Agentic reasoning failed: {str(e)}"
        print(error_msg)
        return {'error': error_msg}

@app.route('/agentic-reasoning/analyze', methods=['POST'])
def agentic_analyze():
    data = request.json
    return jsonify(analyze_agentic(data))

@app.route('/agentic-stream')
def agentic_stream():
    data_str = request.args.get('data')
    if not data_str:
        return Response('data: {"error": "No data provided"}\n\n', mimetype='text/event-stream')

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return Response('data: {"error": "Invalid data"}\n\n', mimetype='text/event-stream')

    def generate():
        try:
            # Decide mode
            if 'case_data' in data:
                nlp = "Not available (standalone mode)"
                anomaly = "Not available (standalone mode)"
                duplicate = "Not available (standalone mode)"
                fraud = json.dumps(data['case_data'])
            else:
                nlp = json.dumps(data.get('nlp_extraction', {}))
                anomaly = json.dumps(data.get('anomaly_detection', {}))
                duplicate = json.dumps(data.get('duplicate_detection', {}))
                fraud = json.dumps(data.get('fraud_network_analysis', {}))

            # Generate explanation
            explanation_prompt_formatted = explain_prompt.format(
                nlp=nlp, anomaly=anomaly, duplicate=duplicate, fraud=fraud
            )
            explanation_raw = llm.invoke([{"role": "user", "content": explanation_prompt_formatted}]).content
            explanation_raw = clean_json_response(explanation_raw)

            try:
                explanation_data = json.loads(explanation_raw)
                explanation_text = explanation_data.get('summary', explanation_raw) + ' ' + ' '.join(explanation_data.get('key_points', []))
            except json.JSONDecodeError:
                explanation_text = explanation_raw

            # Stream explanation word by word
            words = explanation_text.split()
            for word in words:
                yield f'data: {{"type": "explanation", "text": "{word} "}}\n\n'
                time.sleep(0.05)  # simulate real-time

            # Generate audit summary
            audit_prompt_formatted = audit_prompt.format(
                nlp=nlp, anomaly=anomaly, duplicate=duplicate, fraud=fraud, explanation=json.dumps(explanation_data)
            )
            audit_raw = llm.invoke([{"role": "user", "content": audit_prompt_formatted}]).content
            audit_raw = clean_json_response(audit_raw)

            try:
                audit_data = json.loads(audit_raw)
                audit_text = audit_data.get('case_overview', '') + ' ' + ' '.join(audit_data.get('key_fraud_indicators', [])) + ' ' + audit_data.get('evidence_summary', '') + ' ' + audit_data.get('recommended_action', '')
            except json.JSONDecodeError:
                audit_text = audit_raw

            # Stream audit word by word
            words = audit_text.split()
            for word in words:
                yield f'data: {{"type": "audit", "text": "{word} "}}\n\n'
                time.sleep(0.05)

            yield 'data: {"type": "end"}\n\n'

        except Exception as e:
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return Response(generate(), mimetype='text/event-stream')

@app.route('/generate-audio')
def generate_audio():
    text = request.args.get('text')
    if not text:
        return 'No text provided', 400

    try:
        tts = gtts.gTTS(text)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/mpeg')
    except Exception as e:
        return str(e), 500

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
