from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Load LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

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

# Explanation chain
explain_prompt = PromptTemplate(
    input_variables=["case_data"],
    template="""
You are a government fraud detection analyst.

Analyze the following welfare beneficiary data and explain why the case is suspicious.

Case Data:
{case_data}

Provide:
- Bullet-point reasoning
- Mention duplicate identity, shared financial links, or abnormal registrations
- Keep explanation concise and professional
"""
)

explanation_chain = LLMChain(
    llm=llm,
    prompt=explain_prompt
)

# Audit summary chain
audit_prompt = PromptTemplate(
    input_variables=["case_data", "risk_data", "explanation"],
    template="""
You are a senior government welfare fraud auditor.

Generate an official audit summary for the following suspicious beneficiary case.

CASE DATA:
{case_data}

RISK ANALYSIS:
{risk_data}

AI EXPLANATION:
{explanation}

Your report must include:

1. Case Overview
2. Key Fraud Indicators
3. Risk Score Interpretation
4. Evidence Summary
5. Recommended Government Action

Write in clear, formal, audit-ready language.
"""
)

audit_chain = LLMChain(
    llm=llm,
    prompt=audit_prompt
)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        case_data = data.get('case_data')

        if not case_data:
            return jsonify({'error': 'case_data is required'}), 400

        # Step 1: Risk score
        risk_data = final_risk_score(case_data)

        # Step 2: Explanation
        explanation = explanation_chain.run(case_data=str(case_data))

        # Step 3: Audit summary
        audit_summary = audit_chain.run(
            case_data=str(case_data),
            risk_data=str(risk_data),
            explanation=explanation
        )

        # Prepare response
        result = {
            'case_data': case_data,
            'risk_analysis': risk_data,
            'explanation': explanation,
            'audit_summary': audit_summary
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)
