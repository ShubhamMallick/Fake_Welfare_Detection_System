from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.templating import Jinja2Templates
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import uvicorn
import importlib
import re
import asyncio
import numpy as np

templates = Jinja2Templates(directory="templates")

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# Import the Flask apps from the backends
anomaly_mod = importlib.import_module("Anomaly_Detection.backend")
duplicate_mod = importlib.import_module("Duplicate_Detection.backend")
fraud_mod = importlib.import_module("Fraud_Network_Analysis.backend")
nlp_mod = importlib.import_module("NLP_Extractor.backend_nlp")
admin_decision_mod = importlib.import_module("Admin_Decision_Layer.backend")
# agentic_mod = importlib.import_module("Agentic_Reasoning.backend")

anomaly_app = anomaly_mod.app
duplicate_app = duplicate_mod.app
fraud_app = fraud_mod.app
nlp_app = nlp_mod.app
admin_decision_app = admin_decision_mod.app
# agentic_app = agentic_mod.app

app = FastAPI(title="Prayatna Fraud Detection API", description="Integrated API for all fraud detection backends")

# Mount the Flask apps as WSGI middleware
app.mount("/anomaly", WSGIMiddleware(anomaly_app))
app.mount("/duplicate", WSGIMiddleware(duplicate_app))
app.mount("/fraud-network", WSGIMiddleware(fraud_app))
app.mount("/nlp-extractor", WSGIMiddleware(nlp_app))
app.mount("/admin-decision", WSGIMiddleware(admin_decision_app))
# app.mount("/agentic-reasoning", WSGIMiddleware(agentic_app))

@app.post("/pipeline-basic")
async def pipeline_basic(file: UploadFile = File(...)):
    try:
        # Step 1: Extract features from PDF using NLP
        nlp_result = await asyncio.to_thread(nlp_mod.extract_nlp, file.file)
        if 'error' in nlp_result:
            return JSONResponse(status_code=400, content=nlp_result)

        # Build features dict from NLP results
        regex = nlp_result['regex_results']
        nlp_ents = nlp_result['nlp_results']
        features = {}

        # Parse annual income
        if 'Annual Income' in regex:
            income_str = regex['Annual Income'][0]
            # Remove currency symbols and commas
            clean_income = re.sub(r'[₹,Rs\.\s]', '', income_str)
            try:
                features['annual_income'] = float(clean_income)
            except ValueError:
                features['annual_income'] = 50000
        else:
            features['annual_income'] = 50000

        # Extract other fields
        if 'Aadhaar ID' in regex:
            features['aadhaar_like_id'] = regex['Aadhaar ID'][0]
        if 'Phone Number' in regex:
            features['phone_number'] = regex['Phone Number'][0]
        if 'Bank Account' in regex:
            features['bank_account'] = regex['Bank Account'][0]
        if 'Beneficiary ID' in regex:
            features['beneficiary_id'] = regex['Beneficiary ID'][0]
        if 'Household ID' in regex:
            features['household_id'] = regex['Household ID'][0]
        if 'Name' in nlp_ents and nlp_ents['Name']:
            features['name'] = nlp_ents['Name'][0]
        else:
            features['name'] = ''

        # Set defaults for missing fields
        features['registrations_per_aadhaar'] = 1
        features['bank_shared_count'] = 1
        features['phone_shared_count'] = 1
        features['district'] = 'District_1'

        # Ensure beneficiary_id is present
        if 'beneficiary_id' not in features:
            features['beneficiary_id'] = 'BEN0001'

        # Step 2-4: Run anomaly, duplicate, and fraud predictions in parallel
        anomaly_task = asyncio.to_thread(anomaly_mod.predict_anomaly, features)
        duplicate_task = asyncio.to_thread(duplicate_mod.predict_duplicate, features)
        fraud_task = asyncio.to_thread(fraud_mod.predict_fraud, {'features': {'Beneficiary ID': features['beneficiary_id']}})

        anomaly_result, duplicate_result, fraud_result = await asyncio.gather(anomaly_task, duplicate_task, fraud_task)

        # Normalize anomaly result (allow only successful dicts through)
        if isinstance(anomaly_result, tuple):
            content, status = anomaly_result
            if status != 200:
                return JSONResponse(status_code=status, content=content)
            anomaly_result = content
        elif isinstance(anomaly_result, dict) and 'error' in anomaly_result:
            return JSONResponse(status_code=400, content=anomaly_result)
        
        # Normalize duplicate result
        if isinstance(duplicate_result, tuple):
            content, status = duplicate_result
            if status != 200:
                return JSONResponse(status_code=status, content=content)
            duplicate_result = content
        elif isinstance(duplicate_result, dict) and 'error' in duplicate_result:
            return JSONResponse(status_code=400, content=duplicate_result)
        
        # Normalize fraud result
        if isinstance(fraud_result, tuple):
            content, status = fraud_result
            # Special-case: beneficiary not in dataset should NOT break the pipeline
            if isinstance(content, dict) and content.get('error') == 'Beneficiary not in dataset':
                fraud_result = {
                    'beneficiary_id': features['beneficiary_id'],
                    'ml_prediction': 'Beneficiary Not In Network Dataset',
                    'fraud_probability': 0.0,
                    'normal_probability': 100.0,
                    'connected_component_size': 0,
                    'fraud_ring_detected': False,
                    'degree_centrality': 0.0,
                    'master_agent_detected': False,
                    'beneficiary_details': {}
                }
            elif status != 200:
                return JSONResponse(status_code=status, content=content)
            else:
                fraud_result = content
        elif isinstance(fraud_result, dict) and 'error' in fraud_result:
            # Same special-case handling when predict_fraud returns a plain dict
            if fraud_result.get('error') == 'Beneficiary not in dataset':
                fraud_result = {
                    'beneficiary_id': features['beneficiary_id'],
                    'ml_prediction': 'Beneficiary Not In Network Dataset',
                    'fraud_probability': 0.0,
                    'normal_probability': 100.0,
                    'connected_component_size': 0,
                    'fraud_ring_detected': False,
                    'degree_centrality': 0.0,
                    'master_agent_detected': False,
                    'beneficiary_details': {}
                }
            else:
                return JSONResponse(status_code=400, content=fraud_result)

        # Compile response without agentic
        result = {
            "nlp_extraction": nlp_result,
            "anomaly_detection": anomaly_result,
            "duplicate_detection": duplicate_result,
            "fraud_network_analysis": fraud_result
        }

        result = convert_to_serializable(result)

        return JSONResponse(content=result)

    except Exception as e:
        print("Exception in pipeline_basic:", str(e))
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get("/dashboard-data")
def get_dashboard_data():
    # Load data from admin decisions
    import os
    data_file = os.path.join(os.path.dirname(__file__), 'admin_decisions.json')
    if os.path.exists(data_file):
        import json
        with open(data_file, 'r') as f:
            data = json.load(f)
    else:
        data = {'cases': [], 'audit': []}
    
    anomalies = sum(1 for c in data['cases'] if c.get('prediction') == 'Anomaly Detected')
    duplicates = sum(1 for c in data['cases'] if c.get('prediction') == 'Duplicate Suspected')
    frauds = sum(1 for c in data['cases'] if 'Fraud' in c.get('prediction', ''))
    agentic_reports = len(data['audit'])
    total_cases = len(data['cases'])
    
    return {
        "total_cases": total_cases,
        "anomalies": anomalies,
        "duplicates": duplicates,
        "frauds": frauds,
        "agentic_reports": agentic_reports,
        "cases": data['cases']
    }

@app.post("/generate-report")
async def generate_report(case_data: dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fraud Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt="", ln=True)  # space
    
    def add_data(key, value, indent=0):
        indent_str = "  " * indent
        if isinstance(value, dict):
            pdf.cell(200, 10, txt=f"{indent_str}{key}:", ln=True)
            for subkey, subvalue in value.items():
                add_data(subkey, subvalue, indent + 1)
        elif isinstance(value, list):
            pdf.cell(200, 10, txt=f"{indent_str}{key}:", ln=True)
            for item in value:
                pdf.cell(200, 10, txt=f"{indent_str}  - {item}", ln=True)
        else:
            pdf.cell(200, 10, txt=f"{indent_str}{key}: {value}", ln=True)
    
    for key, value in case_data.items():
        add_data(key, value)
    
    buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='application/pdf', headers={"Content-Disposition": "attachment; filename=fraud_report.pdf"})

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("nlp.html", {"request": request})

@app.get("/nlp")
def nlp_page(request: Request):
    return templates.TemplateResponse("nlp.html", {"request": request})

@app.get("/anomaly")
def anomaly_page(request: Request):
    return templates.TemplateResponse("anomaly.html", {"request": request})

@app.get("/duplicate")
def duplicate_page(request: Request):
    return templates.TemplateResponse("duplicate.html", {"request": request})

@app.get("/fraud")
def fraud_page(request: Request):
    return templates.TemplateResponse("fraud.html", {"request": request})

@app.get("/agentic")
def agentic_page(request: Request):
    return templates.TemplateResponse("agentic.html", {"request": request})

@app.get("/pipeline-page")
def pipeline_page(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})

@app.get("/choice")
def choice_page(request: Request):
    return templates.TemplateResponse("choice.html", {"request": request})

@app.get("/dashboard")
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin-decision-page")
def admin_decision_page(request: Request):
    return templates.TemplateResponse("admin_decision.html", {"request": request})

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("nlp.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)