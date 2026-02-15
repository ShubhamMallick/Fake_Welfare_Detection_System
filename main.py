from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from fpdf import FPDF
from fastapi.responses import StreamingResponse
from io import BytesIO
from fastapi.templating import Jinja2Templates
from fastapi import Request, UploadFile, File
import uvicorn
import importlib
import re
import asyncio

templates = Jinja2Templates(directory="templates")

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

@app.post("/pipeline")
async def pipeline(file: UploadFile = File(...)):
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
        fraud_task = asyncio.to_thread(fraud_mod.predict_fraud, {'beneficiary_id': features['beneficiary_id']})

        anomaly_result, duplicate_result, fraud_result = await asyncio.gather(anomaly_task, duplicate_task, fraud_task)

        if 'error' in anomaly_result:
            return JSONResponse(status_code=400, content=anomaly_result)
        if 'error' in duplicate_result:
            return JSONResponse(status_code=400, content=duplicate_result)
        if 'error' in fraud_result:
            return JSONResponse(status_code=400, content=fraud_result)

        # Step 5: Prepare case data for agentic reasoning
        case_data = features.copy()
        case_data['anomaly_score'] = anomaly_result.get('anomaly_score', 0)
        case_data['duplicate_risk'] = duplicate_result.get('duplicate_risk', 0) if duplicate_result.get('prediction') == 'Duplicate Detected' else 0
        case_data['fraud_probability'] = fraud_result.get('fraud_probability', 0)
        # Additional fields for risk scoring
        case_data['bank_shared_count'] = fraud_result['beneficiary_details'].get('bank_account', 1)  # simplistic
        case_data['phone_shared_count'] = fraud_result['beneficiary_details'].get('phone_number', 1)
        case_data['agent_cluster_size'] = fraud_result.get('connected_component_size', 1)

        # Step 6: Run agentic reasoning
        # agentic_result = await asyncio.to_thread(agentic_mod.analyze_agentic, {'case_data': case_data})
        # if 'error' in agentic_result:
        #     return JSONResponse(status_code=400, content=agentic_result)

        # Compile final response
        result = {
            'nlp_extraction': nlp_result,
            'anomaly_detection': anomaly_result,
            'duplicate_detection': duplicate_result,
            'fraud_network_analysis': fraud_result
            # 'agentic_reasoning': agentic_result
        }

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

@app.get("/dashboard-data")
def get_dashboard_data():
    try:
        # Import necessary modules
        import pandas as pd
        import networkx as nx
        from Fraud_Network_Analysis.backend import df, G
        
        # Basic stats
        total_beneficiaries = int(len(df))
        fraud_rings = int(sum(1 for c in nx.connected_components(G) if len(c) >= 5))
        
        # Anomaly stats (simplified)
        anomaly_count = int((df['fraud_ring_member'] == 1).sum())
        
        # Duplicate stats (simplified, using linkage)
        duplicate_count = int((df['is_duplicate'] == 1).sum())
        
        # Risk distribution
        high_risk = int((df['phone_degree'] > 3).sum())
        medium_risk = int(((df['phone_degree'] <= 3) & (df['phone_degree'] > 1)).sum())
        low_risk = int((df['phone_degree'] <= 1).sum())
        
        return {
            "total_beneficiaries": total_beneficiaries,
            "fraud_rings": fraud_rings,
            "anomaly_count": anomaly_count,
            "duplicate_count": duplicate_count,
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            }
        }
    except Exception as e:
        return {"error": str(e)}

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
    pdf.output(buffer)
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
