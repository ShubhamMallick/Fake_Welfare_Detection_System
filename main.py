from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import uvicorn

# Import the Flask apps from the backends
from Anomaly_Detection.backend import app as anomaly_app
from Duplicate_Detection.backend import app as duplicate_app
from Fraud_Network_Analysis.backend import app as fraud_app
from nlp_extractor.backend_nlp import app as nlp_app
from Agentic_Reasoning.backend import app as agentic_app

app = FastAPI(title="Prayatna Fraud Detection API", description="Integrated API for all fraud detection backends")

# Mount the Flask apps as WSGI middleware
app.mount("/anomaly", WSGIMiddleware(anomaly_app))
app.mount("/duplicate", WSGIMiddleware(duplicate_app))
app.mount("/fraud-network", WSGIMiddleware(fraud_app))
app.mount("/nlp-extractor", WSGIMiddleware(nlp_app))
app.mount("/agentic-reasoning", WSGIMiddleware(agentic_app))

@app.get("/")
def root():
    return {"message": "Welcome to Prayatna Fraud Detection API", "endpoints": {
        "anomaly": "/anomaly/predict",
        "duplicate": "/duplicate/predict",
        "fraud-network": "/fraud-network/predict",
        "nlp-extractor": "/nlp-extractor/extract",
        "agentic-reasoning": "/agentic-reasoning/analyze"
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
