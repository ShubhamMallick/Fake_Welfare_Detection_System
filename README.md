# Prayatna – Fake Welfare / Fraud Detection System

An integrated, modular fraud-detection system for welfare/beneficiary programs. The project combines multiple ML modules (NLP extraction, anomaly detection, duplicate detection, fraud-network analysis) behind a single **FastAPI** gateway and provides a modern multi-page web UI (Jinja2 templates).

## Features

- **NLP Feature Extraction (PDF)**
  - Extracts structured features from beneficiary PDFs using regex + spaCy NER.
- **Anomaly Detection**
  - IsolationForest-based anomaly scoring for fraud risk patterns.
- **Duplicate Detection**
  - XGBoost pipeline for detecting likely duplicate/linked beneficiary registrations.
- **Fraud Network Analysis**
  - NetworkX graph + ML model to detect fraud rings and high-centrality “master agent” behavior.
- **Pipeline Orchestration**
  - Upload one PDF and run NLP → anomaly → duplicate → fraud-network sequentially.
- **Dashboard**
  - Visual + tabular overview of cases and scores.
- **Admin Decision Layer**
  - Stores admin decisions and maintains an audit trail in `admin_decisions.json`.

## Project Structure

```
Prayatna/
  main.py
  requirements.txt
  admin_decisions.json

  templates/
    nlp.html
    anomaly.html
    duplicate.html
    fraud.html
    agentic.html
    pipeline.html
    dashboard.html
    choice.html
    admin_decision.html

  NLP_Extractor/
    backend_nlp.py
    NLP_Extractor.py
    Document*.pdf/docx

  Anomaly_Detection/
    backend.py
    isolation_forest_model.pkl
    Anomaly_Detection_50000.csv

  Duplicate_Detection/
    backend.py
    duplicate_detection_pipeline_xgb.pkl
    duplicate_detection_50000_v4.csv

  Fraud_Network_Analysis/
    backend.py
    fraud_network_model.pkl
    fraud_network_50000.csv
    graph_cache.pkl

  Admin_Decision_Layer/
    backend.py

  Agentic_Reasoning/
    backend.py
    *.ipynb
```

## Architecture Overview

- **FastAPI** (`main.py`) is the primary entrypoint.
- Individual module backends are implemented as **Flask apps** and are mounted into FastAPI using **`WSGIMiddleware`**.
- The browser UI is served via **Jinja2 templates** from `templates/`.

### Mounted Backends (FastAPI → Flask)

`main.py` mounts these Flask apps:

- `/nlp-extractor` → `NLP_Extractor/backend_nlp.py`
- `/anomaly` → `Anomaly_Detection/backend.py`
- `/duplicate` → `Duplicate_Detection/backend.py`
- `/fraud-network` → `Fraud_Network_Analysis/backend.py`
- `/admin-decision` → `Admin_Decision_Layer/backend.py`

> Note: There is also an LLM-based agentic backend in `Agentic_Reasoning/backend.py`, but it is currently **not mounted** in `main.py`.

## Setup (Windows)

### 1) Create & activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install spaCy model

```powershell
python -m spacy download en_core_web_sm
```

## Run the Application

Start the integrated server:

```powershell
python main.py
```

By default, it runs on:

- `http://localhost:8000`

## Web Pages (UI)

These are served by FastAPI routes in `main.py`:

- `/` or `/nlp` → NLP Feature Extraction (`templates/nlp.html`)
- `/anomaly` → Anomaly Detection (`templates/anomaly.html`)
- `/duplicate` → Duplicate Detection (`templates/duplicate.html`)
- `/fraud` → Fraud Network Analysis UI (`templates/fraud.html`)
- `/pipeline-page` → Full Pipeline UI (`templates/pipeline.html`)
- `/dashboard` → Dashboard (`templates/dashboard.html`)
- `/choice` → Choice / feature display UI (`templates/choice.html`)
- `/admin-decision-page` → Admin decision UI (`templates/admin_decision.html`)
- `/agentic` → Agentic reasoning UI (`templates/agentic.html`)

## API Endpoints

### FastAPI Orchestrator

- `POST /pipeline`
  - Upload a PDF and run the full pipeline.
  - Returns a combined JSON:
    - `nlp_extraction`
    - `anomaly_detection`
    - `duplicate_detection`
    - `fraud_network_analysis`

- `GET /dashboard-data`
  - Returns case stats + cases list from `admin_decisions.json`.

- `POST /generate-report`
  - Generates a PDF report (FPDF) from a case JSON payload.

### NLP Extractor (Flask; mounted under `/nlp-extractor`)

- `POST /nlp-extractor/extract`
  - Form-data file field: `pdf`

### Anomaly Detection (Flask; mounted under `/anomaly`)

- `POST /anomaly/predict`
  - JSON body with numeric features:
    - `annual_income`
    - `registrations_per_aadhaar`
    - `bank_shared_count`
    - `phone_shared_count`

### Duplicate Detection (Flask; mounted under `/duplicate`)

- `POST /duplicate/predict`
  - JSON body (examples):
    - `aadhaar_like_id`, `phone_number`, `bank_account`, `household_id`, `name`, `district`

### Fraud Network Analysis (Flask; mounted under `/fraud-network`)

- `POST /fraud-network/predict`
  - JSON body:
    - `beneficiary_id`

### Admin Decision Layer (Flask; mounted under `/admin-decision`)

- `GET /admin-decision/cases`
- `POST /admin-decision/decide`
- `GET /admin-decision/audit`
- `GET /admin-decision/init-cases`
- `POST /admin-decision/agentic-reasoning/analyze`
  - **Currently returns mock** explanation/audit summary text based on input probabilities.

## Data Flow (Typical Usage)

1. Go to **NLP Extraction** (`/nlp`) and extract features from a PDF.
2. Use the extracted features across other modules (stored client-side in `localStorage` on some pages).
3. Run **Pipeline** (`/pipeline-page`) to execute all modules automatically.
4. Review results on **Dashboard** (`/dashboard`).
5. Submit a decision via **Admin Decision** (`/admin-decision-page`).

## Models & Assets

This repo includes trained models and datasets used by the modules:

- `Anomaly_Detection/isolation_forest_model.pkl`
- `Duplicate_Detection/duplicate_detection_pipeline_xgb.pkl`
- `Fraud_Network_Analysis/fraud_network_model.pkl`
- Large CSV datasets under each module directory.

## Troubleshooting

- **spaCy model missing**
  - Run: `python -m spacy download en_core_web_sm`

- **Large first-time fraud network load**
  - Fraud network builds a graph (or loads `graph_cache.pkl`). First run may be slower.

## License

See `LICENSE`.
