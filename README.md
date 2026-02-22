# Prayatna – Fake Welfare / Fraud Detection System

An integrated, modular fraud-detection system for welfare/beneficiary programs. The project combines multiple ML modules (NLP extraction, anomaly detection, duplicate detection, fraud-network analysis) with AI-powered agentic reasoning behind a single **FastAPI** gateway and provides a modern multi-page web UI with voice navigation capabilities.

## 🚀 Key Features

### Core Detection Modules
- **📄 NLP Feature Extraction (PDF)**
  - Extracts structured features from beneficiary PDFs using regex + spaCy NER
  - Supports multiple document formats
- **📊 Anomaly Detection**
  - IsolationForest-based anomaly scoring for fraud risk patterns
  - Real-time fraud probability assessment
- **🔄 Duplicate Detection**
  - XGBoost pipeline for detecting likely duplicate/linked beneficiary registrations
  - Advanced fuzzy matching algorithms
- **🕸️ Fraud Network Analysis**
  - NetworkX graph + ML model to detect fraud rings and high-centrality "master agent" behavior
  - Visual network analysis with graph caching

### 🤖 AI-Powered Features
- **🧠 Agentic Reasoning**
  - Integrated OpenAI/LangChain-powered analysis
  - Automated case explanation and audit summary generation
  - Intelligent fraud pattern recognition
- **🎙️ Voice Navigation**
  - Hands-free operation with voice commands
  - Fuzzy matching for natural language input
  - Section navigation and pipeline control

### User Interface
- **🎨 Modern Web UI**
  - Responsive design with glassmorphism effects
  - Animated navigation bar with hover effects
  - Voice-controlled interface
- **📈 Pipeline Orchestration**
  - Upload one PDF and run NLP → anomaly → duplicate → fraud-network → agentic reasoning sequentially
  - Real-time progress tracking
- **📊 Interactive Dashboard**
  - Visual + tabular overview of cases and scores
  - Admin decision tracking and audit trails
- **⚙️ Admin Decision Layer**
  - Stores admin decisions and maintains audit trail in `admin_decisions.json`
  - Automated report generation

## 🏗️ Project Structure

```
Prayatna/
├── main.py                           # FastAPI main application
├── requirements.txt                  # Python dependencies
├── admin_decisions.json              # Admin decision storage
├── README.md                         # This file
│
├── templates/                        # Jinja2 HTML templates
│   ├── pipeline.html                 # Main pipeline interface with voice nav
│   ├── nlp.html                      # NLP extraction UI
│   ├── anomaly.html                  # Anomaly detection UI
│   ├── duplicate.html                # Duplicate detection UI
│   ├── fraud.html                    # Fraud network analysis UI
│   ├── agentic.html                  # Agentic reasoning UI
│   ├── dashboard.html                # Dashboard overview
│   ├── choice.html                   # Feature selection UI
│   └── admin_decision.html           # Admin decision interface
│
├── NLP_Extractor/
│   ├── backend_nlp.py               # Flask NLP service
│   ├── NLP_Extractor.py             # Core NLP logic
│   └── Document*.pdf/docx           # Sample documents
│
├── Anomaly_Detection/
│   ├── backend.py                   # Flask anomaly service
│   ├── isolation_forest_model.pkl   # Trained ML model
│   └── Anomaly_Detection_50000.csv  # Training dataset
│
├── Duplicate_Detection/
│   ├── backend.py                   # Flask duplicate service
│   ├── duplicate_detection_pipeline_xgb.pkl  # XGBoost model
│   └── duplicate_detection_50000_v4.csv      # Training data
│
├── Fraud_Network_Analysis/
│   ├── backend.py                   # Flask fraud network service
│   ├── fraud_network_model.pkl      # Network analysis model
│   ├── fraud_network_50000.csv      # Network dataset
│   └── graph_cache.pkl              # Cached network graphs
│
├── Admin_Decision_Layer/
│   └── backend.py                   # Flask admin decision service
│
├── Agentic_Reasoning/
│   ├── backend.py                   # Flask agentic reasoning service
│   ├── Explaining_Suspicious_Cases.ipynb    # Explanation logic
│   └── Audit_Summary_Generator.ipynb         # Audit generation
│
└── venv/                           # Virtual environment (not in repo)
```

## 🏛️ Architecture Overview

- **FastAPI** (`main.py`) serves as the primary entrypoint and API gateway
- Individual module backends are implemented as **Flask applications** and mounted into FastAPI using **`WSGIMiddleware`**
- The browser UI is served via **Jinja2 templates** from the `templates/` directory
- **Voice navigation** powered by Web Speech API with fuzzy matching
- **Agentic reasoning** integrated with OpenAI API for intelligent analysis

### 🔗 Mounted Backends (FastAPI → Flask)

`main.py` mounts these Flask applications:

- `/nlp-extractor` → `NLP_Extractor/backend_nlp.py`
- `/anomaly` → `Anomaly_Detection/backend.py`
- `/duplicate` → `Duplicate_Detection/backend.py`
- `/fraud-network` → `Fraud_Network_Analysis/backend.py`
- `/admin-decision` → `Admin_Decision_Layer/backend.py`
- `/agentic-reasoning` → `Agentic_Reasoning/backend.py` *(Now fully integrated!)*

## ⚙️ Installation & Setup (Windows)

### 1) Create & Activate Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2) Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install spaCy Language Model

```powershell
python -m spacy download en_core_web_sm
```

### 4) Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # or your preferred endpoint
```

## � Docker Deployment (Recommended)

The application now includes full Docker support for easy deployment and sharing.

### Prerequisites
- Docker Desktop installed and running
- Git (for version control)

### Quick Start with Docker

#### 1) Clone Repository
```bash
git clone https://github.com/yourusername/Prayatna.git
cd Prayatna
```

#### 2) Configure Environment
Create `.env` file:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_BASE_URL=https://openrouter.ai/api/v1" >> .env
```

#### 3) Build and Run
```bash
# Build Docker image
docker-compose build

# Start application
docker-compose up -d

# Check status
docker-compose ps
```

#### 4) Access Application
- **Main Interface:** http://localhost:8000/pipeline-page
- **API Docs:** http://localhost:8000/docs

### Docker Commands Reference

```bash
# Build image
docker-compose build

# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Restart
docker-compose restart

# Clean rebuild
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

### Project Structure (Updated)

```
Prayatna/
├── main.py                           # FastAPI main application
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker image configuration
├── docker-compose.yml               # Docker Compose configuration
├── .dockerignore                    # Docker ignore file
├── admin_decisions.json              # Admin decision storage
├── README.md                         # This file
│
├── templates/                        # Jinja2 HTML templates
│   ├── pipeline.html                 # Main pipeline interface with voice nav
│   ├── nlp.html                      # NLP extraction UI
│   └── ...                           # Other templates
│
├── NLP_Extractor/                    # NLP processing module
├── Anomaly_Detection/               # Anomaly detection module
├── Duplicate_Detection/             # Duplicate detection module
├── Fraud_Network_Analysis/          # Network analysis module
├── Admin_Decision_Layer/           # Admin decision module
└── Agentic_Reasoning/              # AI reasoning module
```

## 🌐 Live Deployment Options

Deploy your Docker application to live servers for sharing with others.

### 🚀 Render (Free, Recommended)

1. **Sign up:** Go to [render.com](https://render.com)
2. **Connect:** Link your GitHub repository
3. **Configure:**
   - Runtime: `Docker`
   - Service Name: `prayatna-fraud-detection`
   - Environment Variables: `OPENAI_API_KEY`, `OPENAI_BASE_URL`
4. **Deploy:** Click "Create Web Service"
5. **Access:** `https://your-app.onrender.com/pipeline-page`

### 🚂 Railway

1. **Sign up:** [railway.app](https://railway.app)
2. **Connect:** GitHub repository
3. **Auto-deploy:** Railway detects Docker automatically
4. **Configure:** Add environment variables
5. **Access:** `https://your-project.up.railway.app/pipeline-page`

### Current Deployment Status

**✅ Successfully Deployed on Railway:** The application is live and operational on Railway, supporting large Docker images (3.17GB+).

- **Status:** Live and healthy
- **URL:** https://fakewelfaredetectionsystem-production.up.railway.app/pipeline-page
- **Features:** Complete fraud detection pipeline with NLP extraction, anomaly detection, duplicate detection, fraud network analysis, and AI-powered agentic reasoning
- **Voice Navigation:** Hands-free operation with voice commands
- **Environment:** Production-ready with OpenAI API integration

### 🏠 VPS Deployment

For full control, deploy to a VPS (DigitalOcean, $5/month):

```bash
# On your VPS server
sudo apt update
sudo apt install docker.io docker-compose
git clone https://github.com/yourusername/Prayatna.git
cd Prayatna
docker-compose up -d
```

### 🔗 Share with Friends

**Temporary Sharing (5 minutes):**
```bash
# Install ngrok
npm install -g ngrok
ngrok http 8000
# Share the https://abc123.ngrok.io URL
```

**Permanent Sharing:** Use Render/Railway deployment above.

## �🚀 Running the Application

Start the integrated FastAPI server using uvicorn:

```powershell
uvicorn main:app --reload
```

The application runs on: **`http://localhost:8000`**

For production deployment without auto-reload:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🌐 Web Interface

### Main Pages
- **`/`** → Home/Dashboard redirect
- **`/pipeline-page`** → Full pipeline interface with voice navigation *(Main UI)*
- **`/dashboard`** → Analytics dashboard with case overview
- **`/admin-decision-page`** → Admin decision interface

### Module-Specific Pages
- **`/nlp`** → NLP Feature Extraction
- **`/anomaly`** → Anomaly Detection
- **`/duplicate`** → Duplicate Detection
- **`/fraud`** → Fraud Network Analysis
- **`/agentic`** → Agentic Reasoning (AI Analysis)

### Voice Navigation Commands 🎙️

The main pipeline page (`/pipeline-page`) supports voice commands:

#### **Special Commands:**
- `"run basic pipeline"` → Execute fraud detection pipeline
- `"run enhanced reasoning"` / `"run agentic reasoning"` → AI-powered analysis
- `"start listening"` → Activate voice navigation
- `"stop listening"` / `"exit"` → Deactivate voice navigation
- `"help"` → Show available commands

#### **Navigation Commands:**
- `"go to overview"` / `"show dashboard"` / `"main page"`
- `"go to nlp"` / `"natural language processing"` / `"text analysis"`
- `"go to anomaly"` / `"anomaly detection"` / `"find anomalies"`
- `"go to duplicate"` / `"duplicate detection"` / `"find duplicates"`
- `"go to fraud"` / `"fraud analysis"` / `"fraud network"`
- `"go to integrated"` / `"full analysis"` / `"results"`
- `"go to agentic"` / `"agentic reasoning"` / `"ai analysis"`

**Example Usage:** Say *"go to overview"* or *"run basic pipeline"* while on the pipeline page.

## 📡 API Endpoints

### FastAPI Orchestrator

- **`POST /pipeline`**
  - Upload PDF and run complete pipeline
  - Returns: `nlp_extraction`, `anomaly_detection`, `duplicate_detection`, `fraud_network_analysis`

- **`GET /dashboard-data`**
  - Returns case statistics and admin decisions from `admin_decisions.json`

- **`POST /generate-report`**
  - Generates PDF report from case data using FPDF

### Module APIs (Flask, mounted under respective paths)

#### NLP Extractor (`/nlp-extractor`)
- **`POST /nlp-extractor/extract`** → Extract features from uploaded PDF

#### Anomaly Detection (`/anomaly`)
- **`POST /anomaly/predict`** → Score fraud probability from features

#### Duplicate Detection (`/duplicate`)
- **`POST /duplicate/predict`** → Detect duplicate registrations

#### Fraud Network Analysis (`/fraud-network`)
- **`POST /fraud-network/predict`** → Analyze beneficiary networks

#### Admin Decision Layer (`/admin-decision`)
- **`GET /admin-decision/cases`** → Get case list
- **`POST /admin-decision/decide`** → Submit admin decision
- **`GET /admin-decision/audit`** → Get audit trail
- **`GET /admin-decision/init-cases`** → Initialize sample cases
- **`POST /admin-decision/agentic-reasoning/analyze`** → AI-powered case analysis

## 🔄 Workflow (Typical Usage)

1. **Navigate** to Pipeline page (`/pipeline-page`)
2. **Upload** a beneficiary PDF document
3. **Run** basic pipeline: NLP → Anomaly → Duplicate → Fraud Network
4. **Run** agentic reasoning for AI-powered analysis and recommendations
5. **Review** results on interactive dashboard
6. **Submit** admin decisions with audit trail
7. **Generate** automated reports

### Voice-Controlled Workflow 🎙️
- **Say** *"start listening"* to activate voice navigation
- **Say** *"run basic pipeline"* to execute analysis
- **Say** *"go to [section]"* to navigate between results
- **Say** *"run enhanced reasoning"* for AI analysis

## 🤖 AI Features & Configuration

### Agentic Reasoning
- **OpenAI Integration**: GPT-powered case analysis and explanations
- **LangChain**: Structured prompts for consistent AI responses
- **Error Handling**: Comprehensive rate limit and API error management
- **Fallback Responses**: Graceful degradation when AI services unavailable

### Voice Navigation
- **Web Speech API**: Browser-native speech recognition
- **Fuzzy Matching**: Intelligent command interpretation
- **Multi-language Support**: Extensible for additional languages
- **Accessibility**: Hands-free operation for improved UX

## 📊 Models & Data Assets

Pre-trained models and datasets included:

- **Anomaly Detection**: `isolation_forest_model.pkl`
- **Duplicate Detection**: `duplicate_detection_pipeline_xgb.pkl`
- **Fraud Network**: `fraud_network_model.pkl`
- **Training Data**: Large CSV datasets (50k+ samples each)
- **Network Cache**: `graph_cache.pkl` for performance

## 🔧 Troubleshooting

### Common Issues
- **spaCy Model Missing** → Run: `python -m spacy download en_core_web_sm`
- **OpenAI API Errors** → Check API key and rate limits
- **Voice Recognition Not Working** → Ensure HTTPS or localhost, check browser permissions
- **Large Initial Load** → Fraud network builds graphs on first run (may take time)
- **Memory Issues** → Reduce batch sizes in configuration if needed

### Performance Tips
- Use the cached network graphs for faster subsequent runs
- Process documents in smaller batches for memory efficiency
- Voice commands work best in quiet environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit with descriptive messages
5. Push to your fork and create a Pull Request

## 📄 License

See `LICENSE` file for details.

## 🙏 Acknowledgments

- Built with FastAPI, Flask, and modern web technologies
- Powered by OpenAI GPT and LangChain
- ML models trained on comprehensive welfare datasets
- Voice navigation using Web Speech API

---

**Prayatna** - *Committed to transparent and efficient welfare fraud detection* 🛡️
