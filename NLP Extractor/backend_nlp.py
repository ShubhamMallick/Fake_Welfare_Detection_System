from flask import Flask, request, jsonify
import re
import spacy
import pdfplumber
import pandas as pd
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load spaCy model
@st.cache_resource  # But for Flask, load once
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        raise Exception("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")

nlp = load_spacy_model()

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

# Define patterns for extraction
PATTERNS = {
    "Aadhaar ID": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    "Phone Number": r"\b[6-9]\d{9}\b",
    "Bank Account": r"\b\d{9,18}\b",
    "Beneficiary ID": r"\bBEN\d{4,12}\b",
    "Household ID": r"\bHH\d{3,12}\b",
    "Age": r"\b(1[89]|[2-9]\d)\b",
    "Annual Income": r"(?:₹|Rs\.?)\s?\d{2,7}(?:,\d{3})*"
}

def regex_extract(text):
    """Extract information using regex patterns."""
    results = {}
    for field, pattern in PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            results[field] = list(set(matches))  # Remove duplicates
    return results

def spacy_extract(text):
    """Extract entities using spaCy."""
    doc = nlp(text)
    results = {
        "Name": [],
        "Location": [],
        "Organization": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            results["Name"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            results["Location"].append(ent.text)
        elif ent.label_ == "ORG":
            results["Organization"].append(ent.text)
    
    # Remove duplicates
    for key in results:
        results[key] = list(set(results[key]))
    
    return results

@app.route('/extract', methods=['POST'])
def extract():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        uploaded_file = request.files['pdf']
        if uploaded_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not uploaded_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF'}), 400

        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        if not text:
            return jsonify({'error': 'Could not extract text from the uploaded file'}), 400

        # Extract information
        regex_results = regex_extract(text)
        nlp_results = spacy_extract(text)

        # Prepare response
        result = {
            'regex_results': regex_results,
            'nlp_results': nlp_results,
            'raw_text': text
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
