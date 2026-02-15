import streamlit as st
import re
import spacy
import pdfplumber
import pandas as pd
from io import BytesIO
import tempfile
import os
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="Document Information Extractor",
    page_icon="📄",
    layout="wide"
)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it using:")
        st.code("python -m spacy download en_core_web_sm")
        st.stop()

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
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

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

def main():
    st.title("📄 Document Information Extractor")
    st.write("Upload a PDF document to extract structured information")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text
            text = extract_text_from_pdf(uploaded_file)
            
            if text:
                # Extract information
                regex_results = regex_extract(text)
                nlp_results = spacy_extract(text)
                
                # Display results in tabs
                tab1, tab2 = st.tabs(["📋 Extracted Data", "🔍 Raw Text"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔍 Core Information")
                        if regex_results:
                            for field, values in regex_results.items():
                                if values:
                                    with st.expander(f"{field} ({len(values)})", expanded=True):
                                        for value in values:
                                            st.code(value)
                        else:
                            st.info("No core information found.")
                    
                    with col2:
                        st.subheader("👤 Entity Recognition")
                        if any(nlp_results.values()):
                            for field, values in nlp_results.items():
                                if values:
                                    with st.expander(f"{field} ({len(values)})", expanded=True):
                                        for value in values:
                                            st.code(value)
                        else:
                            st.info("No named entities found.")
                
                with tab2:
                    st.subheader("Extracted Text")
                    st.text_area("Raw text from the document", text, height=400)
                
                # Prepare data for download
                download_data = []
                max_len = max(
                    len(v) for d in [regex_results, nlp_results] 
                    for v in d.values()
                ) or 1
                
                for i in range(max_len):
                    row = {}
                    for d in [regex_results, nlp_results]:
                        for key, values in d.items():
                            row[key] = values[i] if i < len(values) else ""
                    download_data.append(row)
                
                # Download button
                st.download_button(
                    label="📥 Download as CSV",
                    data=pd.DataFrame(download_data).to_csv(index=False).encode('utf-8'),
                    file_name="extracted_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Could not extract text from the uploaded file.")

if __name__ == "__main__":
    main()