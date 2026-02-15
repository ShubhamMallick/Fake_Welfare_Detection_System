# =========================================================
# Fraud Network Detection Dashboard (FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import networkx as nx
import joblib
from pyvis.network import Network
import tempfile
import os
import warnings

# Suppress Arrow serialization warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore', category=UserWarning, message='.*ArrowSerializationWarning.*')

# ---------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fraud Network Detection",
    page_icon="🕸️",
    layout="wide"
)

# ---------------------------------------------------------
# Load dataset + ML model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("fraud_network_50000.csv")


@st.cache_resource
def load_model():
    return joblib.load("fraud_network_model.pkl")


df = load_data()
model = load_model()

# ---------------------------------------------------------
# Build NetworkX graph (cached)
# ---------------------------------------------------------
@st.cache_resource
def build_graph(data):
    G = nx.Graph()

    # Add nodes
    for _, row in data.iterrows():
        G.add_node(row["beneficiary_id"])

    # Helper to add edges
    def add_edges(col):
        grouped = data.groupby(col)["beneficiary_id"].apply(list)
        for group in grouped:
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        G.add_edge(group[i], group[j])

    # Shared infrastructure edges
    for column in ["phone_number", "bank_account", "agent_id", "aadhaar_like_id"]:
        add_edges(column)

    return G


G = build_graph(df)

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("🕸️ Fraud Network Detection System")
st.write(
    "Hybrid **Machine Learning + Graph Intelligence** system for "
    "detecting duplicate beneficiaries, fraud rings, and master agents."
)

st.markdown("---")

# =========================================================
# 1️⃣ Beneficiary Search
# =========================================================
st.header("🔍 Beneficiary Fraud Check")

beneficiary_list = df["beneficiary_id"].values

# Add example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("👨‍💼 Test Non-Fraud Example"):
        # Find a beneficiary with low risk characteristics
        non_fraud_example = df[df["fraud_ring_member"] == 0].iloc[0]["beneficiary_id"]
        st.session_state.selected_id = non_fraud_example
with col2:
    if st.button("🕵️ Test Fraud Example"):
        # Find a beneficiary with high risk characteristics
        fraud_example = df[df["fraud_ring_member"] == 1].iloc[0]["beneficiary_id"]
        st.session_state.selected_id = fraud_example

# Initialize session state if not exists
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = beneficiary_list[0]

# Use the selected ID from session state or the dropdown
selected_id = st.selectbox(
    "Select Beneficiary ID",
    beneficiary_list,
    index=beneficiary_list.tolist().index(st.session_state.selected_id)
)

# Update session state when dropdown changes
if selected_id != st.session_state.selected_id:
    st.session_state.selected_id = selected_id

row = df[df["beneficiary_id"] == selected_id].iloc[0]

# ML prediction using degree features
input_df = pd.DataFrame([{
    "phone_degree": int(row["phone_degree"]),
    "bank_degree": int(row["bank_degree"]),
    "household_size": int(row["household_size"]),
    "agent_cluster_size": int(row["agent_cluster_size"])
}])

pred = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

fraud_prob = prob * 100
normal_prob = (1 - prob) * 100

col1, col2 = st.columns(2)

with col1:
    if pred == 1:
        st.error(f"⚠️ Fraud Ring Suspected\n\nRisk: **{fraud_prob:.2f}%**")
    else:
        st.success(f"✅ Appears Normal\n\nConfidence: **{normal_prob:.2f}%**")

with col2:
    st.write("### Beneficiary Details")
    st.write(row[[
        "aadhaar_like_id",
        "phone_number",
        "bank_account",
        "agent_id",
        "district"
    ]])

st.markdown("---")

# =========================================================
# 2️⃣ Fraud Ring Detection (Graph Component)
# =========================================================
st.header("🧩 Fraud Ring Analysis")

component = next(c for c in nx.connected_components(G) if selected_id in c)
component_size = len(component)

st.write(f"**Connected Component Size:** {component_size}")

if component_size >= 5:
    st.error("⚠️ This beneficiary belongs to a FRAUD RING")
else:
    st.success("✅ No fraud-ring behavior detected")

st.markdown("---")

# =========================================================
# 3️⃣ Master Agent Detection (Centrality)
# =========================================================
st.header("👑 Master Agent Detection")

degree_centrality = nx.degree_centrality(G)
centrality_score = degree_centrality[selected_id]

threshold = pd.Series(degree_centrality).quantile(0.95)

st.write(f"**Degree Centrality:** {centrality_score:.4f}")

if centrality_score >= threshold:
    st.error("🚨 Possible MASTER AGENT detected")
else:
    st.success("Normal network influence")

st.markdown("---")

# =========================================================
# 4️⃣ Interactive Fraud Network Visualization (PyVis)
# =========================================================
st.header("🌐 Fraud Network Visualization")

show_graph = st.checkbox("Show connected fraud network")

if show_graph:

    sub_nodes = list(component)[:50]  # limit for performance
    subgraph = G.subgraph(sub_nodes)

    net = Network(height="500px", width="100%", notebook=False)

    for node in subgraph.nodes():
        color = "red" if node == selected_id else "skyblue"
        net.add_node(node, label=node, color=color)

    for edge in subgraph.edges():
        net.add_edge(edge[0], edge[1])

    # Save to temporary file and display
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
    net.save_graph(tmp_file.name)
    tmp_file.close()  # Close the file handle before reading it back
    
    # Display in Streamlit
    try:
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.components.v1.html(html_data, height=520)
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file.name)
        except (PermissionError, FileNotFoundError):
            pass  # File will be cleaned up by OS eventually

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("AI-Powered Fraud Network Detection • ML + Graph Intelligence • Hackathon Final Demo")
