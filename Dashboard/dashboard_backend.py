from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import networkx as nx
import os

app = Flask(__name__)
CORS(app)

# Load data from all modules
# Fraud Network Analysis
fraud_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'Fraud_Network_Analysis', 'fraud_network_50000.csv'))
# Load graph if cached
graph_cache = os.path.join(os.path.dirname(__file__), '..', 'Fraud_Network_Analysis', 'graph_cache.pkl')
if os.path.exists(graph_cache):
    import joblib
    G = joblib.load(graph_cache)
else:
    # Build graph roughly
    G = nx.Graph()
    for _, row in fraud_df.iterrows():
        G.add_node(row["beneficiary_id"])
    # Add edges (simplified)
    for col in ["phone_number", "bank_account", "agent_id", "aadhaar_like_id"]:
        grouped = fraud_df.groupby(col)["beneficiary_id"].apply(list)
        for group in grouped:
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        G.add_edge(group[i], group[j])

# Anomaly Detection - assume similar df
anomaly_df = fraud_df  # Placeholder

# Duplicate Detection - assume similar
duplicate_df = fraud_df

# NLP - no df
# Agentic - no df

@app.route('/stats')
def get_stats():
    total_beneficiaries = len(fraud_df)
    fraud_rings = sum(1 for c in nx.connected_components(G) if len(c) >= 5)
    anomalies = (fraud_df['fraud_ring_member'] == 1).sum()
    duplicates = (fraud_df['aadhaar_count'] > 1).sum()
    high_risk = (fraud_df['phone_degree'] > 3).sum()
    medium_risk = ((fraud_df['phone_degree'] <= 3) & (fraud_df['phone_degree'] > 1)).sum()
    low_risk = (fraud_df['phone_degree'] <= 1).sum()
    
    return jsonify({
        'total_beneficiaries': total_beneficiaries,
        'fraud_rings': fraud_rings,
        'anomalies': anomalies,
        'duplicates': duplicates,
        'risk_distribution': {
            'high': high_risk,
            'medium': medium_risk,
            'low': low_risk
        }
    })

@app.route('/fraud-network')
def get_fraud_network():
    components = [len(c) for c in nx.connected_components(G)]
    component_counts = pd.Series(components).value_counts().to_dict()
    return jsonify(component_counts)

@app.route('/anomalies')
def get_anomalies():
    # Placeholder
    return jsonify({'anomalies': anomalies})

@app.route('/duplicates')
def get_duplicates():
    # Placeholder
    return jsonify({'duplicates': duplicates})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
