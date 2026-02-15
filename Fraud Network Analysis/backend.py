from flask import Flask, request, jsonify
import pandas as pd
import joblib
import networkx as nx
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset + ML model
@st.cache_data
def load_data():
    return pd.read_csv("fraud_network_50000.csv")

@st.cache_resource
def load_model():
    return joblib.load("fraud_network_model.pkl")

df = load_data().head(1000)
model = load_model()

# Build NetworkX graph
def build_graph(data):
    cache_file = os.path.join(os.path.dirname(__file__), "graph_cache.pkl")
    
    if os.path.exists(cache_file):
        print("Loading cached graph...")
        return joblib.load(cache_file)
    else:
        print("Building graph...")
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

        print("Saving graph to cache...")
        joblib.dump(G, cache_file)
        return G

G = build_graph(df)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        selected_id = data.get('beneficiary_id')

        if not selected_id or selected_id not in df["beneficiary_id"].values:
            return jsonify({'error': 'Invalid beneficiary_id'}), 400

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

        # Fraud Ring Detection (Graph Component)
        try:
            component = next(c for c in nx.connected_components(G) if selected_id in c)
            component_size = len(component)
            fraud_ring = component_size >= 5
        except StopIteration:
            component_size = 0
            fraud_ring = False

        # Master Agent Detection (Centrality)
        try:
            degree_centrality = nx.degree_centrality(G)
            centrality_score = degree_centrality[selected_id]
            threshold = pd.Series(list(degree_centrality.values())).quantile(0.95)
            is_master_agent = centrality_score >= threshold
        except KeyError:
            centrality_score = 0.0
            is_master_agent = False

        # Prepare response
        result = {
            'beneficiary_id': selected_id,
            'ml_prediction': 'Fraud Ring Suspected' if pred == 1 else 'Appears Normal',
            'fraud_probability': round(fraud_prob, 2),
            'normal_probability': round(normal_prob, 2),
            'connected_component_size': component_size,
            'fraud_ring_detected': fraud_ring,
            'degree_centrality': round(centrality_score, 4),
            'master_agent_detected': is_master_agent,
            'beneficiary_details': {
                'aadhaar_like_id': row['aadhaar_like_id'],
                'phone_number': row['phone_number'],
                'bank_account': row['bank_account'],
                'agent_id': row['agent_id'],
                'district': row['district']
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
