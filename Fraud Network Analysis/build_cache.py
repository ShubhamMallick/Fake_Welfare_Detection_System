import pandas as pd
import networkx as nx
import joblib

df = pd.read_csv("fraud_network_50000.csv")

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

print("Building graph...")
G = build_graph(df)

cache_file = "graph_cache.pkl"
print("Saving graph to cache...")
joblib.dump(G, cache_file)

print("Graph cached successfully.")
