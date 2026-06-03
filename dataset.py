import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

def create_synthetic_logistics_graph(num_nodes=25, num_edges=100):
    G = nx.DiGraph()
    
    # 1. Node Attributes 
    # Features: [lat, lon, capacity, hub_port, hub_airport, hub_rail, hub_dist]
    for i in range(num_nodes):
        G.add_node(i, 
                   lat=np.random.uniform(-90, 90),
                   lon=np.random.uniform(-180, 180),
                   capacity=np.random.uniform(100, 10000),
                   hub_port=np.random.choice([0, 1]),
                   hub_airport=np.random.choice([0, 1]),
                   hub_rail=np.random.choice([0, 1]),
                   hub_dist=np.random.choice([0, 1]))

    # 2. Edge Attributes (Multi-modal) 
    # Modes: 0=road, 1=rail, 2=air, 3=sea
    nodes = list(G.nodes())
    for _ in range(num_edges):
        u, v = np.random.choice(nodes, 2, replace=False)
        mode = np.random.randint(0, 4)
        
        G.add_edge(u, v,
                   dist=np.random.uniform(10, 2000),
                   cost=np.random.uniform(0.1, 50.0),
                   time=np.random.uniform(1, 100),
                   reliability=np.random.uniform(0.8, 0.999),
                   carbon=np.random.uniform(5, 500),
                   mode=mode)
    return G

def generate_shipments_and_labels(G, num_shipments=500):
    shipments = []
    labels = []
    
    nodes = list(G.nodes())
    
    for _ in range(num_shipments):
        origin, dest = np.random.choice(nodes, 2, replace=False)
        
        # Shipment Features: [weight, urgency, budget, deadline, origin, dest] [cite: 144]
        weight = np.random.uniform(1, 1000)
        urgency = np.random.uniform(0, 1)
        budget = np.random.uniform(500, 5000)
        deadline = np.random.uniform(24, 720)
        
        try:
            # 3. Ground-Truth Routing Oracle (Dijkstra) [cite: 149]
            # Optimizing for cost as primary weight
            path = nx.shortest_path(G, source=origin, target=dest, weight='cost')
            
            total_cost = 0
            total_time = 0
            total_reliability = 1.0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = G[u][v]
                total_cost += edge_data['cost'] * weight
                total_time += edge_data['time']
                total_reliability *= edge_data['reliability'] # Product of reliability [cite: 151]
            
            shipments.append([weight, urgency, budget, deadline, origin, dest])
            labels.append([total_cost, total_time, total_reliability])
            
        except nx.NetworkXNoPath:
            continue # Skip if no path exists
            
    return np.array(shipments), np.array(labels)

def tensorize_graph(G):
    node_features = []
    for i in G.nodes():
        d = G.nodes[i]
        node_features.append([d['lat'], d['lon'], d['capacity'], 
                              d['hub_port'], d['hub_airport'], d['hub_rail'], d['hub_dist']])
    
    edge_index = []
    edge_features = []
    
    for u, v, d in G.edges(data=True):
        edge_index.append([u, v])
        
        # One-hot encode the transport mode [cite: 141]
        mode_onehot = [0, 0, 0, 0]
        mode_onehot[d['mode']] = 1
        
        feat = [d['dist'], d['cost'], d['time'], d['reliability'], d['carbon']] + mode_onehot
        edge_features.append(feat)
        
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Feature Normalization [cite: 161]
    scaler = MinMaxScaler()
    x = torch.tensor(scaler.fit_transform(x), dtype=torch.float)
    edge_attr = torch.tensor(scaler.fit_transform(edge_attr), dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)