import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from dataset import create_synthetic_logistics_graph, generate_shipments_and_labels, tensorize_graph
from model import MPGAT_Logistics
from baselines import MLP_Baseline, GCN_Baseline, GraphSAGE_Baseline

def evaluate_models():
    print("Preparing Evaluation Dataset...")
    G = create_synthetic_logistics_graph(num_nodes=40, num_edges=250)
    shipments, labels = generate_shipments_and_labels(G, num_shipments=500)
    
    shipment_scaler = MinMaxScaler()
    shipments[:, :4] = shipment_scaler.fit_transform(shipments[:, :4])
    label_scaler = MinMaxScaler()
    labels = label_scaler.fit_transform(labels)
    
    pyg_graph = tensorize_graph(G)
    shipments_tensor = torch.tensor(shipments, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    # Initialize all models
    models = {
        "MLP": MLP_Baseline(node_in_dim=7, shipment_in_dim=4),
        "GCN": GCN_Baseline(node_in_dim=7, shipment_in_dim=4),
        "GraphSAGE": GraphSAGE_Baseline(node_in_dim=7, shipment_in_dim=4),
        "MP-GAT": MPGAT_Logistics(node_in_dim=7, edge_in_dim=9, shipment_in_dim=4)
    }

    results = []
    epochs = 30 

    print("Training and Evaluating Models...")
    for name, model in models.items():
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Pass appropriate arguments based on model constraints
            if name == "MLP":
                preds = model(pyg_graph.x, shipments_tensor)
            elif name in ["GCN", "GraphSAGE"]:
                preds = model(pyg_graph.x, pyg_graph.edge_index, shipments_tensor)
            else: # MP-GAT utilizes edge attributes
                preds = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr, shipments_tensor)
            
            loss = criterion(preds, labels_tensor)
            loss.backward()
            optimizer.step()
        
        # Execute Evaluation
        model.eval()
        with torch.no_grad():
            if name == "MLP":
                final_preds = model(pyg_graph.x, shipments_tensor)
            elif name in ["GCN", "GraphSAGE"]:
                final_preds = model(pyg_graph.x, pyg_graph.edge_index, shipments_tensor)
            else:
                final_preds = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr, shipments_tensor)
            
            y_true = labels_tensor.numpy()
            y_pred = final_preds.numpy()
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results.append({"Model": name, "MSE": mse, "MAE": mae, "R2": r2})

    # Output Results
    print("\nEvaluation Metrics Comparison")
    print("-" * 55)
    print(f"{'Model':<15} | {'MSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 55)
    for res in results:
        print(f"{res['Model']:<15} | {res['MSE']:<10.4f} | {res['MAE']:<10.4f} | {res['R2']:<10.4f}")

if __name__ == "__main__":
    evaluate_models()