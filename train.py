import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from dataset import create_synthetic_logistics_graph, generate_shipments_and_labels, tensorize_graph
from model import MPGAT_Logistics

def train_model():
    print("Generating Synthetic Graph and Routing Labels...")
    G = create_synthetic_logistics_graph(num_nodes=40, num_edges=250)
    shipments, labels = generate_shipments_and_labels(G, num_shipments=1000)
    
    # Normalize Shipments and Labels for stable training [cite: 147]
    shipment_scaler = MinMaxScaler()
    shipments[:, :4] = shipment_scaler.fit_transform(shipments[:, :4])
    
    label_scaler = MinMaxScaler()
    labels = label_scaler.fit_transform(labels)
    
    # Tensorize data
    pyg_graph = tensorize_graph(G)
    shipments_tensor = torch.tensor(shipments, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    
    # Initialize Model & Optimizer
    model = MPGAT_Logistics(node_in_dim=7, edge_in_dim=9, shipment_in_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # [cite: 216]
    
    # Loss Weights for [Cost, Time, Reliability] [cite: 190, 192]
    lambda_weights = torch.tensor([0.4, 0.4, 0.2]) 
    mse_loss = nn.MSELoss(reduction='none')

    epochs = 60 # Standard convergence frame [cite: 372]
    
    print("Starting Training...")
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.edge_attr, shipments_tensor)
        
        # Calculate Multi-objective Weighted Loss 
        raw_loss = mse_loss(predictions, labels_tensor)
        weighted_loss = (raw_loss * lambda_weights).mean()
        
        weighted_loss.backward()
        
        # Gradient Clipping [cite: 100]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Weighted Loss: {weighted_loss.item():.4f}")
            
    print("Training Complete. Model is ready for inference.")
    return model, pyg_graph, shipments_tensor

if __name__ == "__main__":
    train_model()