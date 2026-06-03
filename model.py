import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MPGAT_Logistics(nn.Module):
    def __init__(self, node_in_dim=7, edge_in_dim=9, shipment_in_dim=4, hidden_dim=64, heads=4):
        super(MPGAT_Logistics, self).__init__()
        
        # 1. Linear Encoders [cite: 168]
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        # 2. Multi-head GAT Layer [cite: 178]
        self.gat = GATConv(in_channels=hidden_dim, 
                           out_channels=hidden_dim // heads, 
                           heads=heads, 
                           edge_dim=hidden_dim, 
                           concat=True)
        
        # 3. Shipment Fusion Module [cite: 181]
        self.shipment_encoder = nn.Sequential(
            nn.Linear(shipment_in_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 4. Multi-task Regression Head (Cost, Time, Reliability) [cite: 187]
        # Input size: origin_node(hidden_dim) + dest_node(hidden_dim) + shipment(hidden_dim)
        fusion_dim = hidden_dim * 3 
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) 
        )

    def forward(self, x, edge_index, edge_attr, shipments):
        # Encode inputs
        h = F.relu(self.node_encoder(x))
        e = F.relu(self.edge_encoder(edge_attr))
        
        # Message passing with Graph Attention
        h = self.gat(h, edge_index, edge_attr=e)
        
        # Process shipments in parallel
        # Shipments format: [weight, urgency, budget, deadline, origin_idx, dest_idx]
        shipment_features = shipments[:, :4]
        origins = shipments[:, 4].long()
        destinations = shipments[:, 5].long()
        
        z = self.shipment_encoder(shipment_features)
        
        # Extract embeddings for routing pairs [cite: 184]
        h_orig = h[origins]
        h_dest = h[destinations]
        
        # Fusion [cite: 184]
        phi = torch.cat([h_orig, h_dest, z], dim=1)
        
        # Predict: [Cost, Time, Reliability] [cite: 187]
        predictions = self.regression_head(phi)
        
        return predictions