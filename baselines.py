import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

# Baseline 1: Multi-Layer Perceptron (Tabular Only)
class MLP_Baseline(nn.Module):
    def __init__(self, node_in_dim=7, shipment_in_dim=4, hidden_dim=64):
        super(MLP_Baseline, self).__init__()
        # Concatenates origin node features, destination node features, and shipment constraints
        input_dim = (node_in_dim * 2) + shipment_in_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # Predicts Cost, Time, Reliability
        )

    def forward(self, x, shipments):
        shipment_features = shipments[:, :4]
        origins = shipments[:, 4].long()
        destinations = shipments[:, 5].long()
        
        # Access raw node features without message passing
        x_orig = x[origins]
        x_dest = x[destinations]
        
        phi = torch.cat([x_orig, x_dest, shipment_features], dim=1)
        return self.mlp(phi)

# Baseline 2: Graph Convolutional Network (Uniform Aggregation)
class GCN_Baseline(nn.Module):
    def __init__(self, node_in_dim=7, shipment_in_dim=4, hidden_dim=64):
        super(GCN_Baseline, self).__init__()
        self.conv1 = GCNConv(node_in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.shipment_encoder = nn.Linear(shipment_in_dim, hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, edge_index, shipments):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        
        shipment_features = shipments[:, :4]
        origins = shipments[:, 4].long()
        destinations = shipments[:, 5].long()
        
        z = F.relu(self.shipment_encoder(shipment_features))
        h_orig = h[origins]
        h_dest = h[destinations]
        
        phi = torch.cat([h_orig, h_dest, z], dim=1)
        return self.regression_head(phi)

# Baseline 3: GraphSAGE (Static Pooling)
class GraphSAGE_Baseline(nn.Module):
    def __init__(self, node_in_dim=7, shipment_in_dim=4, hidden_dim=64):
        super(GraphSAGE_Baseline, self).__init__()
        self.sage1 = SAGEConv(node_in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.shipment_encoder = nn.Linear(shipment_in_dim, hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x, edge_index, shipments):
        h = F.relu(self.sage1(x, edge_index))
        h = F.relu(self.sage2(h, edge_index))
        
        shipment_features = shipments[:, :4]
        origins = shipments[:, 4].long()
        destinations = shipments[:, 5].long()
        
        z = F.relu(self.shipment_encoder(shipment_features))
        h_orig = h[origins]
        h_dest = h[destinations]
        
        phi = torch.cat([h_orig, h_dest, z], dim=1)
        return self.regression_head(phi)