# FYP_MultiModal-Logistics

A PyTorch-based Deep Learning approach for estimating and predicting multi-modal logistics routing metrics (cost, time, and reliability) using **Graph Attention Networks (GAT)**. This repository houses the development of **MP-GAT** (Multi-Modal Graph Attention Network), which incorporates multi-modal edges (road, rail, air, sea) and complex shipment features to effectively forecast routing consequences.

## 🌟 Key Features
- **Multi-Modal Graph Representation**: Models logistics networks where nodes are hubs (ports, airports, rail yards) and edges are transport routes possessing comprehensive attributes (distance, cost, time, reliability, carbon emissions, and mode).
- **Advanced Graph Neural Networks**: Utilizes `GATConv` with edge-feature integration to allow message passing that strictly factors in edge constraints and properties.
- **Multi-Task Regression Head**: Predicts three critical routing components simultaneously based on origin, destination, and shipment constraints:
  - **Total Cost**
  - **Total Time**
  - **Route Reliability**
- **Baseline Comparisons**: Benchmarks the proposed MP-GAT model against traditional Graph Convolutional Networks (GCN), GraphSAGE, and non-graph Multi-Layer Perceptron (MLP) baselines.

---

## 📁 Codebase Structure

| File | Description |
| :--- | :--- |
| `dataset.py` | Generates a synthetic multi-modal logistics graph using `networkx`. Implements ground-truth labeling using Dijkstra's shortest path routing (optimized for cost) and normalizes/tensorizes features for PyTorch Geometric (`PyG`). |
| `model.py` | Contains the core architecture of `MPGAT_Logistics`. It encompasses linear node/edge encoders, a multi-head GAT layer, a parallel shipment encoder, and a fusion layer leading into a multi-task regression head. |
| `baselines.py` | Contains simpler baseline architectures (`MLP_Baseline`, `GCN_Baseline`, `GraphSAGE_Baseline`) to benchmark the performance of the proposed MP-GAT. |
| `train.py` | The main training script. It handles dataset generation, dataset normalization, model initialization, weighted multi-objective loss computation, gradient clipping, and the training loop for the `MPGAT_Logistics` model. |
| `evaluate.py` | The evaluation suite. It compares the loss and performance (MSE, MAE, $R^2$ Score) of the MP-GAT model against MLP, GCN, and GraphSAGE baselines over evaluation data. |
| `main.py` | Entry point script for executing the primary training/execution workflow. |
| `requirements.txt` | Core package dependencies required to run the exact environment locally. |

---

## ⚙️ Prerequisites & Setup

### 1. Requirements

Ensure you have Python 3.8+ installed. The main dependencies used in this pipeline are:
- `torch`
- `torch-geometric`
- `scikit-learn`
- `networkx`
- `numpy`

### 2. Installation Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/007Anmol/FYP_MultiModal-Logistics.git
   cd FYP_MultiModal-Logistics
   ```

2. **Create a Virtual Environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install the top-level dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you run into issues installing `torch-geometric`, please verify the compatibility of your PyTorch version and CUDA combinations and install directly from the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).*

---

## 🚀 How to Run

### Training the MP-GAT Model
To spin up a synthetic dataset and train the `MPGAT_Logistics` network:
```bash
python train.py
```
This script will output the multi-objective weighted loss over `60` epochs, reporting validation updates every 10 epochs.

### Evaluating & Benchmarking
To compare the architecture against the baselines and view comprehensive quantitative metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and $R^2$ configurations:
```bash
python evaluate.py
```
This generates a tabular comparison in your terminal across the MLP, GCN, GraphSAGE, and MP-GAT models allowing you to evaluate the added benefit of multi-modal edge features and attention mechanisms.

---

## 🧠 Data & Model Formulations

### Attributes and Nodes:
The synthetic dataset generates logistics hubs characterized by features:
`[Latitude, Longitude, Capacity, Hub_Port, Hub_Airport, Hub_Rail, Hub_Distribution]`

### Edges (The Routes):
The network routes are strictly multi-modal (0=road, 1=rail, 2=air, 3=sea) and factor multiple real-world transport aspects simultaneously:
`[Distance, Cost, Time, Reliability, Carbon_Footprint, One_Hot_Encoded_Mode[4]]`

### Shipment Inputs:
When querying the model to find route constraints, dynamic shipment details are integrated natively:
`[Weight, Urgency, Budget, Deadline]`

### Prediction / Fusion 
Origin constraints, destination capabilities, and shipment requirements are extracted into a high-dimensional fusion vector (`phi`). The regression head predicts the final metrics of the journey (how much it will *cost*, how long it will *take*, its *survival/reliability metric*) mirroring optimal paths derived from standard Operation Optimization algorithms (Dijkstra's base truths).