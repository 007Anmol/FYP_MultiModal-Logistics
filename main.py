import time
from train import train_model

def run_system():
    # Execute full training pipeline
    trained_model, graph_data, sample_shipments = train_model()
    
    # Demonstrate Inference Speed (GAT vs Search) 
    print("\nRunning Inference Test...")
    trained_model.eval()
    
    start_time = time.time()
    _ = trained_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, sample_shipments)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print(f"Processed {len(sample_shipments)} shipment queries in {inference_time:.4f} seconds.")
    print(f"Average time per query: {(inference_time / len(sample_shipments)) * 1000:.4f} ms.")

if __name__ == "__main__":
    run_system()