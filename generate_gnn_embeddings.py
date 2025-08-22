import torch
import numpy as np
from torch_geometric.loader import DataLoader
from text_graph_dataset import ReentrancyGraphDataset  
from train_gnn import GCNClassifier 
from torch_geometric.nn import global_mean_pool

graph_dir = "text_graphs"
dataset = ReentrancyGraphDataset(graph_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# load trained GNN from train_gnn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNClassifier(in_channels=len(dataset.node_vocab), hidden_channels=64).to(device)
model.load_state_dict(torch.load("gnn_model.pt")) 
model.eval()

#generate embeddings
embeddings = []
skipped_indices = []


with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.edge_index.numel() == 0:
            skipped_indices.append(i)
            print(f"Skipping graph {i} with no edges.")
            continue
        x = model.conv1(batch.x, batch.edge_index).relu()
        x = model.conv2(x, batch.edge_index).relu()
        pooled = global_mean_pool(x, batch.batch)
        embeddings.append(pooled.cpu().numpy())


# Save embeddings
gnn_embeddings = np.vstack(embeddings)
np.save("gnn_embeddings.npy", gnn_embeddings)
np.save("skipped_indices.npy", np.array(skipped_indices))

print(f"Saved GNN embeddings to gnn_embeddings.npy — shape: {gnn_embeddings.shape}")

