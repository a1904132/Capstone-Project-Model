import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from text_graph_dataset import ReentrancyGraphDataset
from sklearn.metrics import classification_report

class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.classifier(x))


dataset = ReentrancyGraphDataset("text_graphs")
train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCNClassifier(in_channels=len(dataset.node_vocab), hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

# model training
for epoch in range(10):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch).squeeze()
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            preds = (out > 0.5).int().cpu()
            y_true.extend(batch.y.cpu().int().tolist())
            y_pred.extend(preds.tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f"Epoch {epoch+1}: Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

def generate_classification_report(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            preds = (out > 0.5).int().cpu().tolist()
            y_true.extend(batch.y.cpu().int().tolist())
            y_pred.extend(preds)

    print("\n Final Classification Report \n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Non-Reentrancy", "Reentrancy"],
        digits=2
    ))

generate_classification_report(model, val_loader)

torch.save(model.state_dict(), "gnn_model.pt")
print("GNN model saved to gnn_model.pt")
