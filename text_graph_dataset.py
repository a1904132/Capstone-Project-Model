import os
import json
import torch
from torch_geometric.data import Data, Dataset
from collections import defaultdict

class ReentrancyGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
        self.node_vocab = self.build_vocab()

    def build_vocab(self):
        vocab = defaultdict(lambda: len(vocab))
        for file in self.files:
            with open(os.path.join(self.root_dir, file)) as f:
                data = json.load(f)
                for node in data['nodes']:
                    _ = vocab[node]
        return dict(vocab)

    def encode_nodes(self, nodes):
        indices = [self.node_vocab[n] for n in nodes]
        x = torch.eye(len(self.node_vocab))[indices]
        return x

    def process_graph(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        node_list = data['nodes']
        edge_list = data['edges']
        node_idx = {name: idx for idx, name in enumerate(node_list)}
        edge_index = torch.tensor([[node_idx[src], node_idx[dst]] for src, dst in edge_list], dtype=torch.long).t().contiguous()
        x = self.encode_nodes(node_list)
        y = torch.tensor([data['label']], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=y)

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        return self.process_graph(file_path)
