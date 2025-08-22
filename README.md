# Capstone-Project-Model
includes all the files used to build and run the Hybrid GNN-Transformer Model  along with the dataset .csv file and a readme file describing the order in which files must be ran 


This repository contains the follwoing files:
*  SC_Vuln_8label.csv
*  reentrancy_sample.csv
*  graph_gen.py
*  text_graph_dataset.py
*  train_gnn.py
*  generate_gnn_embeddings.py
*  TranGNN.py

# DATASET FILES
1.  SC_Vuln_8label.csv:  This is the initial dataset as retireved from kaggle (https://www.kaggle.com/datasets/tranduongminhdai/smart-contract-vulnerability-datset/data?select=SC_Vuln_8label.csv)

2.  reentrancy_sample.csv:  This is the balanced dataset with a sample size of 1200 smart contracts

# MODEL FILES
1.  graph_gen.csv:   Running this converts the solidity contract code in  reentrancy_sample.csv to graph representations

2.  text_graph_dataset.py:  Loads the JSON graphs made by graph_gen.csv and prepares the labeled graph data for tarining the GNN

3.  train_gnn.py:  Trains a GCN on the graph data ad saved the trained GNN weights

4.  generate_gnn_embeddings.py:  loads the trained GNN and runs every graph through the GNN and extracts the graph embeddings and skipped graphs and saves them to .npy files.

5.  TranGNN.py:  Implements the hybrid model.


# EXECUTION SEQUENCE

```mermaid
flowchart TD
    A[graph_gen.py] --> B[train_gnn.py]
    B --> C[generate_gnn_embeddings.py]
    C --> D[TranGNN.py]


