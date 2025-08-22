import pandas as pd
import re
import networkx as nx
import os
import json

df = pd.read_csv("reentrancy_sample.csv")
output_dir = "text_graphs"
os.makedirs(output_dir, exist_ok=True)

for idx, row in df.iterrows():
    code = row['code']
    label = row['is_reentrancy']
    graph = nx.DiGraph()

    functions = re.findall(r'function\s+(\w+)\s*\(', code)
    for fn in functions:
        graph.add_node(fn)

    for caller in functions:
        body_match = re.search(rf'function\s+{caller}\s*\(.*?\)\s*.*?\{{(.*?)\}}', code, re.DOTALL)
        if body_match:
            body = body_match.group(1)
            for callee in functions:
                if callee in body and callee != caller:
                    graph.add_edge(caller, callee)

    for keyword in ['call.value', 'delegatecall', 'require', 'revert', 'assert']:
        if keyword in code:
            graph.add_node(keyword)
            for fn in functions:
                graph.add_edge(fn, keyword)

    data = {
        "label": int(label),
        "edges": list(graph.edges()),
        "nodes": list(graph.nodes())
    }

    with open(os.path.join(output_dir, f"graph_{idx}.json"), 'w') as f:
        json.dump(data, f)
