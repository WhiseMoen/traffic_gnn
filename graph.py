# graph.py
# -*- coding: utf-8 -*-
import torch
import networkx as nx
import matplotlib.pyplot as plt

def get_edge_index():
    """Возвращает двунаправленный edge_index для 6 перекрёстков."""
    edge_index = torch.tensor([
        [0,1,1,2,0,3,1,4,2,5,3,4,4,5],
        [1,0,2,1,3,0,4,1,5,2,4,3,5,4]
    ], dtype=torch.long)
    return edge_index

def visualize_graph(edge_index):
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(5,5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=1500, font_size=12)
    plt.title("Сеть перекрёстков")
    plt.show()