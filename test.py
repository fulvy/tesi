import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

#efinisci il grafo
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Converte i dati in un oggetto NetworkX Graph
graph = nx.Graph()
graph.add_nodes_from(range(data.num_nodes))
graph.add_edges_from(data.edge_index.t().tolist())

# show graph
pos = nx.spring_layout(graph)  # Puoi utilizzare diverse posizioni per il layout
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10, font_color='black',
        font_weight='bold', width=1.5, edge_color='gray')

plt.show()
