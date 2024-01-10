import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import NormalizeFeatures
from sklearn.datasets import make_moons
import networkx as nx
import matplotlib.pyplot as plt

# dataset toy usando make_moons di sklearn
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# build graph utilizzando NetworkX
G = nx.Graph()

# add node
for i in range(len(X)):
    G.add_node(i, x=X[i])

for i in range(len(X)):
    for j in range(i+1, len(X)):
        distance = torch.norm(torch.tensor(X[i]) - torch.tensor(X[j])).item()
        G.add_edge(i, j, distance=distance)

# converte il grafo un oggetto PyTorch Geometric Data
data = from_networkx(G)

# show graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=y, cmap=plt.cm.Set1, node_size=200)
plt.show()

# define GNN
class SimpleGNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGNNLayer, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x.float())
        return x

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = SimpleGNNLayer(input_dim, hidden_dim)
        self.conv2 = SimpleGNNLayer(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.leaky_relu(  self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# addestra la GNN
input_dim = X.shape[1]
hidden_dim = 64
output_dim = 2

gnn = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# train e test
data.train_mask = torch.zeros(len(y), dtype=torch.long)  # torch.bool (?)
data.test_mask = torch.zeros(len(y), dtype=torch.uint8)

# uas i primi 80 nodi come set di training
data.train_mask[:80] = 1
data.test_mask[80:] = 1

for epoch in range(100):
    optimizer.zero_grad()
    output = gnn(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], torch.tensor(y[data.train_mask]))
    loss.backward()
    optimizer.step()

# validation la GNN
gnn.eval()
with torch.no_grad():
    pred = gnn(data.x, data.edge_index).argmax(dim=1)

# accuracy
accuracy = (pred[data.test_mask] == y[data.test_mask]).sum().item() / data.test_mask.sum().item()
#accuracy = (pred[data.test_mask] == y[data.test_mask]).to(torch.int).sum().item() / data.test_mask.to(torch.int).sum().item()
#accuracy = (pred[data.test_mask] == y[data.test_mask]).int().sum().item() / data.test_mask.int().sum().item()

print(f'Accuracy: {accuracy}')
