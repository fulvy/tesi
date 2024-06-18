import os
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import device
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import CosineEmbeddingLoss, HingeEmbeddingLoss
from torch_geometric.utils.convert import from_networkx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% Definizione del modello GNN
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x.float(), edge_index)
        x = F.relu(x)
        x = self.conv2(x.float(), edge_index)
        x = global_mean_pool(x, batch)  # Pooling globale per ottenere una rappresentazione del grafo
        x = self.fc1(x.float())
        x = self.relu(x)
        x = self.fc2(x.float())

        return x


#%% Definizione del modello Siamese
class SiameseGNN(nn.Module):
    def __init__(self, gnn):
        super(SiameseGNN, self).__init__()
        self.gnn = gnn

    def forward_once(self, data):
        return self.gnn(data.x, data.edge_index, data.batch)

    def forward(self, data1, data2):
        output1 = self.forward_once(data1)
        output2 = self.forward_once(data2)
        return output1, output2


class SiameseGNNDataset(Dataset):
    def __init__(self, graphs, labels):
        super().__init__()
        self.graphs = [from_networkx(g, group_node_attrs=['feature']) for g in graphs]
        self.idx = []
        self.tgt = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                self.idx.append((i, j))
                self.tgt.append(1 if labels[i] == labels[j] else -1)

    def convert_graph(self, graph):
        graph.x = graph.x.float()  # Convertire le feature dei nodi a float
        graph.edge_index = graph.edge_index.long()  # Assicurarsi che gli indici degli spigoli siano interi lunghi
        return graph

    def __getitem__(self, index):
        i, j = self.idx[index]
        #return self.graphs[i], self.graphs[j], self.tgt[index]
        return self.graphs[i], self.graphs[j], torch.tensor(self.tgt[index], dtype=torch.float)


    def __len__(self):
        return len(self.idx)


#%% Caricamento dei grafi
def load_graphs(folder):
    graph_list = []
    labels = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'rb') as f:
            graph = pkl.load(f)
            graph_list.append(graph)
            labels.append(int(filename.split('_')[0]))
    return graph_list, labels


#%% Caricamento dei grafi e creazione delle coppie
folder = 'C:/Users/fulvi/DataspellProjects/tesi/grafi'
graphs, labels = load_graphs(folder)

train_set = SiameseGNNDataset(graphs, labels)

#creazione del DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

#definizione dell'architettura GNN
in_channels = 17
hidden_channels = 64
out_channels = 32
gnn = GNN(in_channels, hidden_channels, out_channels)

model = SiameseGNN(gnn)
model = model.to(device)

#definizione dell'optimizer e della loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CosineEmbeddingLoss()


#%% Funzione di addestramento
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for data1, data2, label in train_loader:
        data1 = data1.to(device)
        data2 = data2.to(device)
        label = label.to(device).float()  # Convertire a float
        optimizer.zero_grad()
        output1, output2 = model(data1, data2)
        #print(f'data1: {data1.x.dtype}, data2: {data2.x.dtype}, label: {label.dtype}')  # Debug print
        #print(f'output1: {output1.dtype}, output2: {output2.dtype}')                    # Debug print
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/20], Loss: {avg_loss:.4f}')


#addestramento del modello
num_epochs = 20
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
