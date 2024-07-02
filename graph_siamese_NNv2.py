import os
import pickle as pkl
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import device
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils.convert import from_networkx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% VALIDATION FUNCTION
def calculate_frr_far_eer(thresholds, distances, labels):
    FAR = []
    FRR = []

    for thresh in thresholds:
        # FAR: frequenza con la quale il sistema rifiuta ingiustamente individui che sono autorizzati
        false_acceptances = np.sum((distances < thresh) & (labels == 0))
        total_imposters = np.sum(labels == 0)
        far = false_acceptances / total_imposters if total_imposters != 0 else 0

        # FRR: frequenza con cui il sistema è ingannato da estranei che riescono a essere autorizzati
        false_rejections = np.sum((distances >= thresh) & (labels == 1))
        total_legitimate = np.sum(labels == 1)
        frr = false_rejections / total_legitimate if total_legitimate != 0 else 0

        FAR.append(far)
        FRR.append(frr)

    FAR = np.array(FAR)
    FRR = np.array(FRR)

    # EER: indica l'errore del sistema nel punto in cui FRR = FAR
    # Un sistema biometrico è tanto migliore quanto minore è il valore del suo ERR (Error Equal Rate).
    eer_index = np.nanargmin(np.abs(FAR - FRR))
    EER = thresholds[eer_index]

    return FAR, FRR, EER


"""
# TEST di esempio 
thresholds = np.linspace(0, 1, 100)
distances = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.3])
labels = np.array([0, 0, 1, 1, 1, 0])

FAR, FRR, EER = calculate_frr_far_eer(thresholds, distances, labels)
print(f"FAR: {FAR}")
print(f"FRR: {FRR}")
print(f"EER: {EER}")
"""


#%% Definizione del modello GNN e triplette loss
class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pooling globale per ottenere una rappresentazione del grafo
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


#%% Definizione del modello Siamese
class SiameseGNN(nn.Module):
    def __init__(self, gnn):
        super(SiameseGNN, self).__init__()
        self.gnn = gnn

    def forward_once(self, data):
        return self.gnn(data.x, data.edge_index, data.batch)

    def forward(self, data1, data2, data3):
        output1 = self.forward_once(data1)
        output2 = self.forward_once(data2)
        output3 = self.forward_once(data3)
        return output1, output2, output3


class SiameseGNNDataset(Dataset):
    def __init__(self, graphs, labels):
        super(SiameseGNNDataset, self).__init__()
        #self.graphs = [from_networkx(g, group_node_attrs=['feature']) for g in graphs]
        self.graphs = [self.convert_graph(from_networkx(g, group_node_attrs=['feature'])) for g in graphs]
        self.labels = labels

    def convert_graph(self, graph):
        graph.x = graph.x.float()  #converto le feature dei nodi a float
        return graph

    def __getitem__(self, index):
        anchor = self.graphs[index]
        positive = random.choice(
            [self.graphs[i] for i in range(len(self.labels)) if self.labels[i] == self.labels[index]])
        negative = random.choice(
            [self.graphs[i] for i in range(len(self.labels)) if self.labels[i] != self.labels[index]])
        return anchor, positive, negative

    def __len__(self):
        return len(self.graphs)


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


######## START ########
#%% Creazione delle coppie
folder = 'C:/Users/fulvi/DataspellProjects/tesi/feature2pkl'
graphs, labels = load_graphs(folder)

train_set = SiameseGNNDataset(graphs, labels)
print(f"Number of graphs: {len(train_set)}")

#creazione del DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

#definizione dell'architettura GNN
in_channels = 13
hidden_channels = 64
out_channels = 32
gnn = GNN(in_channels, hidden_channels, out_channels)

model = SiameseGNN(gnn).to(device)

#definizione dell'optimizer e della loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
#criterion = CosineEmbeddingLoss()
criterion = TripletMarginLoss(margin=1.0)

#%% Funzione di addestramento e plot con Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
dataiter = iter(train_loader)
example, _, _ = next(dataiter)

writer.add_graph(model.gnn, [example.x, example.edge_index, example.batch])


#%% Train loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(*batch)
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)

    print(f'Epoch [{epoch + 1}/20], Loss: {avg_loss:.4f}')


writer.flush()

num_epochs = 20
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)

embeddings = torch.stack([model.forward_once(train_set.graphs[i]) for i in range(len(train_set))])[:, 0, :]
writer.add_embedding(mat=embeddings, metadata=train_set.labels)
