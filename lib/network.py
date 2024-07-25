import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv, global_mean_pool

import pickle as pkl
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from lib.metrics import intra_inter_distance, compute_validation_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        """
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        """


        """
        self.conv1 = GATv2Conv(in_channels, hidden_channels // 8, heads=8, concat=True)
        self.conv2 = GATv2Conv(hidden_channels, out_channels // 8, heads=8, concat=True)
        """

        """
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        """


        self.conv1 = TransformerConv(in_channels, hidden_channels//8, heads=8, concat=True)
        self.conv2 = TransformerConv(hidden_channels, out_channels//8, heads=8, concat=True)


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


#Definizione del modello Siamese
class SiameseGNN(nn.Module):
    def __init__(self, gnn):
        super(SiameseGNN, self).__init__()
        self.gnn = gnn

    def transform(self, data):
        ret = self.forward_once(data)
        return ret.detach().cpu().numpy()

    def forward_once(self, data):
        return self.gnn(data.x, data.edge_index, data.batch)

    def forward(self, data1, data2, data3):
        output1 = self.forward_once(data1)
        output2 = self.forward_once(data2)
        output3 = self.forward_once(data3)
        return output1, output2, output3


class FulvioNet:

    def __init__(self, in_channels=13, hidden_channels=128, out_channels=64, n_epochs=500,
                 step_size=5, margin=1, writer=None):

        gnn = GNN(in_channels, hidden_channels, out_channels)

        self.name = f'GNN_{hidden_channels}_{out_channels}'
        self.siamese = SiameseGNN(gnn).to(device)
        self.optimizer = optim.Adam(self.siamese.parameters(), lr=.001)
        #self.scheduler = StepLR(self.optimizer, step_size=step_size)
        self.writer = writer
        self.n_epochs = n_epochs
        self.criterion = TripletMarginLoss(margin=margin)

    def fit(self, train_loader, gallery_graphs, gallery_labels, probe_graphs, probe_labels):

        # plot network graph on tensorboard
        dataiter = iter(train_loader)
        example, _, _ = next(dataiter)
        self.writer.add_graph(self.siamese.gnn, [example.x, example.edge_index, example.batch])

        for epoch in range(self.n_epochs):
            self.train_one_epoch(train_loader, epoch, gallery_graphs, probe_graphs,
                                 gallery_labels, probe_labels)
            with open(f'checkpoints/{self.name}_{epoch}.pickle', 'wb') as f:
                pkl.dump(self.siamese, f)
        return self

    def train_one_epoch(self, train_loader, epoch, gallery_graphs, probe_graphs,
                        gallery_labels, probe_labels):

        self.siamese.train()
        total_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            anchor_output, positive_output, negative_output = self.siamese(*batch)
            loss = self.criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        #self.scheduler.step()
        self.siamese.eval()

        with torch.no_grad():
            embedding_gallery = [self.siamese.transform(g)[0] for g in gallery_graphs]
            embedding_probe = [self.siamese.transform(g)[0] for g in probe_graphs]

        test_embedding = embedding_gallery + embedding_probe
        test_labels = gallery_labels + probe_labels
        sil = silhouette_score(test_embedding, test_labels)
        intra_inter = intra_inter_distance(test_embedding, test_labels)

        far, frr, roc_auc, eer = compute_validation_metrics(embedding_probe, embedding_gallery, probe_labels,
                                                            gallery_labels)

        avg_loss = total_loss / len(train_loader)

        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('metric/test/roc_auc', roc_auc, epoch)
        self.writer.add_scalar('metric/test/eer', eer, epoch)
        self.writer.add_scalar('metric/test/silhouette', sil, epoch)
        self.writer.add_scalar('metric/test/intra_inter_distances', intra_inter, epoch)

        print(f'Epoch [{epoch + 1}/{self.n_epochs}], Loss: {avg_loss:.4f}')

    def transform(self, graphs):
        self.siamese.eval()
        with torch.no_grad():
            return [self.siamese.transform(g)[0] for g in graphs]


class SiameseGNNDataset(Dataset):
    def __init__(self, graphs, labels):
        super(SiameseGNNDataset, self).__init__()
        self.graphs = graphs
        self.labels = labels

    def __getitem__(self, index):
        anchor = self.graphs[index]
        positive = random.choice(
            [self.graphs[i] for i in range(len(self.labels)) if self.labels[i] == self.labels[index]])
        negative = random.choice(
            [self.graphs[i] for i in range(len(self.labels)) if self.labels[i] != self.labels[index]])
        return anchor, positive, negative

    def __len__(self):
        return len(self.graphs)
