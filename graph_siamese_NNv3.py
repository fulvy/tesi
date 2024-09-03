from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from lib.network import *
from lib.utils import load_graphs, convert_graphs, split_test, save_matlab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Caricamento dei grafi
folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_nn'
#folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_radius'

graphs, labels = load_graphs(folder)

(train_graphs, probe_test_graphs, gallery_test_graphs,
 train_labels, probe_test_labels, gallery_test_labels) = split_test(graphs, labels, 45)

(train_graphs, probe_valid_graphs, gallery_valid_graphs,
 train_labels, probe_valid_labels, gallery_valid_labels) = split_test(train_graphs, train_labels, 40)
print('hey')
#%% creazione del dataset e del DataLoader
gallery_valid_graphs = convert_graphs(gallery_valid_graphs)
probe_valid_graphs = convert_graphs(probe_valid_graphs)
train_graphs = convert_graphs(train_graphs)

train_set = SiameseGNNDataset(train_graphs, train_labels)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

writer = SummaryWriter()

print('hey')
#%% creazione e addestramento del modello
model = FulvioNet(in_channels=13, hidden_channels=128, out_channels=256, max_epoch=1000, margin=4,learning_rate=0.0015,
                  layer_type='pna', writer=writer)

model.fit(train_loader, gallery_valid_graphs, gallery_valid_labels, probe_valid_graphs, probe_valid_labels)
print('hey')
#%% embedding del test set e visualizzazione
gallery_test_graphs = convert_graphs(gallery_test_graphs)
probe_test_graphs = convert_graphs(probe_test_graphs)

test_graphs = gallery_test_graphs + probe_test_graphs
test_labels = gallery_test_labels + probe_test_labels
test_embedding = model.transform(test_graphs)

embeddings = torch.Tensor(test_embedding)
writer.add_embedding(mat=embeddings, metadata=test_labels)
print('hey')
