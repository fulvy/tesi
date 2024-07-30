import numpy as np
from sklearn.metrics import pairwise_distances
from torch_geometric.loader import DataLoader
from lib.network import *
from lib.utils import load_graphs, convert_graphs, split_test
from torch.utils.tensorboard import SummaryWriter
from hyperparameters_script import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Caricamento dei grafi
folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_nn'
#folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_radius'

graphs, labels = load_graphs(folder)

(train_graphs, probe_test_graphs, gallery_test_graphs,
 train_labels, probe_test_labels, gallery_test_labels) = split_test(graphs, labels, 45)

(train_graphs, probe_valid_graphs, gallery_valid_graphs,
 train_labels, probe_valid_labels, gallery_valid_labels) = split_test(train_graphs, train_labels, 40)

#%% creazione del dataset e del DataLoader
gallery_graphs = convert_graphs(gallery_valid_graphs)
probe_graphs = convert_graphs(probe_valid_graphs)
train_graphs = convert_graphs(train_graphs)

train_set = SiameseGNNDataset(train_graphs, train_labels)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

writer = SummaryWriter()

#%% creazione e addestramento del modello
model = FulvioNet(in_channels=13, hidden_channels=128, out_channels=64, max_epoch=800, margin=1,
                  writer=writer)

model.fit(train_loader, gallery_graphs, gallery_valid_labels, probe_graphs, probe_valid_labels)

#%% embedding del test set e visualizzazione
gallery_graphs = convert_graphs(gallery_test_graphs)
probe_graphs = convert_graphs(probe_test_graphs)

test_graphs = gallery_graphs + probe_graphs
test_labels = gallery_test_labels + probe_test_labels
test_embedding = model.transform(test_graphs)

embeddings = torch.Tensor(test_embedding)
writer.add_embedding(mat=embeddings, metadata=test_labels)

"""
#%% salva la dm e le etichette in matlab
from scipy.io import savemat

embedding_probe = model.transform(probe_graphs)
embedding_gallery = model.transform(gallery_graphs)
dm = pairwise_distances(embedding_probe, embedding_gallery)

savemat('DM_TEST.mat', {'DM_TEST': dm})
savemat('PROBE_TEST.mat', {'PROBE_TEST': np.array(probe_test_labels)})
savemat('GALLERY_TEST.mat', {'GALLERY_TEST': np.array(gallery_test_labels)})
"""

"""
DM è una matrice 600 x 600
Probe = [ 1 1 1 1.., 2 2 2, ...]
Gallery = [1 1 1 1..., 2,2,2,2,....]
SRR = ones(1, length(Probe));
srr_th=0;
titolo='Esperimento...'

SystemPerformanceFromDM(-(distance_matrix),c_probe,c_gallery,SRR,srr_th,titolo)
"""
