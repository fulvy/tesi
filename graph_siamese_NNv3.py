import numpy as np
from sklearn.metrics import pairwise_distances
from torch_geometric.loader import DataLoader
from lib.network import *
from lib.utils import load_graphs, convert_graphs
from torch.utils.tensorboard import SummaryWriter
from scipy.io import savemat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% Caricamento dei grafi
#folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_nn'
folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_radius'
train_graphs, probe_graphs, gallery_graphs, train_labels, probe_labels, gallery_labels = load_graphs(folder, 40)


#%% creazione del dataset e del DataLoader
gallery_graphs = convert_graphs(gallery_graphs)
probe_graphs = convert_graphs(probe_graphs)
train_graphs = convert_graphs(train_graphs)

train_set = SiameseGNNDataset(train_graphs, train_labels)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

writer = SummaryWriter()


#%% creazione e addestramento del modello
model = FulvioNet(in_channels=13, hidden_channels=128, out_channels=64, n_epochs=800, margin=1,
                  writer=writer, step_size=5)

model.fit(train_loader, gallery_graphs, gallery_labels, probe_graphs, probe_labels)


#%% embedding del test set e visualizzazione
test_graphs = gallery_graphs + probe_graphs
test_labels = gallery_labels + probe_labels
test_embedding = model.transform(test_graphs)

embeddings = torch.Tensor(test_embedding)
writer.add_embedding(mat=embeddings, metadata=test_labels)



#%%
"""
#%% salva la dm e le etichette in matlab
embedding_probe = model.transform(probe_graphs)
embedding_gallery = model.transform(gallery_graphs)
dm = pairwise_distances(embedding_probe, embedding_gallery)

savemat('DM_TEST.mat', {'DM_TEST': dm})
savemat('PROBE_TEST.mat', {'PROBE_TEST': np.array(probe_labels)})
savemat('GALLERY_TEST.mat', {'GALLERY_TEST': np.array(gallery_labels)})
"""

