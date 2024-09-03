from sklearn.metrics import pairwise_distances

from lib.network import *
from lib.utils import load_graphs, convert_graphs, split_test, save_matlab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Caricamento dei grafi
folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_nn'

graphs, labels = load_graphs(folder)

(train_graphs, probe_test_graphs, gallery_test_graphs,
 train_labels, probe_test_labels, gallery_test_labels) = split_test(graphs, labels, 45)

(_, probe_valid_graphs, gallery_valid_graphs,
 _, probe_valid_labels, gallery_valid_labels) = split_test(train_graphs, train_labels, 40)

#%% creazione del dataset e del DataLoader
gallery_valid_graphs = convert_graphs(gallery_valid_graphs)
probe_valid_graphs = convert_graphs(probe_valid_graphs)
gallery_test_graphs = convert_graphs(gallery_test_graphs)
probe_test_graphs = convert_graphs(probe_test_graphs)

#%% creazione e addestramento del modello

model_name = ''
with open(f'checkpoints/{model_name}', 'wb') as f:
    model = pkl.load(f)

#%%

embedding_probe_test = model.transform(probe_test_graphs)
embedding_gallery_test = model.transform(gallery_test_graphs)
dm_test = pairwise_distances(embedding_probe_test, embedding_gallery_test)
save_matlab('TEST', dm_test, probe_test_labels, gallery_test_labels)

embedding_probe_valid = model.transform(probe_test_graphs)
embedding_gallery_valid = model.transform(gallery_test_graphs)
dm_valid = pairwise_distances(embedding_probe_valid, embedding_gallery_valid)
save_matlab('VALID', dm_valid, probe_valid_labels, gallery_valid_labels)
