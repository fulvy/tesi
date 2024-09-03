import os
import pickle as pkl
from scipy.io import savemat
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.utils import degree
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def split_test(graphs, labels, train_size):
    """
    splitta i grafi in train e test. i grafi di test vengono suddivisi in probe e gallery
    :param graphs: lista di grafi
    :param labels:  lista di etichette indicanti i sogetti
    :param train_size:  numero di grafi nel train set
    :return:
    """
    train_graphs = []
    probe_graphs = []
    gallery_graphs = []
    train_labels = []
    probe_labels = []
    gallery_labels = []
    for i in range(len(graphs)):
        graph = graphs[i]
        soggetto = labels[i]

        if soggetto < train_size:
            train_graphs.append(graph)
            train_labels.append(soggetto)

        elif i % 2 == 0:
            probe_graphs.append(graph)
            probe_labels.append(soggetto)

        else:
            gallery_graphs.append(graph)
            gallery_labels.append(soggetto)
    return train_graphs, probe_graphs, gallery_graphs, train_labels, probe_labels, gallery_labels


def load_graphs(folder):
    """
    se fai train_size = 0 te lo splitta solamente in probe e gallery
    """
    graphs = []
    labels = []

    for filename in os.listdir(folder):
        soggetto = filename.replace('.pkl', '').split('_')[0]
        labels.append(int(soggetto))
        with open(os.path.join(folder, filename), 'rb') as f:
            graphs.append(pkl.load(f))
    return graphs, labels


def convert_graphs(graphs):
    ret = []
    for g in graphs:
        to_add = from_networkx(g, group_node_attrs=['feature'])
        to_add.x = to_add.x.float()  # converto le feature dei nodi a float
        ret.append(to_add)
    return ret


"""
Per calcolare il parametro deg da passare ai layer PNA, devi calcolare 
il grado di ciascun nodo nel tuo grafo e quindi ottenere la distribuzione dei gradi
"""


def compute_deg(graphs):
    all_degrees = []
    for data in graphs:
        edge_index = data.edge_index[0]  # Ottieni gli indici dei nodi di partenza per ogni grafo
        deg = degree(edge_index, dtype=torch.long)  # Calcola il grado per ogni nodo
        all_degrees.append(deg)

    # Ottieni la distribuzione dei gradi per l'intero dataset
    # Unisci tutti i gradi calcolati in un unico tensor
    all_degrees = torch.cat(all_degrees, dim=0)

    # Conta la distribuzione dei gradi
    deg_histogram = torch.bincount(all_degrees)

    return deg_histogram


def save_matlab(experiment_name, dm, probe_label, gallery_label):
    savemat(f'{experiment_name}_DM.mat', {'DM_TEST': dm})
    savemat(f'{experiment_name}_PROBE.mat', {'PROBE_TEST': np.array(probe_label)})
    savemat(f'{experiment_name}_GALLERY.mat', {'GALLERY_TEST': np.array(gallery_label)})
