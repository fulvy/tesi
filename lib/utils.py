import os
import pickle as pkl

from torch_geometric.utils import from_networkx


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
        to_add.x = to_add.x.float()  #converto le feature dei nodi a float
        ret.append(to_add)
    return ret
