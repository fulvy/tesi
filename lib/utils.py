import os
import pickle as pkl

from torch_geometric.utils import from_networkx


def load_graphs(folder, train_size=0):
    """
    se fai train_size = 0 te lo splitta solamente in probe e gallery
    """
    train_graphs = []
    probe_graphs = []
    gallery_graphs = []
    train_labels = []
    probe_labels = []
    gallery_labels = []

    for filename in os.listdir(folder):
        soggetto, idx = filename.replace('.pkl', '').split('_')
        soggetto = int(soggetto)
        idx = int(idx)
        with open(os.path.join(folder, filename), 'rb') as f:
            graph = pkl.load(f)

            if soggetto < train_size:
                train_graphs.append(graph)
                train_labels.append(soggetto)

            elif idx % 2 == 0:
                probe_graphs.append(graph)
                probe_labels.append(soggetto)

            else:
                gallery_graphs.append(graph)
                gallery_labels.append(soggetto)

    if train_size == 0:
        return probe_graphs, gallery_graphs, probe_labels, gallery_labels

    return train_graphs, probe_graphs, gallery_graphs, train_labels, probe_labels, gallery_labels


def convert_graphs(graphs):
    ret = []
    for g in graphs:
        to_add = from_networkx(g, group_node_attrs=['feature'])
        to_add.x = to_add.x.float()  #converto le feature dei nodi a float
        ret.append(to_add)
    return ret
