from scipy import stats
from random import choice
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import DataLoader

from lib.network import FulvioNet, SiameseGNNDataset
from lib.utils import split_test, load_graphs, convert_graphs

#folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_radius'
folder = 'C:/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_nn'

graphs, labels = load_graphs(folder)

(train_graphs, _, _,
 train_labels, _, _) = split_test(graphs, labels, 45) # prendo solo 45 grafi

# 40 nel train 5 nel validation
(train_graphs, probe_valid_graphs, gallery_valid_graphs,
 train_labels, probe_valid_labels, gallery_valid_labels) = split_test(train_graphs, train_labels, 40)

gallery_valid_graphs = convert_graphs(gallery_valid_graphs)
probe_valid_graphs = convert_graphs(probe_valid_graphs)
train_graphs = convert_graphs(train_graphs)

train_set = SiameseGNNDataset(train_graphs, train_labels)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)


def sample_hyperparameters(grid):
    assert all([isinstance(grid[param], list) or hasattr(grid[param], 'rvs')
                for param in grid.keys()])
    return {
        param: choice(grid[param]) if isinstance(grid[param], list)
        else grid[param].rvs()
        for param in grid.keys()
    }


grid = {
    'in_channels': [13],
    'hidden_channels': [64, 128, 256, 512],
    'out_channels': [64, 128, 256, 512],
    'max_epoch': [1000],
    'learning_rate': stats.uniform(.0001, .005),
    'margin': stats.uniform(0, 10),
    'writer': [None],
    'tolerance': [50],
    #'layer_type': ['gcn', 'gat', 'transformer', 'pna']
    'layer_type': ['pna']
}

res = {}
n_iter = 50
for i in tqdm(range(n_iter), desc='fitting and evaluating models'):  # for n_iter

    hyperparameters = sample_hyperparameters(grid)  # sample hyperparameters
    toadd = hyperparameters.copy()  # save hyperparameters for the new entry
    print(f'\nhyperparameters: {hyperparameters}')
    # fittng
    model = FulvioNet(**hyperparameters)

    # compute metric
    toadd['err'] = model.fit(train_loader, gallery_valid_graphs, gallery_valid_labels, probe_valid_graphs,
                             probe_valid_labels)

    res[i] = toadd  # add new entry to results
    pd.DataFrame.from_dict(res, orient='index').to_csv('hypersearch_result.csv')  # build dataframe
