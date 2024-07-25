import numpy as np
from tqdm import tqdm
import pickle as pkl
import networkx as nx
import pandas as pd
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from sklearn.metrics import pairwise_distances
import os


#%% EXTRACT GRAPHS WITH NearestNeighbors()
in_dir = f'/Users/fulvi/DataspellProjects/tesi/casia_graph_txt/'
out_dir = f'/Users/fulvi/DataspellProjects/tesi/casia_graph_pkl_nn/'

columns = ["X", "Y"] + [f'd{i + 1}' for i in range(13)]

for f_name in tqdm(os.listdir(in_dir)):
    out_file = out_dir + f_name.replace('.txt', '.pkl')
    df = pd.read_csv(in_dir + f_name, delimiter=" ", header=None, names=columns, skiprows=1, index_col=False)

    model = NearestNeighbors(n_neighbors=5)
    model.fit(df[['X', 'Y']])
    graph = model.kneighbors_graph()

    G = nx.from_numpy_array(graph)
    node_attr = {i: df.iloc[i, 2:].to_numpy().flatten() for i in range(df.shape[0])}
    nx.set_node_attributes(G, node_attr, 'feature')

    with open(out_file, 'wb') as f:
        pkl.dump(G, f)



#%% EXTRACT GRAPHS WITH radius_neighbors_graph()
in_dir = f'/Users/fulvi/DataspellProjects/tesi/casia_graphs_txt/'
out_dir = f'/Users/fulvi/DataspellProjects/tesi/casia_graphs_pkl_radius/'

columns = ["X", "Y"] + [f'd{i + 1}' for i in range(13)]

for f_name in tqdm(os.listdir(in_dir)):
    out_file = out_dir + f_name.replace('.txt', '.pkl')
    df = pd.read_csv(in_dir + f_name, delimiter=" ", header=None, names=columns, skiprows=1, index_col=False)

    DM = pairwise_distances(df[['X', 'Y']], metric='euclidean')
    radius = np.quantile([DM[i, j] for i in range(DM.shape[0]) for j in range(i, DM.shape[1])], 0.05)
    # determina il 5° percentile delle distanze (usato come raggio per i vicini più prossimi)

    nn = NearestNeighbors()
    nn.fit(df[['X', 'Y']])
    graph = nn.kneighbors_graph(n_neighbors=1, mode='distance').todense()
    graph1 = nn.radius_neighbors_graph(radius=radius, mode='distance').todense()
    graph = np.maximum(graph, graph1)
    """Creo due grafi dei vicini più prossimi: 
       graph con un solo vicino più prossimo per ogni punto
       graph1 con tutti i vicini entro il raggio calcolato
       Combina i due grafi prendendo il massimo delle distanze per ciascuna coppia di punti
    """

    G = nx.from_numpy_array(graph)
    node_attr = {i: df.iloc[i, 2:].to_numpy().flatten() for i in range(df.shape[0])}
    nx.set_node_attributes(G, node_attr, 'feature')

    with open(out_file, 'wb') as f:
        pkl.dump(G, f)


#%%
import pandas as pd

path1 = f'/Users/fulvi/DataspellProjects/tesi/casia_graphs_txt/0000_000.txt'
path2 = f'/Users/fulvi/DataspellProjects/tesi/casia_graphs_txt/0004_013.txt'

test1 = pd.read_csv(path1, delimiter=" ", header=None, skiprows=1, index_col=False)
test2 = pd.read_csv(path2, delimiter=" ", header=None, skiprows=1, index_col=False)
