from tqdm import tqdm
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os

in_dir = f'/Users/fulvi/DataspellProjects/tesi/gallery/'
out_dir = f'/Users/fulvi/DataspellProjects/tesi/gallery_grafi/'

columns = [
    "X", "Y", "Average Score", "Direction", "Area", "Perimeter", "Eccentricity",
    "Extent", "MajorAxisLength", "MinorAxisLength", "Orientation", "Solidity",
    "Circularity", "MaxFeretDiameter", "MaxFeretAngle", "MinFeretDiameter", "MinFeretAngle"
]

#%%
for f_name in tqdm(os.listdir(in_dir)):
    out_file = out_dir + f_name.replace('.txt', '.pkl')
    df = pd.read_csv(in_dir + f_name, delimiter=" ", header=None, names=columns, skiprows=1)

    model = NearestNeighbors(n_neighbors=5)
    model.fit(df[['X', 'Y']])
    graph = model.kneighbors_graph()

    G = nx.from_numpy_array(graph)
    node_attr = {i: df.iloc[i, :].to_numpy().flatten() for i in range(df.shape[0])}
    nx.set_node_attributes(G, node_attr, 'feature')

    with open(out_file, 'wb') as f:
        pkl.dump(G, f)

