import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from karateclub import Graph2Vec, WaveletCharacteristic
from sklearn.decomposition import KernelPCA, PCA

#%% carico i grafi
graph_dict = {}
for filename in os.listdir('grafi'):
    with open('grafi/' + filename, 'rb') as f:
        graph = pkl.load(f)
        graph_dict[filename] = graph

# %% embedding con graph2vec con parametri di default
graph2vec_model = Graph2Vec()
#graph2vec_model = WaveletCharacteristic()

# fitto il modello sul  grafo
graph2vec_model.fit(list(graph_dict.values()))
# ottengo l'embedding del grafo
embeddings = {k: graph2vec_model.infer([graph_dict[k]]) for k in graph_dict.keys()}

#%% riduco l'embedding a due dimensioni con la PCA
emb = []
c = []
for k in graph_dict.keys():
    emb.append(embeddings[k])
    s = int(k.split('_')[1][1])
    c.append(int(k.split('_')[0][1:]))

#pca = KernelPCA(n_components=2, kernel='cosine')
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(np.concatenate(emb))

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=c, cmap='tab20')  # 20 cluster max
plt.title("Graph2Vec embedding")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#%% calcolo silhouette_score
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(np.concatenate(emb), c)
print(silhouette)  #-0.3638413 metric='cosine'
#-0.18333729
