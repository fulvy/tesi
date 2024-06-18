import os
import pickle as pkl
from karateclub import Graph2Vec
from sklearn.metrics.pairwise import cosine_similarity

#GALLERY
#%% carico i grafi
graph_dict = {}

for filename in os.listdir('gallery_grafi'):
    with open('gallery_grafi/' + filename, 'rb') as f:
        graph = pkl.load(f)
        graph_dict[filename] = graph

# embedding con graph2vec con parametri di default
graph2vec_model = Graph2Vec()

# fitto il modello sul  grafo
graph2vec_model.fit(list(graph_dict.values()))

# ottengo l'embedding del grafo
embeddings = {k: graph2vec_model.infer([graph_dict[k]]) for k in graph_dict.keys()}

emb_gallery = []
c_gallery = []
print(type(c_gallery))
for k in graph_dict.keys():
    emb_gallery.append(embeddings[k])
    c_gallery.append(int(k.split('_')[0]))

#----------------------------------------------------------------------------------------------------------------------#

#PROBE
#%% carico i grafi
graph_dict = {}

for filename in os.listdir('probe_grafi'):
    with open('probe_grafi/' + filename, 'rb') as f:
        graph = pkl.load(f)
        graph_dict[filename] = graph

# ottengo l'embedding del grafo
embeddings = {k: graph2vec_model.infer([graph_dict[k]]) for k in graph_dict.keys()}

emb_probe = []
c_probe = []
for k in graph_dict.keys():
    emb_probe.append(embeddings[k])
    c_probe.append(int(k.split('_')[0]))

#%%
from scipy.io import savemat
from sklearn.metrics import pairwise_distances
import numpy as np

embedding_array_gallery = np.concatenate(emb_gallery)
embedding_array_probe = np.concatenate(emb_probe)

#%%
distance_matrix = pairwise_distances(embedding_array_probe, embedding_array_gallery, metric=cosine_similarity)
print(distance_matrix.mean())

print(f"Dimensione DM {distance_matrix.shape}")
print(f"Diagonale DM: {np.diag(distance_matrix)}")

savemat('distance_matrix.mat', {'distance_matrix': distance_matrix})

#%%
savemat('c_gallery.mat', {'c_gallery': c_gallery})
savemat('c_probe.mat', {'c_probe': c_probe})

