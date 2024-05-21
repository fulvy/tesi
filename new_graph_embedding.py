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

#%% embedding con graph2vec con parametri di default
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
    c.append(int(k.split('_')[0]))

#pca = KernelPCA(n_components=2, kernel='cosine')
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(np.concatenate(emb))

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=c, cmap='tab20')
plt.title("Graph2Vec embedding")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

#%% calcolo silhouette_score
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(np.concatenate(emb), c, metric='cosine')
print(silhouette)

                   #-0.18445517 kernel pca
                   #-0.1824845  pca

#%%
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

cluster = KMeans(n_clusters=60, random_state=0).fit_predict(np.concatenate(emb))
np.random.seed(123)
rand = [np.random.randint(0,59) for _ in range(cluster.shape[0])]


v_score = v_measure_score(c, cluster)
v_score_rand = v_measure_score(c, rand)
print((f"v_score: {v_score} random: {v_score_rand}"))
