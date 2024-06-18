import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from karateclub import Graph2Vec, WaveletCharacteristic
from scipy.io import savemat
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
from sklearn.metrics import silhouette_score, pairwise_distances

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

"""
Metrica per valutare la qualità della clusterizzazione, bilanciando l'omogeneità e la completezza.
Questa metrica è indipendente dai valori assoluti delle etichette: 
una permutazione dei valori delle etichette della classe o del cluster non modificherà in alcun modo il valore del punteggio.
Questa metrica è inoltre simmetrica: il passaggio label_true con label_pred restituirà lo stesso valore.
Ciò può essere utile per misurare l'accordo di due strategie di assegnazione di etichette indipendenti sullo stesso set di dati 
quando il "real ground truth" non è nota.
"""

v_score = v_measure_score(c, cluster)
v_score_rand = v_measure_score(c, rand)
print(f"v_score: {v_score} --- random v_score: {v_score_rand}")
#v_score: 0.30231617598101374 --- random v_score: 0.31693013884703486


"""
La randomizzazione serve per avere un termine di confronto. 
La misura di v per i cluster casuali fornisce un benchmark o un punto di riferimento che indica 
quale sarebbe il valore della metrica se i cluster fossero completamente casuali. 
Confrontando la V-measure dei cluster reali con quella dei cluster casuali, può valutare quanto 
meglio (o peggio) la clusterizzazione effettiva sta performando rispetto ad una pura assegnazione casuale
"""


#%% matrice delle distanze tutti-contro-tutti
from sklearn.metrics import pairwise_distances
from scipy.io import savemat

embedding_array = np.concatenate(emb)
print(type(embedding_array))
distance_matrix = pairwise_distances(embedding_array, metric='euclidean')

print(f"Dimensione DM {distance_matrix.shape}")
print(f"Diagonale DM (dovrebbe essere 0): {np.diag(distance_matrix)}")

savemat('distance_matrix.mat', {'distance_matrix': distance_matrix})


#%% negata della DM
negated_distance_matrix = -distance_matrix

print(f"Dimensione -DM {negated_distance_matrix.shape}")
print(f"Diagonale -DM: {np.diag(negated_distance_matrix)}")

savemat('negated_distance_matrix.mat', {'negated_distance_matrix': negated_distance_matrix})


#%% Normalizzare la matrice delle distanze negate
min_value = np.min(negated_distance_matrix)
max_value = np.max(negated_distance_matrix)
norm_negated_distance_matrix = (negated_distance_matrix - min_value) / (max_value - min_value)

print(f"Dimensione norm_negated_DM {norm_negated_distance_matrix.shape}")
print(f"Diagonale norm_negated_DM: {np.diag(norm_negated_distance_matrix)}")

savemat('norm_negated_distance_matrix.mat', {'norm_negated_distance_matrix': norm_negated_distance_matrix})