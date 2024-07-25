import os
import pickle as pkl
import numpy as np
from karateclub import Graph2Vec
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize

#%% caricamento grafi_old
graph_dict = {}
for filename in os.listdir('grafi_old'):
    with open('grafi_old/' + filename, 'rb') as f:
        graph = pkl.load(f)
        graph_dict[filename] = graph

# embedding dei grafi_old (parametri di default)
graph2vec_model = Graph2Vec()
graph2vec_model.fit(list(graph_dict.values()))

embeddings = {k: graph2vec_model.infer([graph_dict[k]]) for k in graph_dict.keys()}

# creazione degli array di embedding e delle etichette
X = np.concatenate(list(embeddings.values()))
y = np.array([int(name.split('_')[0]) for name in graph_dict.keys()])

X = normalize(X)

#%% train e test
train_set = []
train_labels = []

for i in range(X.shape[0]):
    curr = X[i, :]
    curr_class = y[i]

    X_pos = X[y == curr_class, :]
    j = np.random.randint(0, X_pos.shape[0])
    X_neg = X[y != curr_class, :]
    h = np.random.randint(0, X_neg.shape[0])
    #pos = curr * X_pos[j, :]
    #neg = curr * X_neg[h, :]
    pos = np.concatenate((curr, X_pos[j, :]))
    neg = np.concatenate((curr, X_neg[h, :]))
    train_set += [pos, neg]
    train_labels += [1, 0]

train_set = normalize(np.array(train_set))
train_labels = np.array(train_labels)

#%% SVM kernel rbf
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
print(cross_validate(svm, train_set, train_labels, scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                            'roc_auc']))
svm.fit(train_set, train_labels)


#%% create svm metric
def svm_distance(x1, x2):
    return svm.predict_proba([x1 * x2])[0,0]


#%% carico i grafi_old dalla GALLERY
graph_dict = {}

for filename in os.listdir('gallery_grafi'):
    with open('gallery_grafi/' + filename, 'rb') as f:
        graph = pkl.load(f)
        graph_dict[filename] = graph

# ottengo l'embedding del grafo
embeddings = {k: graph2vec_model.infer([graph_dict[k]]) for k in graph_dict.keys()}

emb_gallery = []
c_gallery = []

for k in graph_dict.keys():
    emb_gallery.append(embeddings[k])
    c_gallery.append(int(k.split('_')[0]))

#%% carico i grafi_old del PROBE
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

#%% calculate
from scipy.io import savemat

embedding_array_gallery = np.concatenate(emb_gallery)
embedding_array_probe = np.concatenate(emb_probe)

distance_matrix_3 = pairwise_distances(embedding_array_probe, embedding_array_gallery, metric=svm_distance)
print(distance_matrix_3.mean())

#print(f"Dimensione DM {distance_matrix_3.shape}")
print(f"Diagonale DM: {np.diag(distance_matrix_3)}")

savemat('distance_matrix_3.mat', {'distance_matrix_3': distance_matrix_3})
