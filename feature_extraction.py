import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, match_descriptors, ORB, BRIEF, CENSURE, Cascade
from sklearn.neighbors import NearestNeighbors
import pickle as pkl

#%%
plt.gray()  # only for visualization

image = Image.open(
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S2_I1.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.title("iride in grayscale")
plt.show()

#%%
descriptor_extractor = SIFT()
#descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(image)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors


#%%
def distance(a, b):  # distanza minima tra due punti

    return abs(a[0] - b[0]) + np.min(np.abs([a[1] - b[1],
                                             a[1] - (b[1] + 1),
                                             (a[1] + 1) - b[1]
                                             ]))


def normalize(dist, image):
    h, w = image.shape
    res = dist.astype(float)
    res[:, 0] /= w
    res[:, 1] /= w
    return res


normalized = normalize(keypoints1, image)

#%%
# mi prendo le 5 componenti più vicine (basate su distance()) per costruire il grafo
model = NearestNeighbors(n_neighbors=5, metric=distance)
model.fit(normalized)  # coordinate dei punti
graph = model.kneighbors_graph(normalized)

# 0.1 (10% dell'immagine): mi prendo le componenti più vicine a distanza normalizzata di 0.1
# graph = model.radius_neighbors_graph(keypoints1, radius=0.1)

pd.DataFrame(graph).to_csv('graph.csv', header=False)

#%% descrittori
plt.imshow(image)
plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=3, color='red')

#%% grafo
to_plot = model.kneighbors(None, return_distance=False, n_neighbors=5)
plt.imshow(image)
plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=3, color='red')

for i in range(to_plot.shape[0]):
    for j in to_plot[i]:
        cx = [keypoints1[i, 0], keypoints1[j, 0]]
        cy = [keypoints1[i, 1], keypoints1[j, 1]]
        a = normalized[i, :]
        b = normalized[j, :]

        if np.min(np.abs([a[1] - b[1],
                          a[1] - (b[1] + 1),
                          (a[1] + 1) - b[1]
                          ])) == np.abs(a[1] - b[1]):

            plt.plot(cy, cx, color='red')

        else:
            plt.plot(cy, cx, color='blue')

#%% per vedere la topologia del grafo
G = nx.from_numpy_array(graph)
#node_attr = {i: descriptors1[i, :] for i in range(descriptors1.shape[0])}
nx.set_node_attributes(G, descriptors1, 'descriptors')
nx.set_node_attributes(G, keypoints1, 'keypoints')

with open("grafo.nx", 'wb') as f:
    pkl.dump(G, f)
    nx.draw(G)
    plt.show()
    print("DONE")




# %%
#import graph2vec
from graph2vec import Graph2Vec

graphs = [G]

graph2vec_model = Graph2Vec(wl_iterations=2, attributed=False, dimensions=128, workers=4, down_sampling=0.0001,
                            epochs=10, learning_rate=0.025, min_count=5, seed=42, erase_base_features=False)

# fitto il modello sul  grafo
graph2vec_model.fit(graph)

# ottengo l'embedding del grafo
embeddings = graph2vec_model.transform(graph)


#%%
import graph2vec

#graphs = [G]

# embedding del grafo utilizzando Graph2Vec
model = graph2vec.Graph2Vec()   #Graph2VecTransformer()
model.fit_transform(graphs)
embeddings = model.get_embedding()

#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# riduco l'embedding a due dimensioni utilizzando PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Graph2Vec embedding")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
