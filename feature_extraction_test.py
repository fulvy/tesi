import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from skimage import morphology
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import SIFT, match_descriptors, ORB, BRIEF, CENSURE, Cascade
from sklearn.neighbors import NearestNeighbors
import pickle as pkl
from sklearn.metrics.pairwise import pairwise_distances

from skimage.exposure import rescale_intensity

#%%
plt.gray()  # only for visualization

image = Image.open(
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\0000_002.png").convert("RGB")

image = rgb2gray(image)  # grayscale


#plt.imshow(image)
#plt.title("iride in grayscale")
#plt.show()


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


#%% #descrittori
p1, p2 = np.percentile(image, [2, 90])
image = exposure.rescale_intensity(image, in_range=(p1, p2))
image = morphology.diameter_opening(image, diameter_threshold=32)

descriptor_extractor = SIFT()
#descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(image)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

plt.imshow(image)
plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=4, color='red')
plt.show()

#%%
normalized = normalize(keypoints1, image)

# %%
distances_test = pairwise_distances(normalized, metric=distance)

#%%
dist = distances_test.flatten()
plt.hist(dist[dist != 0], bins=100)
plt.show()

#%%
# mi prendo le 5 componenti più vicine (basate su distance()) per costruire il grafo
model = NearestNeighbors(n_neighbors=5, metric=distance)
model.fit(normalized)  # coordinate dei punti
#graph = model.kneighbors_graph()

# 0.1 (10% dell'immagine): mi prendo le componenti più vicine a distanza normalizzata di 0.1
graph = model.radius_neighbors_graph(radius=0.3)
graph1 = model.kneighbors_graph(n_neighbors=2)
graph = np.maximum(graph.toarray(), graph1.toarray())

#pd.DataFrame(graph).to_csv('graph_test.csv', header=False)

#%% grafo

to_plot = model.radius_neighbors(None, return_distance=False, radius=0.1)
to_plot1 = model.kneighbors(None, return_distance=False, n_neighbors=2)  # mode=distance

for i in range(len(to_plot)):
    for j in to_plot1[i]:
        to_plot[i] = np.append(to_plot[i], to_plot1[j])
plt.imshow(image)
plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=5, color='red')

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
node_attr = {i: descriptors1[i, :] for i in range(descriptors1.shape[0])}
nx.set_node_attributes(G, descriptors1, 'descriptor')
nx.set_node_attributes(G, keypoints1, 'keypoints')

with open("grafo_test.nx", 'wb') as f:
    pkl.dump(G, f)
    nx.draw(G)
    plt.show()
    print("DONE")
