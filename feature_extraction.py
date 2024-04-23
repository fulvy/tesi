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
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S2_I5.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.title("iride in grayscale")
plt.show()

#%%
descriptor_extractor = SIFT()  #ORB, surf

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
# mi prendo le 5 componenti più vicine (basate su disyance()) per costruire il grafo
model = NearestNeighbors(n_neighbors=5, metric=distance)
model.fit(normalized)  # coordinate dei punti
graph = model.kneighbors_graph(normalized)

# 0.1 (10% dell'immagine): mi prendo le componenti più vicine a distanza normalizzata di 0.1
# graph = model.radius_neighbors_graph(keypoints1, radius=0.1)

pd.DataFrame(graph).to_csv('graph.csv', header=False)

#%%
plt.imshow(image)
plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=3, color='red')

#%%
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

#%%
G = nx.from_numpy_array(graph)
#node_attr = {i: descriptors1[i, :] for i in range(descriptors1.shape[0])}
nx.set_node_attributes(G, descriptors1, 'descriptors')
nx.set_node_attributes(G, keypoints1, 'keypoints')

with open("grafo.nx", 'wb') as f:
    pkl.dump(G, f)
    nx.draw(G)
    plt.show()
    print("DONE")
