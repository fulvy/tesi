from tqdm import tqdm
import pickle as pkl
import networkx as nx
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT
from sklearn.neighbors import NearestNeighbors
import os


#%% functions
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


#%% estraggo i soggetti

for c in tqdm(range(1, 261)):  #ciclo i soggetti
    for s in [1, 2]:  #solo un occhio [1], altrimenti se voglio fare S1 e S2: for s in (1,2)
        for i in range(1, 16):  #ciclo l'indice
            curr = f'UBIRISv2/CLASSES_400_300_Part1/Iridi/C{c}_S{s}_I{i}.png'
            if not os.path.isfile(curr):
                print(curr + ' not exists')
                continue
            image = Image.open(curr).convert("RGB")
            image = rgb2gray(image)  # grayscale

            descriptor_extractor = SIFT()
            #descriptor_extractor = ORB()

            descriptor_extractor.detect_and_extract(image)
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors

            normalized = normalize(keypoints1, image)

            k = min(5, normalized.shape[0] - 1)
            # mi prendo le 5 componenti più vicine (basate su distance()) per costruire il grafo
            model = NearestNeighbors(n_neighbors=k, metric=distance)
            model.fit(normalized)  # coordinate dei punti
            graph = model.kneighbors_graph(n_neighbors=k)  #senza cappi

            G = nx.from_numpy_array(graph)
            node_attr = {i: descriptors1[i, :] for i in range(descriptors1.shape[0])}
            nx.set_node_attributes(G, descriptors1, 'feature')

            with open(f'grafi/C{c}_S{s}_I{i}.pkl', 'wb') as f:
                pkl.dump(G, f)
