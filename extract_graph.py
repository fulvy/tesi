from tqdm import tqdm
import pickle as pkl
import networkx as nx
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, ORB
from sklearn.neighbors import NearestNeighbors
import os
from skimage import morphology
from skimage import exposure


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

            p1, p2 = np.percentile(image, [2, 90])
            image = exposure.rescale_intensity(image, in_range=(p1, p2))
            image = morphology.diameter_opening(image, diameter_threshold=32)

            descriptor_extractor = SIFT()
            #descriptor_extractor = ORB()

            descriptor_extractor.detect_and_extract(image)
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors

            normalized = normalize(keypoints1, image)

            #k = min(5, normalized.shape[0] - 1)

            model = NearestNeighbors(n_neighbors=5, metric=distance)
            model.fit(normalized)  # coordinate dei punti

            # 0.1 (10% dell'immagine): mi prendo le componenti più vicine a distanza normalizzata di 0.1
            graph = model.radius_neighbors_graph(radius=0.3)
            graph1 = model.kneighbors_graph(n_neighbors=2)
            graph = np.maximum(graph.toarray(), graph1.toarray())

            G = nx.from_numpy_array(graph)
            node_attr = {i: descriptors1[i, :] for i in range(descriptors1.shape[0])}
            nx.set_node_attributes(G, descriptors1, 'feature')

            with open(f'grafi/C{c}_S{s}_I{i}.pkl', 'wb') as f:
                pkl.dump(G, f)
