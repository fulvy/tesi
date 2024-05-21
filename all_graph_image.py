from tqdm import tqdm
import pickle as pkl
import networkx as nx
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, ORB
from sklearn.neighbors import NearestNeighbors
import os
import matplotlib.pyplot as plt
import matplotlib
from skimage import morphology
from skimage import exposure

matplotlib.use('Agg')

plt.gray()  # only for visualization


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

for c in tqdm(range(1, 51)):  #ciclo i soggetti
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
            plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=6, color='red')

            normalized = normalize(keypoints1, image)

            #k = min(5, normalized.shape[0] - 1)
            # mi prendo le 5 componenti più vicine (basate su distance()) per costruire il grafo

            model = NearestNeighbors(n_neighbors=2, metric=distance)
            model.fit(normalized)  # coordinate dei punti

            graph = model.radius_neighbors_graph(radius=0.1)
            graph1 = model.kneighbors_graph(n_neighbors=2)
            graph = np.maximum(graph.toarray(), graph1.toarray())

            to_plot = model.radius_neighbors(None, return_distance=False, radius=0.1)
            to_plot1 = model.kneighbors(None, return_distance=False, n_neighbors=2)

            for k in range(len(to_plot)):
                for h in to_plot1[k]:
                    to_plot[k] = np.append(to_plot[k], to_plot1[h])

            plt.imshow(image)

            plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=5, color='red')

            for p in range(to_plot.shape[0]):
                for q in to_plot[p]:
                    cx = [keypoints1[p, 0], keypoints1[q, 0]]
                    cy = [keypoints1[p, 1], keypoints1[q, 1]]
                    a = normalized[p, :]
                    b = normalized[q, :]

                    if np.min(np.abs([a[1] - b[1],
                                      a[1] - (b[1] + 1),
                                      (a[1] + 1) - b[1]
                                      ])) == np.abs(a[1] - b[1]):

                        plt.plot(cy, cx, color='red')
                        pass
                    else:
                        plt.plot(cy, cx, color='blue')
                        pass

            plt.savefig(f'graph_image_SIFT/C{c}_S{s}_I{i}.png')
            plt.clf()
