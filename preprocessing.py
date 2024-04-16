import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from skimage.color import label2rgb, rgb2gray
from skimage.measure import label, regionprops
from sklearn.neighbors import kneighbors_graph

import networkx as nx
import pickle as pkl

plt.gray()  # only for visualization

image = Image.open(
    "C:\\Users\\fulvi\\PycharmProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S1_I12.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.title("iride in grayscale")
plt.show()

n_bins = 13
i = 0
node_features = {}


def discretization(image, n_bins):  # discretazion

    imgmax = image.max()
    imgmin = image.min()
    bins = np.linspace(imgmin, imgmax, n_bins)  # creo un array di n_bins valori equidistanti
    return np.digitize(image, bins, right=False)  # associo ogni pixel ad un bin


image = discretization(image, n_bins)

# plt.gray()
plt.imshow(image)
plt.title("immagine discretizzata")
plt.show()

# create a dataset
result = {'centroide_x': [], 'centroide_y': [], 'bin': [], 'n_comp': [], 'area_relative_to_image': [],
          'area_relative_to_bbox': [], 'bbox_height': [], 'bbox_width': []
          }

# extract components
for bin in range(n_bins):
    bin_image = image <= bin
    label_image = label(bin_image)
    feature_image = label2rgb(label_image, image=image <= bin, bg_label=0)  # coloro le regioni etichettate
    plt.imshow(feature_image)
    plt.title(f"regione {bin}")
    plt.show()

    # extract regions properties
    regions = regionprops(label_image)
    for region in regions:
        # area della regione
        region_area = region.area

        # area rispetto alle dimensioni dell'immagine
        image_area = label_image.shape[0] * label_image.shape[1]
        area_relative_to_image = region_area / image_area

        # area rispetto all'area del bbox
        bbox_area = region.bbox_area
        area_relative_to_bbox = region_area / bbox_area

        # height, weight del bbox
        bbox_height = region.bbox[2] - region.bbox[0]
        bbox_width = region.bbox[3] - region.bbox[1]

        node_features[i] = {'centroide_x': region.centroid[0],
                            'centroide_y': region.centroid[1],
                            'bin': bin,
                            'n_comp': len(regions),

                            'area_relative_to_image': area_relative_to_image,
                            'area_relative_to_bbox': area_relative_to_bbox,
                            'bbox_height': bbox_height,
                            'bbox_width': bbox_width
                            }

        result['centroide_x'].append(region.centroid[0])
        result['centroide_y'].append(region.centroid[1])
        result['bin'].append(bin)
        result['n_comp'].append(len(regions))

        result['area_relative_to_image'].append(area_relative_to_image)
        result['area_relative_to_bbox'].append(area_relative_to_bbox)
        result['bbox_height'].append(bbox_height)
        result['bbox_width'].append(bbox_width)
        #  area / dim immagine
        #  area / area_bbox
        #  dimensioni height weight ( si ricavano da bbox )

# node features to csv
res = pd.DataFrame.from_dict(result)
res.to_csv('result.csv')

# mi prendo le 5 componenti più vicine per costruire il grafo
graph = kneighbors_graph(res[['centroide_x', 'centroide_y']], n_neighbors=5, mode='distance').toarray().astype(int)
pd.DataFrame(graph).to_csv('graph.csv', header=False)

# create a graph with networkx
G = nx.from_numpy_array(graph)
nx.set_node_attributes(G, node_features)

# salvo il grafo con pickle
with open("grafo.nx", 'wb') as f:
    pkl.dump(G, f)
    nx.draw(G)
    plt.show()
    print("DONE")
