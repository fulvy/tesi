import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.color import label2rgb, rgb2gray
from skimage.measure import label, regionprops
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from skimage.segmentation import clear_border

from PIL import Image
import networkx as nx
import pickle as pkl

#%%
plt.gray()  # only for visualization

image = Image.open(
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S1_I5.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.title("iride in grayscale")
plt.show()

#%%
n_bins = 13
i = 0
node_features = {}

#%%
def discretization(image, n_bins):  # discretazion

    imgmax = image.max()
    imgmin = image.min()
    bins = np.linspace(imgmin, imgmax, n_bins)  # creo un array di n_bins valori equidistanti
    return np.digitize(image, bins, right=False)  # associo ogni pixel ad un bin


def distance(a, b):  # distanza minima tra due punti
    translated_a = a
    translated_b = b
    translated_a[1] += 1
    translated_b[1] += 1
    return np.min([np.linalg.norm(a - b), np.linalg.norm(translated_a - b),
                   np.linalg.norm(a - translated_b)])


image = discretization(image, n_bins)

plt.imshow(image)
plt.title("immagine discretizzata")
plt.show()

# creo il dataset con queste features date da regionprops()
result = {'centroide_x': [], 'centroide_y': [], 'bin': [], 'n_comp': [], 'area_relative_to_image': [],
          'area_relative_to_bbox': [], 'bbox_height': [], 'bbox_width': []
          }

# extract components
for bin in range(n_bins):
    bin_image = image <= bin
    cleared_bin_image = clear_border(bin_image)  # pulisco i bordi con clear_border di skimage
    label_image = label(cleared_bin_image)
    h, w = label_image.shape  # to normalize
    feature_image = label2rgb(label_image, image=image <= bin, bg_label=0)  # coloro le regioni etichettate #
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

        node_features[i] = {'centroide_x': region.centroid[0] / h,
                            'centroide_y': region.centroid[1] / w,
                            'bin': bin,
                            'n_comp': len(regions),

                            'area_relative_to_image': area_relative_to_image,
                            'area_relative_to_bbox': area_relative_to_bbox,
                            'bbox_height': bbox_height,
                            'bbox_width': bbox_width
                            }

        result['centroide_x'].append(region.centroid[0] / h)
        result['centroide_y'].append(region.centroid[1] / w)
        result['bin'].append(bin)
        result['n_comp'].append(len(regions))

        result['area_relative_to_image'].append(area_relative_to_image)
        result['area_relative_to_bbox'].append(area_relative_to_bbox)
        result['bbox_height'].append(bbox_height)
        result['bbox_width'].append(bbox_width)

# node features to csv
res = pd.DataFrame.from_dict(result)
res.to_csv('result.csv')

# mi prendo le 5 componenti più vicine (basate su disyance()) per costruire il grafo
model = NearestNeighbors(n_neighbors=5, metric=distance)
model.fit(res[['centroide_x', 'centroide_y']])  # coordinate dei punti
graph = model.kneighbors_graph(res[['centroide_x', 'centroide_y']])
# graph = kneighbors_graph(res[['centroide_x', 'centroide_y']], n_neighbors=5, mode='distance').toarray().astype(int)

# 0.1 (10% dell'immagine): mi prendo le componenti più vicine a distanza normalizzata di 0.1
# graph = model.radius_neighbors_graph(res[['centroide_x', 'centroide_y']], radius=0.1)

pd.DataFrame(graph).to_csv('graph.csv', header=False)

# create a graph with networkx
G = nx.from_numpy_array(graph)
nx.set_node_attributes(G, node_features)

# salvo il grafo con pickle
with open("grafo.nx", 'wb') as f:
    pkl.dump(G, f)
    # G = pkl.load(f)
    nx.draw(G)
    plt.show()
    print("DONE")
