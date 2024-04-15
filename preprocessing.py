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
    "C:\\Users\\fulvi\\PycharmProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S1_I4.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.show()

n_bins = 13


def discretization(image, n_bins):  # discretazion

    imgmax = image.max()
    imgmin = image.min()
    bins = np.linspace(imgmin, imgmax, n_bins)
    return np.digitize(image, bins, right=False)


image = discretization(image, n_bins)

plt.gray()
plt.imshow(image)
plt.show()

# create a dataset
result = {'centroide_x': [], 'centroide_y': [], 'bin': [], 'n_comp': []}
i = 0
node_features = {}
# extract components
for bin in range(n_bins):
    bin_image = image <= bin
    label_image = label(bin_image)
    feature_image = label2rgb(label_image, image=image <= bin, bg_label=0)
    #plt.imshow(feature_image)
    #plt.show()

    # extract regions properties
    regions = regionprops(label_image)
    for region in regions:
        node_features[i] = {'centroide_x': region.centroid[0],
                            'centroide_y': region.centroid[1],
                            'bin': bin,
                            'n_comp': len(regions)}

        result['centroide_x'].append(region.centroid[0])
        result['centroide_y'].append(region.centroid[1])
        result['bin'].append(bin)
        result['n_comp'].append(len(regions))
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

# salvi con pickle
with open("grafo.nx", 'wb') as f:
    pkl.dump(G, f)
    nx.draw(G)
    plt.show()
    print("DONE")
