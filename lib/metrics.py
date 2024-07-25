import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_curve, auc


def intra_inter_distance(X, y):
    dmat = pairwise_distances(X, metric='euclidean')
    intra = []
    inter = []

    for i in range(dmat.shape[0]):
        for j in range(i + 1, dmat.shape[1]):
            if y[i] != y[j]:
                inter.append(dmat[i, j])
            else:
                intra.append(dmat[i, j])

    # distanza media i grafi della stessa classe / distanza media tra i grafi di classi diverse
    return np.mean(intra) / np.mean(inter)


def compute_validation_metrics(probe_embeddings, gallery_embeddings, probe_labels, gallery_labels):

    dm = pairwise_distances(probe_embeddings, gallery_embeddings)

    distances = [-dm[i, j] for i in range(dm.shape[0]) for j in range(dm.shape[1])]
    y_true = [probe_labels[i] == gallery_labels[j] for i in range(dm.shape[0]) for j in range(dm.shape[1])]

    #print(y_true)
    far, tpr, threshold = roc_curve(y_true, distances)
    frr = 1 - tpr
    roc_auc = auc(far, tpr)  # ??

    # theoretically eer from far and eer from frr should be identical, but they can be slightly differ in reality
    eer_1 = far[np.nanargmin(np.absolute((frr - far)))]
    eer_2 = frr[np.nanargmin(np.absolute((frr - far)))]

    # return the mean of eer from far and from frr
    eer = (eer_1 + eer_2) / 2
    return far, frr, roc_auc, eer
