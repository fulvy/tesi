import numpy as np
from skimage.transform import resize


def apply_mask(img, mask):
    mask[mask == np.min(mask)] = 1
    mask[mask != 1] = 0

    temp = np.copy(img)
    temp[mask == 0] = 0

    a0 = mask.sum(axis=0)  #
    a1 = mask.sum(axis=1)  #
    i, j = np.min(np.argwhere(a0 > 0)), np.max(np.argwhere(a0 > 0))
    p, q = np.min(np.argwhere(a1 > 0)), np.max(np.argwhere(a1 > 0))
    return temp[p:q, i:j]

    return temp
