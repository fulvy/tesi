from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import SIFT, ORB
import os
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import exposure


def normalize(dist, image):
    h, w = image.shape
    res = dist.astype(float)
    res[:, 0] /= w
    res[:, 1] /= w
    return res


plt.ioff()  # Disattiva la modalità interattiva di matplotlib

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
            descriptor_extractor.detect_and_extract(image)
            keypoints1 = descriptor_extractor.keypoints
            descriptors1 = descriptor_extractor.descriptors

            plt.gray()
            plt.imshow(image)
            plt.scatter(keypoints1[:, 1], keypoints1[:, 0], s=6, color='red')
            plt.savefig(f'descriptors_SIFT/C{c}_S{s}_I{i}.png')
            plt.close()
