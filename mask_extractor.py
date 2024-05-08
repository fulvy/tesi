import os
import cv2
import numpy as np
from skimage.feature import SIFT
import matplotlib.pyplot as plt
import mask
from mask import *

best_folder = "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Best"
mask_folder = "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Maschere"
#output_folder = "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Mask_Descriptors"

#%% ciclo su ogni immagine nella cartella Best
for filename in os.listdir(best_folder):
    if filename.endswith(".tiff"):
        iride_image = cv2.imread(os.path.join(best_folder, filename), cv2.IMREAD_GRAYSCALE)
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_image = cv2.imread(os.path.join(mask_folder, mask_filename), cv2.IMREAD_GRAYSCALE)

        masked_image = mask.apply_mask(iride_image, mask_image)

        descriptor_extractor = SIFT()
        descriptor_extractor.detect_and_extract(masked_image)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors

        plt.figure()
        plt.imshow(masked_image, cmap='gray')
        plt.scatter(keypoints[:, 1], keypoints[:, 0], s=5, color='red')
        #plt.show()
        plt.savefig(best_folder + "\\masked_" + filename)

print("finish")
