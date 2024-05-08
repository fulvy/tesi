import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.color import rgb2gray
from PIL import Image
from skimage import morphology

image = Image.open(
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S1_I1.png").convert("RGB")

image = rgb2gray(image)  # grayscale
plt.imshow(image)
plt.gray()
plt.show()

#%%
plt.hist(image.flatten(), bins=128)
plt.title("Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency (num pixel)")
plt.show()

#%%

image = Image.open(
    "C:\\Users\\fulvi\\DataspellProjects\\tesi\\UBIRISv2\\CLASSES_400_300_Part1\\Iridi\\C1_S1_I1.png").convert("RGB")

image = rgb2gray(image)  # grayscale
image = morphology.diameter_opening(image, diameter_threshold=32)

plt.imshow(image)
plt.gray()
plt.show()
