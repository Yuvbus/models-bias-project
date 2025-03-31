import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import canny
from scipy.ndimage import gaussian_filter
from pathlib import Path
import numpy as np
import os
from modelvshuman import constants as c
from os.path import join as pjoin

base_dir = pjoin(c.DATASET_DIR, "white-background")

gray_folder = pjoin(c.DATASET_DIR, "grayscale")
Path(gray_folder).mkdir(parents=True, exist_ok=True)
canny_folder = pjoin(c.DATASET_DIR, "edges")
Path(canny_folder).mkdir(parents=True, exist_ok=True)
silhouette_folder = pjoin(c.DATASET_DIR, "silhouette")
Path(silhouette_folder).mkdir(parents=True, exist_ok=True)


def convert_to_grayscale(image_path):
    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = image[..., :3]
    grayscale_image = color.rgb2gray(image)
    return grayscale_image

def stack_grayscale(grayscale_image):
    stacked_image = np.stack([grayscale_image] * 3, axis=-1)
    return stacked_image

def canny_edges(gray_image):
    smoothed_image = gaussian_filter(gray_image, sigma=2)
    edges = canny(smoothed_image, sigma=1.0)
    inverted_edges = 1 - edges
    return inverted_edges

def silhouette(gray_image):
    grayscale_image = (np.where(gray_image != 1, 0, gray_image) * 255).astype(np.uint8)
    stacked_image = np.stack([grayscale_image] * 3, axis=-1)
    return stacked_image

def save_image(image_path, image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.imsave(image_path, image, cmap='gray')


for root, _, images in os.walk(base_dir):
    subdir = os.path.relpath(root, base_dir)
    Path(os.path.join(gray_folder, subdir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(canny_folder, subdir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(silhouette_folder, subdir)).mkdir(parents=True, exist_ok=True)
    for image in images:
        image_path = os.path.join(subdir, image)
        original_path = os.path.join(base_dir, image_path)
        grayscale_image = convert_to_grayscale(original_path)
        save_image(os.path.join(gray_folder, image_path), stack_grayscale(grayscale_image))
        save_image(os.path.join(canny_folder, image_path), canny_edges(grayscale_image))
        save_image(os.path.join(silhouette_folder, image_path), silhouette(grayscale_image))
