import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from modelvshuman import constants as c
from os.path import join as pjoin


def get_fourier(image):
    non_zero_centered_fourier = np.fft.fft2(image)
    return np.fft.fftshift(non_zero_centered_fourier)


def get_low_pass_filter(image_height, image_width):
    low_pass_filter = np.zeros((image_height, image_width), np.uint8)
    center = image_height // 2, image_width // 2
    cv2.circle(img=low_pass_filter, center=center, radius=40, color=1, thickness=-1)
    return low_pass_filter


def inverse_fourier(fourier, mask):
    filtered_fourier = fourier * mask
    f_ishift_low = np.fft.ifftshift(filtered_fourier)
    image_low_pass = np.fft.ifft2(f_ishift_low)
    return np.abs(image_low_pass)


def create_high_and_low_pass_per_image(image_path, low_pass_path, high_pass_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fourier = get_fourier(image)
    low_pass_filter = get_low_pass_filter(image.shape[0], image.shape[1])
    high_pass_filter = 1 - low_pass_filter
    plt.imsave(low_pass_path, inverse_fourier(fourier, low_pass_filter), cmap='gray')
    plt.imsave(high_pass_path, inverse_fourier(fourier, high_pass_filter), cmap='gray')


def create_band_pass_image(image_path, band_pass_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fourier = get_fourier(image)
    band_pass_filter = get_band_pass_filter(image.shape[0], image.shape[1])
    plt.imsave(band_pass_path, inverse_fourier(fourier, band_pass_filter), cmap='gray')


def get_band_pass_filter(image_height, image_width, low_radius=10, high_radius=50):
    center = image_height // 2, image_width // 2

    low_pass_filter = np.zeros((image_height, image_width), np.uint8)
    cv2.circle(img=low_pass_filter, center=center, radius=low_radius, color=1, thickness=-1)
    high_pass_filter = np.zeros((image_height, image_width), np.uint8)
    cv2.circle(img=high_pass_filter, center=center, radius=high_radius, color=1, thickness=-1)

    return high_pass_filter - low_pass_filter


def create_colored_low_pass_image(image_path, colored_low_pass_path):
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > 250
    low_pass_filter = get_low_pass_filter(image.shape[0], image.shape[1])
    low_pass_channels = []
    for i in range(3):
        fourier = get_fourier(image[:, :, i])
        low_pass_image = inverse_fourier(fourier, low_pass_filter)
        low_pass_channels.append(low_pass_image.astype(np.uint8))
    low_pass_colored = np.stack(low_pass_channels, axis=-1)
    low_pass_colored[mask] = [255, 255, 255]
    plt.imsave(colored_low_pass_path, low_pass_colored)

if __name__ == "__main__":
    high_pass_folder = pjoin(c.DATASET_DIR, "high-pass")
    Path(high_pass_folder).mkdir(parents=True, exist_ok=True)
    low_pass_folder = pjoin(c.DATASET_DIR, "low-pass")
    Path(low_pass_folder).mkdir(parents=True, exist_ok=True)
    band_pass_folder = pjoin(c.DATASET_DIR, "band-pass")
    Path(band_pass_folder).mkdir(parents=True, exist_ok=True)

    white_background_images_dir = high_pass_folder = pjoin(c.DATASET_DIR, "white-background")

    for root, _, images in os.walk(white_background_images_dir):
        subdir = os.path.relpath(root, white_background_images_dir)
        Path(os.path.join(high_pass_folder, subdir)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(low_pass_folder, subdir)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(band_pass_folder, subdir)).mkdir(parents=True, exist_ok=True)
        for image in images:
            image_path_suffix = os.path.join(subdir, image)
            original_path = os.path.join(white_background_images_dir, image_path_suffix)
            low_pass_path = os.path.join(low_pass_folder, image_path_suffix)
            high_pass_path = os.path.join(high_pass_folder, image_path_suffix)
            create_high_and_low_pass_per_image(original_path, low_pass_path, high_pass_path)
            band_pass_path = os.path.join(band_pass_folder, image_path_suffix)
            create_band_pass_image(original_path, band_pass_path)
