from pathlib import Path
from PIL import Image
import numpy as np
import os
from modelvshuman import constants as c
from os.path import join as pjoin


def create_filled_silhouette(silhouette_path, texture_path, save_path):
    silhouette = Image.open(silhouette_path).convert('L')
    texture = Image.open(texture_path).convert('RGBA')
    silhouette_array = np.array(silhouette)
    texture_array = np.array(texture)
    white_background = np.full_like(texture_array, 255)
    mask = silhouette_array == 0
    result_array = np.where(mask[:, :, None], texture_array, white_background)
    result_array = Image.fromarray(result_array, 'RGBA')
    result_array.save(save_path, "PNG")


if __name__ == "__main__":
    cue_conflict_folder = pjoin(c.DATASET_DIR, "filled-silhouette-cue-conflict")
    Path(cue_conflict_folder).mkdir(parents=True, exist_ok=True)
    texture_folder_path = pjoin(c.DATASET_DIR, "texture")
    categories_images_dir = pjoin(c.DATASET_DIR, "silhouette")
    for category_dir in os.listdir(categories_images_dir):
        if category_dir in ["airplane", "boat", "car", "keyboard", "oven", "truck"]:
            continue
        subdir_path = os.path.join(categories_images_dir, category_dir)
        cue_conflict_category = os.path.join(cue_conflict_folder, category_dir)
        Path(cue_conflict_category).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(subdir_path):
            for i, image in enumerate(os.listdir(subdir_path)):
                silhouette_path = os.path.join(subdir_path, image)
                for texture_path in list(Path(texture_folder_path).glob("**/*.*")):
                    texture = texture_path.stem
                    if texture.startswith(category_dir):
                        continue
                    new_image_name = f"{category_dir}{i}_{texture}.png"
                    new_image_path = os.path.join(cue_conflict_category, f"{new_image_name}")
                    create_filled_silhouette(silhouette_path, texture_path, new_image_path)
                    print(new_image_name)