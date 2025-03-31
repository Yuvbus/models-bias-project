import os
import json
from modelvshuman import constants as c
from os.path import join as pjoin

def delete_unlisted_images(folder_path, images_dict):
    for subfolder, _, files in os.walk(folder_path):
        subfolder_name = os.path.basename(subfolder)
        if subfolder_name not in images_dict:
            continue

        valid_images = set(images_dict[subfolder_name])
        files_to_delete = [os.path.join(subfolder, file) for file in files if file not in valid_images]

        for file_path in files_to_delete:
            print(f"Deleting image: {file_path}")
            os.remove(file_path)


with open("consistent_images.json", 'r') as file:
    images_dict = json.load(file)

folder_path  = pjoin(c.DATASET_DIR, "white-background")
delete_unlisted_images(folder_path, images_dict)
