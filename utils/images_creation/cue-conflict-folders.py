import os
import shutil
from modelvshuman import constants as c
from os.path import join as pjoin

source_dir = pjoin(c.DATASET_DIR, "cue-conflict1")

for n in [4, 9, 14, 19, 24, 19]:
    dest_dir = os.path.join(c.DATASET_DIR, f"cue-conflict-{n}")
    os.makedirs(dest_dir, exist_ok=True)

for class_dir in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_dir)
    for image_name in os.listdir(class_path):
        base_name, n_with_ext = image_name.rsplit('_', 1)
        n, ext = os.path.splitext(n_with_ext)
        dest_dir = os.path.join(c.DATASET_DIR, f"cue-conflict-{n}", class_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_image_path = os.path.join(dest_dir, base_name + ext)
        shutil.copy2(os.path.join(class_path, image_name), dest_image_path)
        print(f"Copied: {image_name} to {dest_image_path}")

