import pandas as pd
import glob
import json
from modelvshuman import constants as c
from os.path import join as pjoin

base_dir = pjoin(c.DATASET_DIR, "white-background")

csv_files = glob.glob(f"{base_dir}/*.csv")
category_consistent_images = {}

first_csv = pd.read_csv(csv_files[0])
for category in first_csv["category"].unique():
    category_consistent_images[category] = set(
        first_csv[first_csv["category"] == category]
        .query("category == object_response")["imagename"]
    )

for csv_file in csv_files[1:]:
    df = pd.read_csv(csv_file)

    for category in category_consistent_images.keys():
        valid_images = set(
            df[df["category"] == category]
            .query("category == object_response")["imagename"]
        )
        category_consistent_images[category].intersection_update(valid_images)

for category in category_consistent_images:
    category_consistent_images[category] = list(category_consistent_images[category])

with open("consistent_images.json", "w") as f:
    json.dump(category_consistent_images, f, indent=4)

