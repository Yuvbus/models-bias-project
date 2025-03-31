import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from modelvshuman import constants as c


all_files = [os.path.join(c.PERFORMANCES_DIR, f) for f in os.listdir(c.PERFORMANCES_DIR)]
dataframes = [pd.read_csv(file) for file in all_files]
df = pd.concat(dataframes)

models = [
    "resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_IN",
    "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "resnet50",
    "alexnet", "vgg16", "vit_b_16", "clip", "dinov2"
]
datasets = [
    "stylized",
    "cue-conflict-9",
    "filled-silhouette-cue-conflict"
]

df['subj'] = pd.Categorical(df['subj'], categories=models, ordered=True)
df['dataset_name'] = pd.Categorical(df['dataset_name'], categories=datasets, ordered=True)

performance_data = df.pivot_table(index='subj', columns='dataset_name', values='performance', aggfunc='mean').fillna(0)
performance_data = performance_data.reindex(index=models, columns=datasets).values * 100

fig, ax = plt.subplots(figsize=(6, 8))
bar_width = 0.1
space_between_datasets = 4
index = np.arange(len(datasets)) * (len(models) + space_between_datasets) * bar_width

for i, model in enumerate(models):
    plt.bar(index + i * bar_width, performance_data[i], bar_width, label=model)

ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.ylabel('Performances (%)')
plt.title('Models Performances')

ax.set_axisbelow(True)
plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

plt.xticks([])
plt.ylim(-15, 105)
ax.set_yticks(np.arange(0, 105, 5))
ax.spines['left'].set_bounds(0, 100)

images = [
    "images/AdaIn Style Transfer.png",
    "images/Gatys Style Transfer.jpg",
    "images/Filled Silhouette.png"
]
dataset_titles = [
    "SIN",
    "Style-Transfer\nCue-Conflict",
    "Filled-Silhouette\nCue-Conflict"
]

for i, (image_path, dataset_title) in enumerate(zip(images, dataset_titles)):
    image = plt.imread(image_path)
    im = OffsetImage(image, zoom=0.3)
    ab = AnnotationBbox(im, (index[i] + bar_width * (len(models) / 2) - 0.05, -2),
                        frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(ab)
    plt.text(index[i] + bar_width * (len(models) / 2) - 0.05, -20, dataset_title,
             ha='center', va='top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('models_performance_shape_texture.svg', dpi=300)
plt.show()
