import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modelvshuman import constants as c

# Load data
all_files = [os.path.join(c.PERFORMANCES_DIR, f) for f in os.listdir(c.PERFORMANCES_DIR)]
df = pd.concat([pd.read_csv(file) for file in all_files])

# Define models and datasets
models = ["resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_IN",
          "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "resnet50",
          "alexnet", "vgg16", "vit_b_16", "clip", "dinov2"]

datasets = ["imagenet_validation", "texture", "white-background", "grayscale",
            "silhouette", "edges", "low-pass", "high-pass", "band-pass"]

# Pivot table
df['subj'] = pd.Categorical(df['subj'], categories=models, ordered=True)
df['dataset_name'] = pd.Categorical(df['dataset_name'], categories=datasets, ordered=True)
performance_data = df.pivot_table(index='subj', columns='dataset_name', values='performance', aggfunc='mean').fillna(0) * 100

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Convert dataframe to a table
table = ax.table(cellText=performance_data.round(1).values,
                 colLabels=performance_data.columns,
                 rowLabels=performance_data.index,
                 cellLoc='center', loc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(datasets))])  # Adjust width

# Save as image file
plt.savefig("model_performance.svg", dpi=300, bbox_inches='tight')
plt.show()
print("Chart saved as 'model_performance.svg'")




# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
#
# from modelvshuman import constants as c
#
# # Load data
# all_files = [os.path.join(c.PERFORMANCES_DIR, f) for f in os.listdir(c.PERFORMANCES_DIR)]
# dataframes = [pd.read_csv(file) for file in all_files]
# df = pd.concat(dataframes)
#
# # Define models and datasets
# models = [
#     "resnet50_trained_on_SIN", "resnet50_trained_on_SIN_and_IN",
#     "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN", "resnet50",
#     "alexnet", "vgg16", "vit_b_16", "clip", "dinov2"
# ]
# datasets = [
#     "imagenet_validation", "texture", "white-background", "grayscale",
#     "silhouette", "edges", "low-pass", "high-pass", "band-pass"
# ]
#
# # Categorize models and datasets
# df['subj'] = pd.Categorical(df['subj'], categories=models, ordered=True)
# df['dataset_name'] = pd.Categorical(df['dataset_name'], categories=datasets, ordered=True)
#
# # Pivot data to matrix format
# performance_data = df.pivot_table(index='subj', columns='dataset_name', values='performance', aggfunc='mean').fillna(0)
# performance_data = performance_data.reindex(index=models, columns=datasets) * 100  # Convert to percentages
#
# # Create the figure
# fig, ax = plt.subplots(figsize=(18, 6))
# ax.set_frame_on(False)
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
#
# # Generate the table
# table = plt.table(cellText=np.round(performance_data.values, 1),  # Round numbers
#                   rowLabels=performance_data.index,
#                   colLabels=performance_data.columns,
#                   cellLoc='center', loc='center',
#                   colWidths=[0.12] * len(datasets))  # Adjust column widths
#
# # Format table
# table.auto_set_font_size(False)
# table.set_fontsize(9)
# table.auto_set_column_width([i for i in range(len(datasets))])
#
# # Highlight header row and column
# for (i, j), cell in table.get_celld().items():
#     if i == 0 or j == -1:
#         cell.set_fontsize(10)
#         cell.set_text_props(weight='bold')
#
# # Title
# plt.title("Model Performance Across Datasets (%)", fontsize=12, fontweight='bold')
#
# # Save and show
# plt.savefig("models_performance_table.svg", dpi=300, bbox_inches="tight")
# plt.show()
