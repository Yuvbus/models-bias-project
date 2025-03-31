import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_evaluation():
    models = ["resnet50",
              "resnet50_trained_on_SIN",
              "resnet50_trained_on_SIN_and_IN",
              "resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN",
              "alexnet", "vgg16", "dinov2", "vit_b_16", "clip"]
    datasets = ["white-background",
                "grayscale",
                "silhouette",
                "edges",
                "low-pass",
                "high-pass",
                "band-pass",
                "imagenet_validation",
                "stylized"]
    # datasets = [
    #     "texture",
    #     "cue-conflict-9",
    #     "filled-silhouette-cue-conflict",
    # ]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = ["shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    # Plot(plot_types = plot_types, plotting_definition = plotting_def,
    #      figure_directory_name = figure_dirname, dataset_names=["filled-silhouette-cue-conflict"])
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
     figure_directory_name = figure_dirname, dataset_names=["cue-conflict-9"])
    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    # run_plotting()
