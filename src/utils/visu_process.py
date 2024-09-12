import os
from matplotlib import pyplot as plt
import seaborn
import pandas as pd

from utils.directory_process import subdirectory_file_count
from utils.model_process import name_correct


# show barplot
def bar_plot(x, y, plot_property, colors=["blue", "green"]):
    if plot_property["subplot"]:
        plt.subplot(plot_property["subplot"])
    data = pd.DataFrame({"Class": x, "Count": y, "Color": colors[: len(x)]})
    seaborn.barplot(
        x="Class",
        y="Count",
        hue="Class",
        dodge=False,
        data=data,
        palette=colors,
        legend=False,
    )
    plt.title(plot_property["title"], fontsize=plot_property["title_fontsize"])
    plt.xlabel(plot_property["xlabel"], fontsize=plot_property["label_fontsize"])
    plt.ylabel(plot_property["ylabel"], fontsize=plot_property["label_fontsize"])
    plt.xticks(range(len(x)), x)


# show bar plot for count of labels in subdirectory of a directory
def count_bar_plot(master_directory, plot_property):
    dir_name, dir_file_count = subdirectory_file_count(master_directory)
    x = [name_correct(i) for i in dir_name]
    y = dir_file_count
    bar_plot(x, y, plot_property)


# show bar plot for count of labels in subdirectory of a training, validation, testing directory
def show_train_val_test(training_dir, validation_dir, testing_dir, plot_property):
    plt.figure(figsize=plot_property["figsize"])

    title = plot_property["title"]

    plot_property["title"] = title + " (Training)"
    subplot_no = plot_property["subplot"]
    count_bar_plot(training_dir, plot_property)

    plot_property["title"] = title + " (Validation)"
    plot_property["subplot"] = subplot_no + 1
    count_bar_plot(validation_dir, plot_property)

    plot_property["title"] = title + " (Testing)"
    plot_property["subplot"] = subplot_no + 2
    count_bar_plot(testing_dir, plot_property)

    plt.show()


def get_reset_plot_params(
    figsize=(18, 4),
    title="",
    xlabel="",
    ylabel="",
    legends=[],
    title_fontsize=18,
    label_fontsize=13,
    image_file_name="",
    save=False,
    dpi=100,
    update_image=True,
):
    plot_params = {}

    plot_params["figsize"] = figsize

    plot_params["title"] = title

    plot_params["xlabel"] = xlabel
    plot_params["ylabel"] = ylabel

    plot_params["legends"] = legends

    plot_params["title_fontsize"] = title_fontsize
    plot_params["axes.titlesize"] = "small"
    plot_params["label_fontsize"] = label_fontsize

    plot_params["image_file_name"] = image_file_name
    plot_params["save"] = save
    plot_params["update_image"] = update_image

    plot_params["subplot"] = None
    return plot_params


# count number of files in each subdirectory of a directory
def subdirectory_file_count_new(master_directory):
    subdirectories = os.listdir(master_directory)

    subdirectory_names = []
    subdirectory_file_counts = []

    for subdirectory in subdirectories:
        current_directory = os.path.join(master_directory, subdirectory)
        if os.path.isdir(current_directory):
            file_count = len(
                [
                    f
                    for f in os.listdir(current_directory)
                    if os.path.isfile(os.path.join(current_directory, f))
                ]
            )
            subdirectory_names.append(subdirectory)
            subdirectory_file_counts.append(file_count)

    return subdirectory_names, subdirectory_file_counts


# show barplot
def bar_plot_new(x, y, plot_property, colors=["blue", "green", "red"]):
    if "subplot" in plot_property and plot_property["subplot"]:
        plt.subplot(plot_property["subplot"])
    data = pd.DataFrame({"Class": x, "Count": y, "Color": colors[: len(x)]})
    seaborn.barplot(
        x="Class",
        y="Count",
        hue="Class",
        data=data,
        palette=colors,
        dodge=False,
        legend=False,
    )
    plt.title(plot_property["title"], fontsize=plot_property["title_fontsize"])
    plt.xlabel(plot_property["xlabel"], fontsize=plot_property["label_fontsize"])
    plt.ylabel(plot_property["ylabel"], fontsize=plot_property["label_fontsize"])
    plt.xticks(range(len(x)), x)


# show bar plot for count of labels in subdirectory of a directory
def count_bar_plot_new(master_directory, plot_property):
    dir_name, dir_file_count = subdirectory_file_count_new(master_directory)
    x = dir_name
    y = dir_file_count
    bar_plot_new(x, y, plot_property)


# show bar plot for count of labels in subdirectory of a training, validation, testing directory
def show_train_val_test_new(training_dir, validation_dir, testing_dir, plot_property):
    plt.figure(figsize=plot_property["figsize"])

    title = plot_property["title"]

    plot_property["title"] = title + " (Training)"
    plot_property["subplot"] = 131
    count_bar_plot_new(training_dir, plot_property)

    plot_property["title"] = title + " (Validation)"
    plot_property["subplot"] = 132
    count_bar_plot_new(validation_dir, plot_property)

    plot_property["title"] = title + " (Testing)"
    plot_property["subplot"] = 133
    count_bar_plot_new(testing_dir, plot_property)

    plt.show()
