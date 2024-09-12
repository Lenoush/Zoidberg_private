import os
import numpy as np
import PIL.Image as img
import random
import shutil
from sklearn.utils.class_weight import compute_class_weight


def get_weight(y):
    class_weight_current = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )
    return class_weight_current


def load_and_getName(train_path_normal):
    jpeg_files = []
    if os.path.exists(train_path_normal):
        for root, dirs, files in os.walk(train_path_normal, topdown=False):
            for file in files:
                if file.endswith(".jpeg"):
                    jpeg_file = os.path.join(root, file)
                    jpeg_files.append(jpeg_file)
        return jpeg_files
    else:
        return "error de chemin"


def load_and_getName_Pneumonia(train_path_normal):
    virus_files = []
    bacteria_files = []

    if os.path.exists(train_path_normal):
        for root, _, files in os.walk(train_path_normal, topdown=False):
            for file in files:
                if file.endswith(".jpeg"):
                    file_path = os.path.join(root, file)
                    if "virus" in file:
                        virus_files.append(file_path)
                    elif "bacteria" in file:
                        bacteria_files.append(file_path)
        return virus_files, bacteria_files
    else:
        return "error de chemin"


def name_to_data(list_files, binaire):
    dataset = []
    for i in range(len(list_files)):
        root = list_files[i]
        image = img.open(root)
        tmp = np.array(image)
        dataset.append((tmp, binaire))
    return dataset


def mix_dataset_same_labels_Pneumonia(*list_data_path: str):
    mixed_dataset_virus = []
    mixed_dataset_bacteria = []
    for path in list_data_path:
        dataset_virus, dataset_bacteria = load_and_getName_Pneumonia(path)
        mixed_dataset_bacteria.extend(dataset_bacteria)
        mixed_dataset_virus.extend(dataset_virus)
    random.shuffle(mixed_dataset_virus)
    random.shuffle(mixed_dataset_bacteria)
    return mixed_dataset_virus, mixed_dataset_bacteria


def mix_dataset_same_labels(*list_data_path: str):
    mixed_dataset = []
    for path in list_data_path:
        dataset = load_and_getName(path)
        mixed_dataset.extend(dataset)
    random.shuffle(mixed_dataset)
    return mixed_dataset


def load_dataset_train(path_normal, path_malade):
    load = load_and_getName(path_normal)
    dataset_sain = name_to_data(load, 0)

    load_malade = load_and_getName(path_malade)
    dataset_malade = name_to_data(load_malade, 1)

    return dataset_sain, dataset_malade


def save_mixed_dataset(mixed_dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metadata = []
    for idx, (image_array, label) in enumerate(mixed_dataset):
        image = img.fromarray(image_array)
        filename = f"image_{idx}_{label}.jpeg"
        file_path = os.path.join(save_path, filename)
        image.save(file_path)
        metadata.append((filename, label))


def distribuer_images(
    paths,
    train_path,
    test_path,
    vali_path,
    train_ratio=0.7,
    test_ratio=0.15,
    vali_ratio=0.15,
    seed=10,
):
    assert (
        train_ratio + test_ratio + vali_ratio == 1.0
    ), "Les ratios doivent totaliser 1.0"

    random.seed(seed)
    random.shuffle(paths)

    total_images = len(paths)
    train_end = int(train_ratio * total_images)
    test_end = train_end + int(test_ratio * total_images)

    for image in paths[:train_end]:
        shutil.copy(image, train_path)
    for image in paths[train_end:test_end]:
        shutil.copy(image, test_path)
    for image in paths[test_end:]:
        shutil.copy(image, vali_path)


def clear_directory(directory_path):
    dirs_files = os.listdir(directory_path)

    for item in dirs_files:
        item_path = directory_path + item

        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)

    return True
