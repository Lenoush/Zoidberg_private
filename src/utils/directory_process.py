import os


def create_directory(directory_path):
    try:
        os.mkdir(directory_path)
        return True
    except FileExistsError:
        print(f"Directory already exists: {directory_path}")
        return False
    except Exception as e:
        print(f"Could not create directory: {directory_path} due to {e}")
        return False


def clean_path(list_path):
    for path in list_path:
        ds_store_path = os.path.join(path, ".DS_Store")
        if os.path.exists(ds_store_path):
            os.remove(ds_store_path)


def dir_file_count(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


def subdirectory_file_count(master_directory):
    subdirectories = os.listdir(master_directory)

    subdirectory_names = []
    subdirectory_file_counts = []

    for subdirectory in subdirectories:
        current_directory = os.path.join(master_directory, subdirectory)
        file_count = len(os.listdir(current_directory))
        subdirectory_names.append(subdirectory)
        subdirectory_file_counts.append(file_count)

    return subdirectory_names, subdirectory_file_counts
