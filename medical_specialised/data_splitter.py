import os
import shutil
import random

def split_dataset(source_dir, target_dir, split=(0.7, 0.15, 0.15), seed=42):
    random.seed(seed)
    classes = ["Benign", "Malignant", "Normal"]

    for cls in classes:
        files = os.listdir(os.path.join(source_dir, cls))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(split[0] * n_total)
        n_val = int(split[1] * n_total)

        splits = {
            "train": files[:n_train],
            "valid": files[n_train:n_train+n_val],
            "test":  files[n_train+n_val:]
        }

        for split_name, split_files in splits.items():
            dest_dir = os.path.join(target_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(
                    os.path.join(source_dir, cls, f),
                    os.path.join(dest_dir, f)
                )

# Run it:
split_dataset("data", "Data_2")


