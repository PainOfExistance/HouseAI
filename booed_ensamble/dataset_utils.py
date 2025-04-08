# dataset_utils.py
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import SAMPLE_SIZE

import random


def bootstrap_sample_with_errors(filepaths, labels, errors, subset_size=SAMPLE_SIZE):

    full_pool = list(zip(filepaths, labels))

    # Start with the errors
    err_paths, err_labels = zip(*errors) if errors else ([], [])
    errors_combined = list(zip(err_paths, err_labels))

    remaining = subset_size - len(errors_combined)
    if remaining < 0:
        # Too many errors? Just take a random subset of the errors
        errors_combined = random.sample(errors_combined, k=subset_size)
        remaining = 0

    # Fill the rest with bootstrap sampling
    random_samples = random.choices(full_pool, k=remaining)

    final_samples = errors_combined + random_samples
    X, y = zip(*final_samples)
    return list(X), list(y)


def get_generator_from_paths(filepaths, labels, image_size, batch_size, augment=False):
    datagen = ImageDataGenerator(rescale=1./255)
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            zoom_range=0.1,
            horizontal_flip=True
        )

    # Create temporary directory for generator (Keras needs dirs)
    temp_dir = "./temp_training_data"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    class_dirs = list(set(labels))
    for cls in class_dirs:
        os.makedirs(os.path.join(temp_dir, str(cls)), exist_ok=True)

    for img_path, label in zip(filepaths, labels):
        filename = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(temp_dir, str(label), filename))

    return datagen.flow_from_directory(
        temp_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

def create_test_generator(test_dir, image_size=(224, 224), batch_size=32):
    test_datagen = ImageDataGenerator(rescale=1./255)
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

def create_validation_generator(val_dir, image_size=(224, 224), batch_size=32):
    val_datagen = ImageDataGenerator(rescale=1./255)
    return val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )


def load_all_filepaths_and_labels(data_dir):

    filepaths = []
    labels = []
    class_indices = {}

    for idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_indices[class_name] = idx

        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_path, fname))
                labels.append(idx)

    return np.array(filepaths), np.array(labels)
