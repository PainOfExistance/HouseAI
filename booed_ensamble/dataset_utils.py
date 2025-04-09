# dataset_utils.py
import cv2
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import SAMPLE_SIZE
import random
from specialised import MedicalDataGenerator


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

def medical_preprocess(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Gamma correction
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")
    image = cv2.LUT(image, table)

    return image


def get_generator_from_paths(filepaths, labels, image_size, batch_size, augment=False):
    import shutil

    if augment:
        datagen = MedicalDataGenerator(
            preprocessing_function=lambda x: x / 255.0,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            fill_mode='constant',
            cval=0
        )
    else:
        datagen = MedicalDataGenerator(preprocessing_function=lambda x: x / 255.0)

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
    val_datagen = MedicalDataGenerator(preprocessing_function=lambda x: x / 255.0)
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
