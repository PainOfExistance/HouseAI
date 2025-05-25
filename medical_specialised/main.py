import json
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from ensemble import MedicalEnsemble
from evaluator import ModelEvaluator
from model import CancerClassifier
from preprocessing import MedicalDataGenerator, MedicalImagePreprocessor
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from trainer import ModelTrainer


def create_data_generator(preprocessing_function, augment=False):
    """Helper function to create a data generator."""
    if augment:
        return MedicalDataGenerator(
            preprocessing_function=preprocessing_function,
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
    return MedicalDataGenerator(preprocessing_function=preprocessing_function)


def create_generators(
    dataset_dir,
    image_size=(224, 224),
    batch_size=32,
    val_split=0.1,
    test_split=0.2,
    seed=42
):
    # 1. Get all image paths and labels from class folders with proper extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    class_folders = sorted(glob(os.path.join(dataset_dir, "*")))
    class_names = [os.path.basename(f) for f in class_folders if os.path.isdir(f)]
    print(f"Found {len(class_names)} classes: {class_names}")

    image_paths = []
    labels = []
    for class_idx, class_folder in enumerate(class_folders):
        class_images = []
        for ext in image_extensions:
            class_images.extend(glob(os.path.join(class_folder, ext)))
        if not class_images:
            print(f"Warning: No images found in {class_folder}")
            continue
        image_paths.extend(class_images)
        labels.extend([class_names[class_idx]] * len(class_images))

    if not image_paths:
        raise ValueError("No images found in the dataset directory")

    # 2. Stratified splitting (train/val/test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=val_split + test_split,
        random_state=seed,
        stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=test_split/(val_split + test_split),
        random_state=seed,
        stratify=temp_labels
    )

    # 3. Create dataframes
    train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
    val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})
    test_df = pd.DataFrame({'filename': test_paths, 'class': test_labels})

    # 4. Data generators
    train_datagen = MedicalDataGenerator(
        preprocessing_function=lambda x: x/255.0,
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
    valid_test_datagen = MedicalDataGenerator(
        preprocessing_function=lambda x: x/255.0
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb',
        interpolation='lanczos',
        classes=class_names
    )

    validation_generator = valid_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )

    test_generator = valid_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filename',
        y_col='class',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=class_names
    )

    # 5. Class weights
    class_counts = train_generator.classes
    unique_classes = np.unique(class_counts)
    total_samples = len(class_counts)
    class_weights = {}
    for cls in unique_classes:
        class_weights[cls] = total_samples / (len(unique_classes) * np.sum(class_counts == cls))

    print(f"\nClass Weights: {class_weights}")
    train_generator.reset()

    return train_generator, validation_generator, test_generator, class_weights


def train_model(model, train_gen, val_gen, class_weights, lr, epochs, save_path):
    """Helper function to compile, train, and save a model."""
    trainer = ModelTrainer(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights
    )
    trainer.compile_model(lr=lr)
    trainer.train(epochs=epochs)
    model.save(save_path)


def train_diverse_models(train_gen, val_gen, class_weights):
    """Train multiple architectures with medical safety checks."""
    # Train custom CancerClassifier
    custom_model = CancerClassifier.build_model()
    train_model(custom_model, train_gen, val_gen, class_weights, lr=1e-4, epochs=30, save_path='models/custom_effnet.keras')

    # Train DenseNet201
    densenet = tf.keras.applications.DenseNet201(include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(densenet.output)
    x = Dense(4, activation='softmax')(x)
    densenet = Model(densenet.inputs, x)
    train_model(densenet, train_gen, val_gen, class_weights, lr=1e-4, epochs=20, save_path='models/densenet.keras')

    return ['models/custom_effnet.keras', 'models/densenet.keras']


if __name__ == "__main__":
    # Initialize medical data pipeline
    train_gen, val_gen, test_gen, class_weights = create_generators()

    # Train diverse models
    model_paths = train_diverse_models(train_gen, val_gen, class_weights)

    # Create safety-focused ensemble
    ensemble = MedicalEnsemble(model_paths)
    X_val, y_val = next(iter(val_gen))
    ensemble._calculate_recall_weights(X_val, y_val.argmax(axis=1))

    # Clinical evaluation
    print("\n=== MEDICAL ENSEMBLE VALIDATION ===")
    ModelEvaluator.generate_medical_report(ensemble, test_gen)

    # Save deployment-ready package
    deployment_pkg = {
        'ensemble': {
            'model_paths': model_paths,
            'recall_weights': ensemble.recall_weights,
            'last_validated': datetime.now().isoformat()
        },
        'class_mapping': {v: k for k, v in test_gen.class_indices.items()}
    }
    with open('deployment_package.json', 'w') as f:
        json.dump(deployment_pkg, f)

    print("\n✅ Deployment package ready with safety checks")