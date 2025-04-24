import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from ensemble import MedicalEnsemble
from evaluator import ModelEvaluator
from model import CancerClassifier
from preprocessing import MedicalDataGenerator, MedicalImagePreprocessor
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score)
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


def create_generators():
    """Initialize all data generators with medical preprocessing."""
    preprocessing_function = lambda x: x / 255.0
    train_datagen = create_data_generator(preprocessing_function, augment=True)
    val_test_datagen = create_data_generator(preprocessing_function)

    def create_flow(datagen, directory, shuffle):
        return datagen.flow_from_directory(
            directory,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            shuffle=shuffle
        )

    train_gen = create_flow(train_datagen, "../Data/train", shuffle=True)
    val_gen = create_flow(val_test_datagen, "../Data/valid", shuffle=False)
    test_gen = create_flow(val_test_datagen, "../Data/test", shuffle=False)

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    return train_gen, val_gen, test_gen, dict(enumerate(class_weights))


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