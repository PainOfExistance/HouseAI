import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from ensemble import MedicalEnsemble
from evaluator import ModelEvaluator
from model import CancerClassifier
# Import our custom classes
from preprocessing import MedicalDataGenerator, MedicalImagePreprocessor
from sklearn.metrics import (classification_report, confusion_matrix,
                             recall_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from trainer import ModelTrainer


def create_generators():
    """Initialize all data generators with medical preprocessing"""
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

    val_test_datagen = MedicalDataGenerator(
        preprocessing_function=lambda x: x/255.0
    )

    train_gen = train_datagen.flow_from_directory(
        "../Data/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        "../Data/valid",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        "../Data/test",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    # Medical-class-aware weighting
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    return train_gen, val_gen, test_gen, dict(enumerate(class_weights))

def train_diverse_models(train_gen, val_gen):
    """Train multiple architectures with medical safety checks"""
    # 1. Our custom CancerClassifier (EfficientNet-based)
    custom_model = CancerClassifier.build_model()
    custom_trainer = ModelTrainer(
        model=custom_model,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights
    )
    custom_trainer.compile_model(lr=1e-4)
    custom_trainer.train(epochs=20)
    custom_model.save('models/custom_effnet.keras')

    # 2. DenseNet201 for diversity
    densenet = tf.keras.applications.DenseNet201(
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = GlobalAveragePooling2D()(densenet.output)
    x = Dense(4, activation='softmax')(x)
    densenet = Model(densenet.inputs, x)
    
    densenet_trainer = ModelTrainer(
        model=densenet,
        train_gen=train_gen,
        val_gen=val_gen,
        class_weights=class_weights
    )
    densenet_trainer.compile_model(lr=1e-4)
    densenet_trainer.train(epochs=15)  # Fewer epochs for secondary model
    densenet.save('models/densenet.keras')

    return ['models/custom_effnet.keras', 'models/densenet.keras']

if __name__ == "__main__":
    # 1. Initialize medical data pipeline
    train_gen, val_gen, test_gen, class_weights = create_generators()
    
    # 2. Train diverse models
    model_paths = train_diverse_models(train_gen, val_gen)
    
    # 3. Create safety-focused ensemble
    ensemble = MedicalEnsemble(model_paths)
    X_val, y_val = next(iter(val_gen))
    ensemble._calculate_recall_weights(X_val, y_val.argmax(axis=1))
    
    # 4. Clinical evaluation
    print("\n=== MEDICAL ENSEMBLE VALIDATION ===")
    ModelEvaluator.generate_report(ensemble, test_gen)
    
    # 5. Save deployment-ready package
    deployment_pkg = {
        'ensemble': {
            'model_paths': model_paths,
            'recall_weights': ensemble.recall_weights,
            'last_validated': datetime.now().isoformat()
        },
        'class_mapping': {v:k for k,v in test_gen.class_indices.items()}
    }
    with open('deployment_package.json', 'w') as f:
        json.dump(deployment_pkg, f)
    
    print("\n✅ Deployment package ready with safety checks")