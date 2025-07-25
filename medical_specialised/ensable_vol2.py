import os

import numpy as np
import tensorflow as tf
from evaluator import ModelEvaluator
from tensorflow.keras.models import load_model

from main import create_generators

MODEL_DIR = "models"


def load_all_models():
    return [
        load_model(os.path.join(MODEL_DIR, "medical_specialised.keras")),  # EfficientNet
        load_model(os.path.join(MODEL_DIR, "mobilenetv3_best.keras")),
        load_model(os.path.join(MODEL_DIR, "resnet50_best.keras")),
    ]


def predict_with_hybrid(models, x):
    preds = [model.predict(x, verbose=0) for model in models]
    return np.mean(preds, axis=0)


if __name__ == "__main__":
    _, _, test_gen, _ = create_generators("../Data2")

    models = load_all_models()

    x_test, y_test = [], []
    for i in range(len(test_gen)):
        x, y = test_gen[i]
        x_test.append(x)
        y_test.append(y)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    y_pred = predict_with_hybrid(models, x_test)

    ModelEvaluator._generate_classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        list(test_gen.class_indices.keys())
    )

    print("\n=== Hybrid Ensemble Evaluation ===")

    ModelEvaluator.generate_medical_report(
        model=type("HybridModel", (), {"predict": lambda _, x: predict_with_hybrid(models, x)})(),
        test_gen=test_gen
    )
