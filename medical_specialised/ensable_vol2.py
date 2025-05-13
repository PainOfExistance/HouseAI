import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from main import create_generators
from evaluator import ModelEvaluator

MODEL_DIR = "models"


def load_all_models():
    return [
        load_model(os.path.join(MODEL_DIR, "medical_specialised.keras")),  # EfficientNet
        load_model(os.path.join(MODEL_DIR, "mobilenetv3.keras")),
        load_model(os.path.join(MODEL_DIR, "resnet50.keras")),
    ]


def predict_with_hybrid(models, x):
    preds = [model.predict(x, verbose=0) for model in models]
    return np.mean(preds, axis=0)


if __name__ == "__main__":
    _, _, test_gen, _ = create_generators()

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
