import json
import numpy as np
from model import CancerClassifier
from main import create_generators
from boosting import predict_with_ensemble
from evaluator import ModelEvaluator

# === Load saved ensemble info ===
with open("boost_models/ensemble_info.json", "r") as f:
    info = json.load(f)

weight_paths = info["weight_paths"]

# === Load test set ===
_, _, test_gen, _ = create_generators()
class_names = list(test_gen.class_indices.keys())

x_test, y_test = [], []
for i in range(len(test_gen)):
    x, y = test_gen[i]
    x_test.append(x)
    y_test.append(y)
x_test = np.vstack(x_test)
y_test = np.vstack(y_test)

# === Rebuild models and load weights ===
models = []
for path in weight_paths:
    model = CancerClassifier.build_model()
    model.load_weights(path)
    models.append(model)

# === Run ensemble prediction ===
ensemble_preds = predict_with_ensemble(models, x_test)

# === Final report ===
ModelEvaluator._generate_classification_report(
    np.argmax(y_test, axis=1),
    np.argmax(ensemble_preds, axis=1),
    class_names
)
