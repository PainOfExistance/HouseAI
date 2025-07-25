# boosting_ensemble.py

import numpy as np
import os
import json
from model import CancerClassifier
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping


def extract_from_generator(generator):
    x_all, y_all = [], []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        x_all.append(x_batch)
        y_all.append(y_batch)
    return np.vstack(x_all), np.vstack(y_all)


def compute_sample_weights(model, x_train, y_true):
    """Compute updated sample weights based on misclassifications."""
    y_pred = np.argmax(model.predict(x_train, verbose=0), axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    sample_weights = np.ones(len(y_true))
    misclassified = y_pred != y_true_labels
    sample_weights[misclassified] *= 2.0
    return sample_weights / np.sum(sample_weights)


def train_boosting_ensemble(
    train_gen,
    val_gen,
    class_weights=None,
    class_names=None,
    num_rounds=3,
    save_dir="boost_models"
):
    os.makedirs(save_dir, exist_ok=True)
    model_weight_paths = []

    print("\n[INFO] Extracting data from generators...")
    x_train, y_train = extract_from_generator(train_gen)
    x_val, y_val = extract_from_generator(val_gen)

    # 🧮 Analiza porazdelitve razredov
    y_train_labels = np.argmax(y_train, axis=1)
    class_counts = Counter(y_train_labels)
    print(f"\n[INFO] Razredna porazdelitev (train): {dict(class_counts)}")

    if class_weights is None and class_names is not None:
        print("[INFO] Samodejno računam class_weights...")
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        class_weights = {i: w for i, w in enumerate(weights)}
        print(f"[INFO] Izračunani class_weights: {class_weights}")

    sample_weights = np.ones(len(y_train)) / len(y_train)

    for round_idx in range(num_rounds):
        print(f"\n[BOOSTING] Training model {round_idx + 1}/{num_rounds}")
        model = CancerClassifier.build_model()
        trainer = ModelTrainer(model, None, None, class_weights)
        trainer.compile_model(lr=1e-4)

        model.predict(np.zeros((1, 224, 224, 3)))

        early_stopping = EarlyStopping(
            monitor="val_loss",  # ali "val_accuracy"
            patience=3,
            restore_best_weights=True
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
            validation_data=(x_val, y_val),
            epochs=15 if round_idx == 0 else 5,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        weight_path = os.path.join(save_dir, f"boost_model_{round_idx}_weights.h5")
        model.save_weights(weight_path)
        model_weight_paths.append(weight_path)

        sample_weights = compute_sample_weights(model, x_train, y_train)

    # Save ensemble metadata
    ensemble_package = {
        "weight_paths": model_weight_paths,
        "note": "Boosted EfficientNet ensemble (weights only)",
        "num_rounds": len(model_weight_paths)
    }
    with open(os.path.join(save_dir, "ensemble_info.json"), "w") as f:
        json.dump(ensemble_package, f)

    print("\n[INFO] Evaluating final ensemble on validation set...")
    models = []
    for path in model_weight_paths:
        model = CancerClassifier.build_model()
        model.load_weights(path)
        model.predict(np.zeros((1, 224, 224, 3)))
        models.append(model)

    final_preds = predict_with_ensemble(models, x_val)
    ModelEvaluator._generate_classification_report(
        np.argmax(y_val, axis=1),
        np.argmax(final_preds, axis=1),
        class_names
    )

    return model_weight_paths


def predict_with_ensemble(models, x):
    preds = [model.predict(x, verbose=0) for model in models]
    avg_pred = np.mean(preds, axis=0)
    return avg_pred
