import os
import numpy as np
import tensorflow as tf
from evaluator import ModelEvaluator
from evaluator_main import medical_preprocess, MedicalDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# 📂 Nastavitve poti
MODEL_DIR = "medical_specialised/boost_models"
MODEL_PATHS = [
    os.path.join(MODEL_DIR, f"boost_model_{i}_weights.h5") for i in range(3)
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3
DATASET_DIR = "../Data"

def create_model():
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    input_shape = (224, 224, 3)
    num_classes = 3
    weight_decay = 1e-4

    base_model = EfficientNetV2B0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)

    return Model(inputs=base_model.input, outputs=outputs)


def predict_with_ensemble(models, x):
    preds = [model.predict(x, verbose=0) for model in models]
    return np.mean(preds, axis=0)

if __name__ == "__main__":
    test_gen = MedicalDataGenerator(preprocessing_function=lambda x: x / 255.0).flow_from_directory(
        f"{DATASET_DIR}/test",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    class_names = list(test_gen.class_indices.keys())

    models = []
    for path in MODEL_PATHS:
        model = create_model()
        model.load_weights(path)
        models.append(model)

    x_test, y_test = [], []
    for i in range(len(test_gen)):
        x, y = test_gen[i]
        x_test.append(x)
        y_test.append(y)
    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    ensemble_probs = predict_with_ensemble(models, x_test)

    class FakeModel:
        def predict(self, x):
            return predict_with_ensemble(models, x)

    print("\n=== Boosted Ensemble Evaluation ===")
    ModelEvaluator.generate_medical_report(FakeModel(), test_gen)

    print("\n✅ Končano. Shranjeni so: confusion_matrix.png, roc_curves.png in napačne slike.")
