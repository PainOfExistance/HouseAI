# bc_model.py
import os

from sklearn.utils import compute_class_weight
from tensorflow.keras.models import load_model
from specialised import build_medical_model
import numpy as np
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
from config import INIT_LR, WEIGHT_DECAY
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def train_and_evaluate(generator, val_generator, epochs, model_path, y):
    model = build_medical_model()
    model.compile(
        optimizer=AdamW(learning_rate=INIT_LR, weight_decay=WEIGHT_DECAY),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=5, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(
            model_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        TensorBoard(log_dir='./logs', histogram_freq=1)
    ]

    model.fit(
        generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=compute_weights(y, num_classes=4),
        callbacks=callbacks,
        verbose=1
    )

    # Save the model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    y_true = val_generator.classes
    y_probs = model.predict(val_generator)
    y_pred = np.argmax(y_probs, axis=1)
    accuracy = np.mean(y_pred == y_true)

    # Identify misclassified
    errors = []
    filepaths = val_generator.filepaths
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append((filepaths[i], y_true[i]))

    return model_path, accuracy, errors

def compute_weights(labels, num_classes=4):
    labels = np.array(labels, dtype=int)  # 🔥 ensure labels are integers
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

