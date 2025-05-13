import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Basic config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = "../Data"

# Medical preprocessing
def medical_preprocess(image):
    import cv2
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

class MedicalDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, *args, **kwargs):
        batches = super().__call__(*args, **kwargs)
        for batch_x, batch_y in batches:
            processed = np.array([medical_preprocess(img) for img in batch_x])
            yield (processed, batch_y)

# Load test data
test_gen = MedicalDataGenerator(preprocessing_function=lambda x: x / 255.0).flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Load model
model = tf.keras.models.load_model("models/medical_specialised.keras")

# Evaluation logic
def generate_medical_report(model, test_gen):
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_names = list(test_gen.class_indices.keys())

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
        auc_score = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig("roc_curves.png")
    plt.close()

    os.makedirs("misclassified", exist_ok=True)
    misclassified = np.where(y_pred != y_true)[0]
    for idx in tqdm(misclassified[:20], desc="Saving misclassified"):
        batch_idx = idx // BATCH_SIZE
        img_idx = idx % BATCH_SIZE
        batch = test_gen[batch_idx]
        img = batch[0][img_idx]
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        plt.imshow(img)
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.savefig(f"misclassified/{idx}_{true_label}_as_{pred_label}.png")
        plt.close()

generate_medical_report(model, test_gen)
print("[STATUS] Evaluation complete.")