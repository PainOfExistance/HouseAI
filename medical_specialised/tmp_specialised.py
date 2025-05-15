import os
from glob import glob

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, Lambda,
                                     LayerNormalization, Multiply, Reshape,
                                     UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tqdm import tqdm


# =============================================
# 1. SUPERCHARGED PREPROCESSING
# =============================================
def medical_preprocess(image):
    """CT-specific preprocessing pipeline"""
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Gamma correction
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    
    return image

# =============================================
# 2. HYPERPARAMETERS & CONFIG
# =============================================
IMAGE_SIZE = (224, 224)  # Increased for better feature capture (256, 256)
BATCH_SIZE = 32
EPOCHS = 20 #30
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4
DATASET_DIR = "../Data2"
AUGMENTATION_FACTOR = 2  # For oversampling minority classes
BATCH_SIZE = 32
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2
SEED = 42

# =============================================
# 1. MEDICAL DATA GENERATOR CLASS
# =============================================
class MedicalDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, *args, **kwargs):
        batches = super().__call__(*args, **kwargs)
        
        # Apply medical preprocessing to every batch
        for batch_x, batch_y in batches:
            processed = np.array([medical_preprocess(img) for img in batch_x])
            yield (processed, batch_y)

# =============================================
# 2. DATA SPLITTING AND GENERATOR SETUP (UPDATED)
# =============================================
# Get all image paths and labels from class folders with proper extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
class_folders = sorted(glob(os.path.join(DATASET_DIR, "*")))
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
    labels.extend([class_names[class_idx]] * len(class_images))  # Use class names as labels

if not image_paths:
    raise ValueError("No images found in the dataset directory")

# Stratified splitting (train/val/test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels,
    test_size=VAL_SPLIT + TEST_SPLIT,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels,
    test_size=TEST_SPLIT/(VAL_SPLIT + TEST_SPLIT),
    random_state=SEED,
    stratify=temp_labels
)

# Create dataframes
train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})
test_df = pd.DataFrame({'filename': test_paths, 'class': test_labels})

# =============================================
# 3. MEDICAL DATA AUGMENTATION PIPELINE
# =============================================
train_datagen = MedicalDataGenerator(
    preprocessing_function=lambda x: x/255.0,  # Normalization
    rotation_range=15,       # Reduced for medical images
    width_shift_range=0.1,   # Conservative shifts
    height_shift_range=0.1,
    shear_range=0.01,        # Minimal shear
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='constant',    # Black background
    cval=0
)

valid_test_datagen = MedicalDataGenerator(
    preprocessing_function=lambda x: x/255.0
)

# =============================================
# 4. CREATE GENERATORS (UPDATED)
# =============================================
print("\n[STATUS] Creating medical data generators...")

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
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
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=class_names
)

test_generator = valid_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    classes=class_names
)

# Verify generators found images
if train_generator.samples == 0 or validation_generator.samples == 0 or test_generator.samples == 0:
    raise ValueError("No images found in one or more generators. Check your file paths and extensions.")

# =============================================
# 5. CLASS WEIGHT COMPUTATION (UPDATED)
# =============================================
# Get class counts directly from the generator
class_counts = train_generator.classes
unique_classes = np.unique(class_counts)
total_samples = len(class_counts)

class_weights = {}
for cls in unique_classes:
    class_weights[cls] = total_samples / (len(unique_classes) * np.sum(class_counts == cls))

print(f"\nClass Weights: {class_weights}")
# Reset generator after label extraction
train_generator.reset()

# =============================================
# 6. ADVANCED MODEL ARCHITECTURE
# =============================================

# -------------------------------
# Squeeze-and-Excitation block
# -------------------------------
def squeeze_excite_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='swish', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    return Multiply()([input_tensor, se])

# -------------------------------
# Build medical model
# -------------------------------
def build_medical_model():
    base_model = EfficientNetV2B0(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
    )

    # Strategic unfreezing of last 20 layers
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    x = base_model.output
    x = squeeze_excite_block(x)              # ✅ Attention
    x = LayerNormalization()(x)              # ✅ Stabilize post-attention
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = Dropout(0.4)(x)                       # ✅ Slightly lower dropout for stability
    x = Dense(128, activation='swish', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(3, activation='softmax', kernel_regularizer=l2(WEIGHT_DECAY))(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# Instantiate the model
model = build_medical_model()

# =============================================
# 5. OPTIMIZED TRAINING SETUP
# =============================================

optimizer = AdamW(
    learning_rate=INIT_LR,
    weight_decay=WEIGHT_DECAY
)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),  # Critical for medical
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    ),
    TensorBoard(log_dir='./logs', histogram_freq=1)
]

# =============================================
# 6. TRAINING WITH OVERSAMPLING
# =============================================
print("\n[STATUS] Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE * AUGMENTATION_FACTOR,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


# =============================================
# 7. COMPREHENSIVE EVALUATION
# =============================================
def generate_medical_report(model, test_gen):
    print("\n[STATUS] Generating medical evaluation report...")
    
    # Get true and predicted values
    y_true = np.array(test_gen.classes)
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_names = list(test_gen.class_indices.keys())
    
    # 1. Classification Report
    print("\n=== Detailed Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # 2. Confusion Matrix
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 3. ROC Curves (One-vs-Rest)
    plt.figure(figsize=(10,8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_probs[:, i])
        auc_score = roc_auc_score((y_true == i).astype(int), y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend()
    plt.savefig('roc_curves.png')
    plt.close()
    
    # 4. Save Misclassified Examples
    misclassified = np.where(y_pred != y_true)[0]
    os.makedirs('misclassified', exist_ok=True)
    
    for idx in tqdm(misclassified[:20], desc="Saving misclassified examples"):
        batch_idx = idx // BATCH_SIZE
        img_idx = idx % BATCH_SIZE
        
        batch = test_gen[batch_idx]
        img = batch[0][img_idx]
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        
        plt.imshow(img)
        plt.title(f"True: {true_class}\nPred: {pred_class}")
        plt.savefig(f'misclassified/{idx}_{true_class}_as_{pred_class}.png')
        plt.close()

# Load best model and evaluate
keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model('best_model.keras')
generate_medical_report(model, test_generator)

# =============================================
# 8. GRAD-CAM VISUALIZATION
# =============================================
def generate_gradcam(model, img_array, layer_name="top_conv"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    
    return heatmap.numpy()

# Example Grad-CAM generation
sample_batch = next(test_generator)
sample_img = sample_batch[0][0][np.newaxis, ...]
heatmap = generate_gradcam(model, sample_img)

plt.imshow(sample_img[0])
plt.imshow(cv2.resize(heatmap, IMAGE_SIZE), alpha=0.5, cmap='jet')
plt.savefig('gradcam_example.png')
plt.close()

print("\n[STATUS] All evaluations completed!")