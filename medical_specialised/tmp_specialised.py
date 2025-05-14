import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetV2B0, ResNet50
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.layers import (Activation, Add, Attention,
                                     BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, Lambda,
                                     MaxPooling2D, Multiply, Reshape,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tqdm import tqdm


# =============================================
# 1. Enhanced Medical Preprocessing
# =============================================
def medical_preprocess(image):
    """Enhanced CT-specific preprocessing pipeline"""
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Adaptive Gamma correction
    mean_val = np.mean(image)
    gamma = 1.5 if mean_val < 100 else (0.8 if mean_val > 150 else 1.2)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    image = cv2.LUT(image, table)
    
    # Edge enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    image = cv2.addWeighted(image, 0.9, edges, 0.1, 0)
    
    return image

# =============================================
# 2. Configuration Parameters
# =============================================
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4
DATASET_DIR = "../Data"
AUGMENTATION_FACTOR = 3

# =============================================
# 3. Enhanced Data Pipeline
# =============================================
class MedicalDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, *args, **kwargs):
        batches = super().__call__(*args, **kwargs)
        for batch_x, batch_y in batches:
            processed = np.array([medical_preprocess(img) for img in batch_x])
            yield (processed, batch_y)

train_datagen = MedicalDataGenerator(
    preprocessing_function=lambda x: x/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.01,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='constant',
    cval=0,
    channel_shift_range=10,
)

valid_test_datagen = MedicalDataGenerator(
    preprocessing_function=lambda x: x/255.0
)

# Data generators
print("\n[STATUS] Loading datasets...")
train_generator = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    color_mode='rgb',
    interpolation='lanczos'
)

validation_generator = valid_test_datagen.flow_from_directory(
    f"{DATASET_DIR}/valid",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = valid_test_datagen.flow_from_directory(
    f"{DATASET_DIR}/test",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print(f"\nClass Weights: {class_weights}")

# =============================================
# 4. Hybrid CNN Model (EfficientNetV2 + ResNet50)
# =============================================



def channel_attention_module(input_tensor, ratio=8):
    channel = input_tensor.shape[-1]
    
    # Shared layers
    shared_layer_one = Dense(channel//ratio,
                            activation='relu',
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    
    shared_layer_two = Dense(channel,
                            kernel_initializer='he_normal',
                            use_bias=True,
                            bias_initializer='zeros')
    
    # Channel attention using Keras layers
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    channel_attention = Activation('sigmoid')(Add()([avg_pool, max_pool]))
    return Multiply()([input_tensor, channel_attention])

def spatial_attention_module(input_tensor):
    kernel_size = 7
    
    # Channel pooling using Keras layers
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_tensor)
    concat = concatenate([avg_pool, max_pool])
    
    # Convolution
    attention = Conv2D(1, kernel_size,
                      padding='same',
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=False)(concat)
    
    return Multiply()([input_tensor, attention])

def cbam_block(input_tensor):
    x = channel_attention_module(input_tensor)
    x = spatial_attention_module(x)
    return x

def build_hybrid_cnn_model():
    input_layer = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # --------------------------
    # EfficientNetV2 Branch
    # --------------------------
    effnet = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer
    )
    effnet._name = "effnet_branch"
    for layer in effnet.layers[-30:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    # Add CBAM attention to EfficientNet output
    effnet_output = effnet.output
    effnet_output = cbam_block(effnet_output)
    
    # --------------------------
    # ResNet50 Branch
    # --------------------------
    resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer
    )
    resnet._name = "resnet_branch"
    for layer in resnet.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    # Add CBAM attention to ResNet output
    resnet_output = resnet.output
    resnet_output = cbam_block(resnet_output)
    
    # --------------------------
    # Custom CNN Branch
    # --------------------------
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    custom_output = GlobalAveragePooling2D()(x)
    
    # --------------------------
    # Feature Fusion
    # --------------------------
    effnet_features = GlobalAveragePooling2D()(effnet_output)
    resnet_features = GlobalAveragePooling2D()(resnet_output)
    
    # Cross-model attention
    combined_features = concatenate([effnet_features, resnet_features, custom_output])
    
    # Feature recalibration
    attention = Dense(combined_features.shape[-1]//2, activation='relu')(combined_features)
    attention = Dense(combined_features.shape[-1], activation='sigmoid')(attention)
    recalibrated = Multiply()([combined_features, attention])
    
    # Enhanced classifier head
    x = Dense(512, activation='swish', 
              kernel_regularizer=l2(WEIGHT_DECAY))(recalibrated)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='swish', 
              kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(4, activation='softmax', 
                   kernel_regularizer=l2(WEIGHT_DECAY))(x)
    
    model = Model(inputs=input_layer, outputs=outputs)
    return model

model = build_hybrid_cnn_model()

# =============================================
# 5. Optimized Training Setup
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
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
    ]
)

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=12,
        mode='max',
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_hybrid_cnn_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    ),
    TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        profile_batch='10,15'
    )
]

# =============================================
# 6. Training Process
# =============================================
print("\n[STATUS] Training hybrid CNN model...")
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
# 7. Comprehensive Evaluation
# =============================================
def generate_medical_report(model, test_gen):
    print("\n[STATUS] Generating medical evaluation report...")
    
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_names = list(test_gen.class_indices.keys())
    
    # Classification Report
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    plt.figure(figsize=(10,8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # ROC Curves
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
    plt.savefig('roc_curves.png', bbox_inches='tight')
    plt.close()
    
    # Precision-Recall Curves
    plt.figure(figsize=(10,8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(
            (y_true == i).astype(int), y_pred_probs[:, i])
        ap_score = average_precision_score(
            (y_true == i).astype(int), y_pred_probs[:, i])
        plt.plot(recall, precision, label=f'{class_names[i]} (AP = {ap_score:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves by Class')
    plt.legend()
    plt.savefig('pr_curves.png', bbox_inches='tight')
    plt.close()
    
    # Misclassified Examples
    misclassified = np.where(y_pred != y_true)[0]
    os.makedirs('misclassified', exist_ok=True)
    
    for idx in tqdm(misclassified[:min(30, len(misclassified))], 
                   desc="Saving misclassified examples"):
        batch_idx = idx // BATCH_SIZE
        img_idx = idx % BATCH_SIZE
        
        batch = test_gen[batch_idx]
        img = batch[0][img_idx]
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        confidence = np.max(y_pred_probs[idx])
        
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}")
        plt.axis('off')
        plt.savefig(f'misclassified/{idx}_{true_class}_as_{pred_class}.png',
                   bbox_inches='tight')
        plt.close()

# Load best model and evaluate
model = tf.keras.models.load_model('best_hybrid_cnn_model.keras')
generate_medical_report(model, test_generator)

# =============================================
# 8. Grad-CAM Visualization
# =============================================
def generate_gradcam(model, img_array, layer_name="effnet_branch/top_conv"):
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

# Example visualization
sample_batch = next(test_generator)
sample_img = sample_batch[0][0][np.newaxis, ...]

try:
    heatmap = generate_gradcam(model, sample_img)
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(sample_img[0])
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(sample_img[0])
    plt.imshow(cv2.resize(heatmap, IMAGE_SIZE), alpha=0.5, cmap='jet')
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.savefig('gradcam_example.png', bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"\n[WARNING] Grad-CAM failed: {str(e)}")

print("\n[STATUS] All evaluations completed!")