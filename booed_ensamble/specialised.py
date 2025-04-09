import os
import cv2
import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

IMAGE_SIZE = (224, 224)
WEIGHT_DECAY = 1e-4


def medical_preprocess(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

class MedicalDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        batches = super().__call__(*args, **kwargs)
        for batch_x, batch_y in batches:
            processed = np.array([medical_preprocess(img) for img in batch_x])
            yield (processed, batch_y)
def build_medical_model():
    base_model = EfficientNetV2B0(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet',
    )

    for layer in base_model.layers[-20:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish', kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(4, activation='softmax', kernel_regularizer=l2(WEIGHT_DECAY))(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
