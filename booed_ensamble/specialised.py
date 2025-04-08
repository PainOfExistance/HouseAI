import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

IMAGE_SIZE = (224, 224)
WEIGHT_DECAY = 1e-4

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
