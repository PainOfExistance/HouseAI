from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class CancerClassifier:
    @staticmethod
    def build_model(input_shape=(224, 224, 3), num_classes=3, weight_decay=1e-4):
        base_model = EfficientNetV2B0(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Unfreeze top layers
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