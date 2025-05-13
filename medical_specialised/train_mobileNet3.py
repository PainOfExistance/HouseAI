import os
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from main import create_generators

MODEL_DIR = "models"


def build_mobilenet(input_shape=(224, 224, 3), num_classes=4):
    base = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)


if __name__ == "__main__":
    train_gen, val_gen, _, _ = create_generators()

    model = build_mobilenet()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=10)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "mobilenetv3.keras"))