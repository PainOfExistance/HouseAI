import os

from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

from main import create_generators

MODEL_DIR = "models"


def build_mobilenet(input_shape=(224, 224, 3), num_classes=3):
    base = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=Input(shape=input_shape))
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=out)


if __name__ == "__main__":
    train_gen, val_gen, _, _ = create_generators("../Data2")

    model = build_mobilenet()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "mobilenetv3_best.keras"),
        monitor="val_accuracy",  # or "val_loss"
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[checkpoint])

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "mobilenetv3.keras"))