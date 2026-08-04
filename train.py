"""
Train MobileNetV2, EfficientNetB0, and ResNet50 on the Cats vs Dogs dataset
using transfer learning. Saves each model to the models/ directory.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"          # expects data/train and data/validation sub-dirs
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)


def build_model(base_model_fn, model_name: str) -> keras.Model:
    """Build a transfer-learning model with a custom classification head."""
    base = base_model_fn(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False  # freeze base weights

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save(base_model_fn, model_name: str) -> None:
    """Train a single model and save it."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"{'='*60}")

    model = build_model(base_model_fn, model_name)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{model_name}_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    save_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    model.save(save_path)
    print(f"Model saved to {save_path}")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"{model_name} → val_accuracy: {val_acc:.4f}")
    return history


if __name__ == "__main__":
    configs = [
        (MobileNetV2, "mobilenetv2"),
        (EfficientNetB0, "efficientnetb0"),
        (ResNet50, "resnet50"),
    ]

    for base_fn, name in configs:
        train_and_save(base_fn, name)

    print("\nAll models trained and saved to the 'models/' directory.")
