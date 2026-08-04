"""
train_model.py
----------------
Standalone training script for the CIFAR-10 CNN classifier.
This is a script version of CIFAR-10.ipynb - run this if you just want to
(re)generate the trained model file used by the Flask app (main.py) without
opening Jupyter.

Usage:
    python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "cifar10_cnn_model.h5")

CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))

    # first conv block - 32 filters
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # second conv block - 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # third conv block - 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # flatten + dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    print("Training data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    model = build_model()
    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                    min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=6,
                               restore_best_weights=True)

    model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[lr_reducer, early_stop]
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
