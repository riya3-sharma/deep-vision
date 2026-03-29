# HANDWRITTEN DIGIT RECOGNITION SYSTEM WITH CNN MODEL

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

MODEL_PATH = "model.h5"

# LOAD MNIST
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, y_train, x_test, y_test

# BUILD MODEL
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# TRAIN OR LOAD
def get_model():
    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        return keras.models.load_model(MODEL_PATH)

    print("Training model...")

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

    model.save(MODEL_PATH)
    return model

# PERFECT PREPROCESS
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not found")

    # Resize
    img = cv2.resize(img, (28, 28))

    # Auto detect background and invert if needed
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Threshold (clean)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Centering
    coords = np.column_stack(np.where(img > 0))
    if coords.size != 0:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
        digit = cv2.resize(digit, (20, 20))

        new_img = np.zeros((28, 28))
        new_img[4:24, 4:24] = digit
        img = new_img

    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)

    return img

# PREDICT
def predict(model, path):
    img = preprocess_image(path)
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    digit = np.argmax(pred)

    print("Prediction:", digit)

    plt.imshow(img[0].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {digit}")
    plt.axis('off')
    plt.show()

# MAIN

if __name__ == "__main__":
    model = get_model()

    IMAGE_PATH =r"C:\Users\lenovo\Desktop\project\dataset\0.6.png"
    predict(model, IMAGE_PATH)