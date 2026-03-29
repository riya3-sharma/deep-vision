# HANDWRITTEN DIGIT RECOGNITION SYSTEM WITH CNN MODEL
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os

# ─── load & preprocess dataset ───────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# ─── build model ─────────────────────────────────────────────────────────────
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# ─── feedback memory (stores all corrections) ─────────────────────────────────
feedback_images = []   # stores preprocessed images that were wrong
feedback_labels = []   # stores the correct labels given by user


# ─── preprocessing ────────────────────────────────────────────────────────────
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


# ─── retrain on all collected feedback ───────────────────────────────────────
def retrain_on_feedback():
    if len(feedback_images) == 0:
        print("No feedback collected yet.")
        return

    fb_x = np.array(feedback_images).reshape(-1, 28, 28, 1)
    fb_y = np.array(feedback_labels)

    print(f"\n Retraining on {len(fb_x)} feedback sample(s)...")

    # repeat feedback samples 10x so model takes them seriously
    fb_x_repeated = np.repeat(fb_x, 10, axis=0)
    fb_y_repeated = np.repeat(fb_y, 10, axis=0)

    model.fit(fb_x_repeated, fb_y_repeated,
              epochs=5,
              verbose=1)

    print("Model updated with your feedback!\n")


# ─── main predict + feedback loop ────────────────────────────────────────────
def predict_with_feedback(image_path):
    img_array = preprocess_image(image_path)

    prediction = model.predict(img_array, verbose=0)
    digit      = np.argmax(prediction)
    confidence = prediction[0][digit] * 100

    # ── show result ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_array.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Predicted: {digit}  ({confidence:.1f}%)', fontsize=13)
    axes[0].axis('off')

    colors = ['red' if i == digit else 'steelblue' for i in range(10)]
    axes[1].bar(range(10), prediction[0] * 100, color=colors)
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel('Digit')
    axes[1].set_ylabel('Confidence (%)')
    axes[1].set_title('Confidence per Digit')

    plt.tight_layout()
    plt.show()

    print(f"\n Predicted Digit : {digit}  (Confidence: {confidence:.1f}%)")

    # ── ask user for feedback ─────────────────────────────────────────────────
    feedback = input("\nWas this prediction correct? (y / n): ").strip().lower()

    if feedback == 'y':
        print("Great! Prediction confirmed. No retraining needed.\n")

    elif feedback == 'n':
        while True:
            try:
                correct_label = int(input("Enter the correct digit (0-9): ").strip())
                if 0 <= correct_label <= 9:
                    break
                else:
                    print("Please enter a number between 0 and 9.")
            except ValueError:
                print("Invalid input. Please enter a digit 0-9.")

        # save this correction to feedback memory
        feedback_images.append(img_array.reshape(28, 28, 1))
        feedback_labels.append(correct_label)

        print(f"\n Feedback saved! Correct label '{correct_label}' recorded.")
        print(f" Total corrections so far: {len(feedback_labels)}")

        # retrain immediately on this correction + all past corrections
        retrain_on_feedback()

        # verify — predict same image again after retraining
        new_pred    = model.predict(img_array, verbose=0)
        new_digit   = np.argmax(new_pred)
        new_conf    = new_pred[0][new_digit] * 100
        print(f"\n Re-prediction after learning: {new_digit}  ({new_conf:.1f}%)")

        if new_digit == correct_label:
            print("Model now predicts it correctly!")
        else:
            print("Model is still learning — more feedback on similar images will help.")

    else:
        print("Invalid input. Skipping feedback.")

    return digit


# ─── usage ──────────────
predict_with_feedback("C:\\Users\\lenovo\\Desktop\\project\\dataset\\4.jpeg")