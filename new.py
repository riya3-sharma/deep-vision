import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# ─── load & preprocess dataset ───────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

MODEL_PATH = "digit_model.keras"

# ─── build model ─────────────────────────────────────────────────────────────
def build_model():
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
    return model

# ─── load model if exists, else train fresh ──────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from '{MODEL_PATH}' ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully! Skipping training.\n")
else:
    print("No saved model found. Training from scratch...\n")
    model = build_model()

    history = model.fit(
        x_train, y_train,
        epochs=50,
        validation_data=(x_test, y_test)
    )

    # ── save model after training ─────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")

    # ── clean training graph ──────────────────────────────────────────────────
    # epochs_range = range(1, len(history.history['accuracy']) + 1)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # fig.suptitle('Training History', fontsize=14)

    # # Accuracy plot
    # axes[0].plot(epochs_range, history.history['accuracy'],
    #              color='steelblue', linewidth=1.5, label='Train Accuracy')
    # axes[0].plot(epochs_range, history.history['val_accuracy'],
    #              color='tomato', linewidth=1.5, linestyle='--', label='Val Accuracy')
    # axes[0].set_title('Accuracy')
    # axes[0].set_xlabel('Epoch')
    # axes[0].set_ylabel('Accuracy')
    # axes[0].legend(frameon=False)
    # axes[0].spines['top'].set_visible(False)
    # axes[0].spines['right'].set_visible(False)

    # # Loss plot
    # axes[1].plot(epochs_range, history.history['loss'],
    #              color='steelblue', linewidth=1.5, label='Train Loss')
    # axes[1].plot(epochs_range, history.history['val_loss'],
    #              color='tomato', linewidth=1.5, linestyle='--', label='Val Loss')
    # axes[1].set_title('Loss')
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('Loss')
    # axes[1].legend(frameon=False)
    # axes[1].spines['top'].set_visible(False)
    # axes[1].spines['right'].set_visible(False)

    # plt.tight_layout()
    # plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    # plt.show()

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# ─── feedback memory ──────────────────────────────────────────────────────────
feedback_images = []
feedback_labels = []

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
    return img.reshape(1, 28, 28, 1)

# ─── retrain on feedback ──────────────────────────────────────────────────────
def retrain_on_feedback():
    if not feedback_images:
        print("No feedback yet.")
        return
    fb_x = np.repeat(np.array(feedback_images).reshape(-1, 28, 28, 1), 10, axis=0)
    fb_y = np.repeat(np.array(feedback_labels), 10, axis=0)
    print(f"\nRetraining on {len(feedback_images)} correction(s)...")
    model.fit(fb_x, fb_y, epochs=5, verbose=1)

    # ── save updated model after every correction ─────────────────────────────
    model.save(MODEL_PATH)
    print(f"Updated model saved to '{MODEL_PATH}'\n")

# ─── predict + feedback ───────────────────────────────────────────────────────
def predict_with_feedback(image_path):
    img_array  = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    digit      = np.argmax(prediction)
    confidence = prediction[0][digit] * 100

    # ── clean prediction plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_array.reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Predicted: {digit}  ({confidence:.1f}%)', fontsize=12)
    axes[0].axis('off')

    # plain bar chart — no bold, no gridlines, clean spines
    bar_colors = ['tomato' if i == digit else '#aec6e8' for i in range(10)]
    axes[1].bar(range(10), prediction[0] * 100, color=bar_colors, width=0.6)
    axes[1].set_xticks(range(10))
    axes[1].set_xticklabels([str(i) for i in range(10)])
    axes[1].set_xlabel('Digit', fontsize=10)
    axes[1].set_ylabel('Confidence (%)', fontsize=10)
    axes[1].set_title('Prediction Confidence', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()

    print(f"\nPredicted : {digit}  (Confidence: {confidence:.1f}%)")

    # ── feedback ──────────────────────────────────────────────────────────────
    feedback = input("\nWas this correct? (y / n): ").strip().lower()

    if feedback == 'y':
        print("Confirmed. No changes needed.\n")

    elif feedback == 'n':
        while True:
            try:
                correct = int(input("Enter correct digit (0-9): ").strip())
                if 0 <= correct <= 9:
                    break
                print("Enter a number between 0 and 9.")
            except ValueError:
                print("Invalid. Enter a digit 0-9.")

        feedback_images.append(img_array.reshape(28, 28, 1))
        feedback_labels.append(correct)
        print(f"Correction saved. Total corrections: {len(feedback_labels)}")

        retrain_on_feedback()

        new_pred  = model.predict(img_array, verbose=0)
        new_digit = np.argmax(new_pred)
        new_conf  = new_pred[0][new_digit] * 100
        print(f"After learning → Predicted: {new_digit}  ({new_conf:.1f}%)")
        print("Correct!" if new_digit == correct else "Still learning — more examples will help.")
    else:
        print("Skipped.\n")


# ─── usage ────────────────────────────────────────────────────────────────────
predict_with_feedback(r"C:\Users\lenovo\Desktop\project\dataset\7.0.jpeg")