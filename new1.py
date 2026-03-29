# HANDWRITTEN DIGIT RECOGNITION SYSYTEM WITH CNN MODEL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import cv2
import os

# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# normalize 
x_train = x_train / 255.0
x_test  = x_test  / 255.0

# reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

MODEL_PATH = "digit_model.keras"

# build model 
def build_model():
    model = keras.Sequential([

        # convolutional layer
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),

        # second convolutional layer
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        # flatten layer
        layers.Flatten(),

        # dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # output layer
        layers.Dense(10, activation='softmax')
    ])

    # compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#load or train
if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from '{MODEL_PATH}' ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded! Skipping training.\n")
else:
    print("Training from scratch...\n")
    model = build_model()

    # EarlyStopping
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=3,           # stop if no improvement for 3 epochs
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        x_train, y_train,
        epochs=10,            # ✅ fixed: was 50, now 10
        validation_data=(x_test, y_test),
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc*100:.2f}%\n')

#feedback memory
feedback_images = []
feedback_labels = []

#preprocessing 
def preprocess_image(image_path):
    # ✅ check file exists first
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # debug:
    print(f"Image loaded: {img.shape}, pixel range: {img.min()}–{img.max()}")

    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #check if inversion made it mostly white (digit) or mostly black (background)
    white_ratio = np.sum(img == 255) / img.size
    print(f"White pixel ratio after threshold: {white_ratio:.2f}  (should be ~0.1–0.3 for a digit)")

    #if white ratio too high, image was already inverted correctly — flip back
    if white_ratio > 0.5:
        img = cv2.bitwise_not(img)
        print("Re-inverted image (background was white)")

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    else:
        print("WARNING: No digit found in image after thresholding!")

    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    print(f"Final preprocessed pixel range: {img.min():.2f}–{img.max():.2f}")

    return img.reshape(1, 28, 28, 1)

# retrain on feedback
def retrain_on_feedback():
    if not feedback_images:
        print("No feedback yet.")
        return

    fb_x = np.array(feedback_images).reshape(-1, 28, 28, 1)
    fb_y = np.array(feedback_labels)

    # mix feedback with some original MNIST samples of the correct digit
    # so the model doesn't forget everything else
    idx = np.where(y_train == fb_y[-1])[0][:200]
    mix_x = np.concatenate([fb_x, x_train[idx]])
    mix_y = np.concatenate([fb_y, y_train[idx]])

    # repeat only the feedback part more
    fb_x_rep = np.repeat(fb_x, 15, axis=0)
    fb_y_rep = np.repeat(fb_y, 15, axis=0)
    final_x  = np.concatenate([fb_x_rep, x_train[idx]])
    final_y  = np.concatenate([fb_y_rep, y_train[idx]])

    print(f"\nRetraining on {len(feedback_images)} correction(s) + {len(idx)} MNIST samples...")
    model.fit(final_x, final_y, epochs=5, verbose=1)
    model.save(MODEL_PATH)
    print(f"Updated model saved.\n")

#predict + feedback
def predict_with_feedback(image_path):
    img_array  = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    digit      = np.argmax(prediction)
    confidence = prediction[0][digit] * 100

    #no graphs — terminal output only
    print(f"\n{'='*35}")
    print(f"  Predicted Digit : {digit}")
    print(f"  Confidence      : {confidence:.1f}%")
    print(f"{'='*35}")


    #feedback 
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
        print("Correct!" if new_digit == correct else "Still learning — give more examples.")
  
    else:
        print("Skipped.\n")


# usage
predict_with_feedback(r"C:\Users\lenovo\Desktop\project\dataset\2.21.jpeg")