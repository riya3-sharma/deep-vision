#  handwritten digit recognition  using cnn model 

import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# load the dataset
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

# normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# reshape the data to fit the model
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# build the model
model = keras.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    # second layer
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    # flatten layer
    layers.Flatten(),

    # dense layers
    layers.Dense(64,activation='relu'),

    # output layer
    layers.Dense(10,activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(
                x_train,y_train,
                epochs=5,
                validation_data=(x_test,y_test)
        )

# evaluate the model
test_loss, test_acc = model.evaluate(x_test,y_test)
print(f'Test accuracy: {test_acc*100:.2f}%')

# confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # load image
# img_path = r"C:\Users\lenovo\Desktop\project\dataset\6.png"
# img = Image.open(img_path).convert('L')   # grayscale

# # resize to 28x28
# img = img.resize((28,28))

# # convert to numpy
# img = np.array(img)

# # invert colors (white digit on black bg)
# img = 255 - img  

# # normalize
# img = img / 255.0

# # reshape for model
# img = img.reshape(1,28,28,1)

# # prediction
# prediction = model.predict(img)

# # display image
# plt.imshow(img.reshape(28,28), cmap='gray')
# plt.axis('off')
# plt.show()

# print("Predicted Digit:", np.argmax(prediction))

# from PIL import Image, ImageFilter, ImageOps
# import numpy as np
# import cv2  # pip install opencv-python

# def preprocess_custom_image(img_path):
#     # 1. Load and convert to grayscale
#     img = Image.open(img_path).convert('L')
#     img = np.array(img)

#     # 2. Apply slight blur to reduce noise from pen strokes
#     img = cv2.GaussianBlur(img, (5, 5), 0)

#     # 3. Invert if image is dark-digit-on-white (typical handwriting)
#     #    MNIST expects WHITE digit on BLACK background
#     if np.mean(img) > 127:          # background is light → invert
#         img = 255 - img

#     # 4. Threshold to make it clean black/white
#     _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

#     # 5. Crop tightly around the digit bounding box
#     coords = cv2.findNonZero(img)
#     x, y, w, h = cv2.boundingRect(coords)
#     img = img[y:y+h, x:x+w]        # crop to digit only

#     # 6. Add padding around the digit (MNIST has ~4px padding)
#     pad = max(w, h) // 4
#     img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
#                              cv2.BORDER_CONSTANT, value=0)

#     # 7. Resize to 28x28
#     img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

#     # 8. Normalize
#     img = img / 255.0

#     # 9. Reshape for model
#     return img.reshape(1, 28, 28, 1)


# # --- Use it like this ---
# img = preprocess_custom_image(r"C:\Users\lenovo\Desktop\project\dataset\7.png")

# # Debug: visualize what the model actually sees
# plt.imshow(img.reshape(28, 28), cmap='gray')
# plt.title("What the model sees")
# plt.axis('off')
# plt.show()

# prediction = model.predict(img)
# print("Predicted Digit:", np.argmax(prediction))
# print("Confidence:", f"{np.max(prediction)*100:.1f}%")

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_custom_image(img_path, debug=True):
    # Step 1: Load
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.uint8)

    if debug:
        plt.figure(figsize=(14, 3))
        plt.subplot(1, 6, 1)
        plt.imshow(img, cmap='gray')
        plt.title("1. Original")
        plt.axis('off')

    # Step 2: Blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 3: Auto-invert (MNIST = white digit on BLACK bg)
    if np.mean(img) > 127:
        img = 255 - img

    if debug:
        plt.subplot(1, 6, 2)
        plt.imshow(img, cmap='gray')
        plt.title("2. Inverted")
        plt.axis('off')

    # Step 4: Adaptive threshold (handles uneven lighting better)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    if debug:
        plt.subplot(1, 6, 3)
        plt.imshow(img, cmap='gray')
        plt.title("3. Threshold")
        plt.axis('off')

    # Step 5: Morphological cleanup (fills holes, smooths edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel)

    if debug:
        plt.subplot(1, 6, 4)
        plt.imshow(img, cmap='gray')
        plt.title("4. Cleaned")
        plt.axis('off')

    # Step 6: Crop bounding box tightly
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit found! Image may be blank after processing.")
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # Step 7: Make square by padding shorter side (keeps aspect ratio)
    side = max(w, h)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = img

    # Step 8: Add 15% padding (MNIST digits don't touch edges)
    pad = side // 6
    square = cv2.copyMakeBorder(
        square, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=0
    )

    if debug:
        plt.subplot(1, 6, 5)
        plt.imshow(square, cmap='gray')
        plt.title("5. Centered")
        plt.axis('off')

    # Step 9: Resize to 28x28
    img_final = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 10: Normalize
    img_final = img_final / 255.0

    if debug:
        plt.subplot(1, 6, 6)
        plt.imshow(img_final, cmap='gray')
        plt.title("6. Final (28x28)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return img_final.reshape(1, 28, 28, 1)


def predict_with_confidence(model, img_path):
    img = preprocess_custom_image(img_path, debug=True)
    predictions = model.predict(img)[0]

    # Show ALL digit probabilities — not just the top 1
    plt.figure(figsize=(8, 3))
    colors = ['red' if i == np.argmax(predictions) else 'steelblue'
              for i in range(10)]
    plt.bar(range(10), predictions * 100, color=colors)
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Confidence (%)")
    plt.title(f"Predicted: {np.argmax(predictions)}  |  "
              f"Confidence: {np.max(predictions)*100:.1f}%")
    plt.tight_layout()
    plt.show()

    print("\nAll probabilities:")
    for digit, prob in enumerate(predictions):
        bar = "█" * int(prob * 40)
        print(f"  {digit}: {bar} {prob*100:.1f}%")

    return np.argmax(predictions)


# --- Run it ---
result = predict_with_confidence(model, r"C:\Users\lenovo\Desktop\project\dataset\7.png")
print(f"\nFinal prediction: {result}")