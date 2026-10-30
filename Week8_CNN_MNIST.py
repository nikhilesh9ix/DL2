# Week 8: CNN on MNIST with Convolution and Pooling Demonstrations
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import mnist

feature = tf.constant([[[[1.], [2.], [3.], [4.]],
                        [[5.], [6.], [7.], [8.]],
                        [[9.], [10.], [11.], [12.]],
                        [[13.], [14.], [15.], [16.]]]])
print("Max Pooling:")
print(tf.nn.max_pool2d(feature, 2, 2, "VALID").numpy()[0, :, :, 0])
print("Average Pooling:")
print(tf.nn.avg_pool2d(feature, 2, 2, "VALID").numpy()[0, :, :, 0])

# Create a simple 7x7 grayscale image
image = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.float32)

image = image.reshape(1, 7, 7, 1)

# Define different convolution kernels
vertical_edge = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32).reshape(3, 3, 1, 1)

horizontal_edge = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32).reshape(3, 3, 1, 1)

sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32).reshape(3, 3, 1, 1)


# Perform convolution
vertical_output = tf.nn.conv2d(
    image, vertical_edge,
    strides=1,
    padding="SAME"
)

horizontal_output = tf.nn.conv2d(
    image, horizontal_edge,
    strides=1,
    padding="SAME"
)

sharpen_output = tf.nn.conv2d(
    image, sharpen,
    strides=1,
    padding="SAME"
)


# Display convolution results
plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.imshow(image[0, :, :, 0], cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(vertical_output[0, :, :, 0], cmap="gray")
plt.title("Vertical Edge")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(horizontal_output[0, :, :, 0], cmap="gray")
plt.title("Horizontal Edge")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(sharpen_output[0, :, :, 0], cmap="gray")
plt.title("Sharpening")
plt.axis("off")

plt.tight_layout()
plt.show()

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train[..., None], X_test[..., None]

model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.AveragePooling2D(2),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
