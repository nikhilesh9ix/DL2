# Week 8: CNN on MNIST with Pooling Demonstration
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
