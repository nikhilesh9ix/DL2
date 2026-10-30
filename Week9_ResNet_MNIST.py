# Week 9: ResNet on MNIST using Residual Blocks
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train[:10000] / 255.0, X_test[:2000] / 255.0
y_train, y_test = y_train[:10000], y_test[:2000]
X_train, X_test = X_train[..., None], X_test[..., None]

def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)

inputs = layers.Input((28, 28, 1))
x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
for filters, stride in [(16, 1), (16, 1), (32, 2), (32, 1), (64, 2), (64, 1)]:
    x = residual_block(x, filters, stride)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

predictions = model.predict(X_test[:10], verbose=0).argmax(axis=1)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    plt.title(f"True:{y_test[i]} Pred:{predictions[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
