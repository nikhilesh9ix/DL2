# Week 9: ResNet on MNIST (short version)
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:10000, ..., None] / 255.0, y_train[:10000]
X_test, y_test = X_test[:2000, ..., None] / 255.0, y_test[:2000]

def conv_bn(x, f, k=3, s=1):
    return layers.BatchNormalization()(layers.Conv2D(f, k, s, padding="same")(x))

def residual_block(x, f, s=1):
    y = layers.ReLU()(conv_bn(x, f, 3, s))
    y = conv_bn(y, f)
    if s != 1 or x.shape[-1] != f:
        x = conv_bn(x, f, 1, s)
    return layers.ReLU()(layers.Add()([y, x]))

inputs = layers.Input((28, 28, 1))
x = layers.ReLU()(conv_bn(inputs, 16))
for f, s in [(16, 1), (16, 1), (32, 2), (32, 1), (64, 2), (64, 1)]:
    x = residual_block(x, f, s)
x = layers.GlobalAveragePooling2D()(x)
model = models.Model(inputs, layers.Dense(10, activation="softmax")(x))

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

pred = model.predict(X_test[:10], verbose=0).argmax(axis=1)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap="gray")
    plt.title(f"True:{y_test[i]} Pred:{pred[i]}")
    plt.axis("off")
plt.show()
