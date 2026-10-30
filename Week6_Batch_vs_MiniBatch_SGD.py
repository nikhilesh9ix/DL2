# Week 6: Full-Batch vs Mini-Batch Training on Fashion-MNIST
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

def build_model():
    model = Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

for name, batch in [("Full-Batch", len(X_train)), ("Mini-Batch", 64)]:
    model = build_model()
    model.fit(X_train, y_train, epochs=3, batch_size=batch, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(name, "Test Accuracy:", round(acc * 100, 2), "%")
