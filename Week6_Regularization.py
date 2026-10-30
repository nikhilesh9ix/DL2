# Week 6: Compare NONE, L1 and L2 Regularization
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers, regularizers

X, y = make_moons(n_samples=1500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

for name, reg in {"NONE": None, "L1": regularizers.L1(0.001),
                  "L2": regularizers.L2(0.001)}.items():
    model = Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(name, "Test Accuracy:", round(acc * 100, 2), "%")
