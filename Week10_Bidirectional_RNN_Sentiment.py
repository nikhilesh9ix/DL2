# Week 10: Bidirectional RNN for IMDB (short version)
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train, y_train = pad_sequences(X_train[:5000], maxlen=100), y_train[:5000]
X_test, y_test = pad_sequences(X_test[:1000], maxlen=100), y_test[:1000]

model = Sequential([
    layers.Embedding(10000, 32, mask_zero=True),
    layers.Bidirectional(layers.SimpleRNN(32)),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

label = ["Negative", "Positive"]
pred = model.predict(X_test[:10], verbose=0)
for i in range(10):
    print(f"Review {i+1}: Predicted={label[int(pred[i][0] >= 0.5)]}, Actual={label[y_test[i]]}")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Bidirectional RNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
