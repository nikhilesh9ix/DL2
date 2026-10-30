# Week 10: Bidirectional RNN for IMDB Sentiment Analysis
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, SimpleRNN, Dense, Dropout

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

model = Sequential([
    Embedding(10000, 32, mask_zero=True),
    Bidirectional(SimpleRNN(32)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(acc * 100, 2), "%")

predictions = model.predict(X_test[:10], verbose=0)
for i in range(10):
    predicted = "Positive" if predictions[i][0] >= 0.5 else "Negative"
    actual = "Positive" if y_test[i] == 1 else "Negative"
    print(f"Review {i+1}: Predicted={predicted}, Actual={actual}")

plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Bidirectional RNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
