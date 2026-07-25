# ============================================================
# Chapter 11: Time Series Prediction with LSTM
# Sections: 11-4-4
# ============================================================

import numpy as np
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ==============================
# Section 11-4-4: Time Series Prediction
# ==============================
print("\n--- 11-4-4: Time Series LSTM ---")

def generate_time_series(n_samples, n_timesteps, noise=0.1):
    t = np.linspace(0, 20, n_samples)
    series = np.sin(t) + noise * np.random.randn(n_samples)
    X, y = [], []
    for i in range(len(series) - n_timesteps):
        X.append(series[i:i+n_timesteps])
        y.append(series[i+n_timesteps])
    return np.array(X), np.array(y)

n_samples = 2000
n_timesteps = 50
X, y = generate_time_series(n_samples, n_timesteps, noise=0.1)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_timesteps, 1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("Training LSTM on time series...")
history_ts = model.fit(X_train, y_train, epochs=5, batch_size=32,
                       validation_split=0.2, verbose=1)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}, Test MAE: {test_mae:.4f}")

predictions = model.predict(X_test)

plt.figure(figsize=(12,5))
plt.plot(y_test[:100], label='Actual')
plt.plot(predictions[:100], label='Predicted')
plt.xlabel('Time Steps'); plt.ylabel('Value')
plt.title('Time Series Prediction with LSTM')
plt.legend(); plt.grid(True)
plt.show()

print("\n=== End of Chapter 11 Group A (Time Series) ===")