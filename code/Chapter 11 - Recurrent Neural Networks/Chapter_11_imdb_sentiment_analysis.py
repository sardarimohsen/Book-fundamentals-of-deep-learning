# ============================================================
# Chapter 11: IMDB Sentiment Analysis with RNN/LSTM/GRU
# Sections: 11-7-1, 11-7-2, 11-7-3
#           11-7-4, 11-7-5, 11-7-6
#           11-7-7, 11-7-8
# ============================================================

import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import SimpleRNN, LSTM, GRU, Embedding, Dense, Dropout
from tf.keras.datasets import imdb
from tf.keras.preprocessing import sequence
from tf.keras.callbacks import EarlyStopping
from tf.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ==============================
# Section 11-7-1: Load & Preprocess IMDB
# ==============================
print("\n--- 11-7-1: Load IMDB ---")
max_features = 10000
max_len = 500
batch_size = 128
embedding_dim = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# ==============================
# Section 11-7-2: SimpleRNN Model
# ==============================
print("\n--- 11-7-2: SimpleRNN Model ---")
def build_simple_rnn():
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        SimpleRNN(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model_rnn = build_simple_rnn()
model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_rnn.summary()

# ==============================
# Section 11-7-3: LSTM Model
# ==============================
print("\n--- 11-7-3: LSTM Model ---")
def build_lstm():
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model_lstm = build_lstm()
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_lstm.summary()

# ==============================
# Section 11-7-4: GRU Model
# ==============================
print("\n--- 11-7-4: GRU Model ---")
def build_gru():
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        GRU(64, return_sequences=False),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

model_gru = build_gru()
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_gru.summary()

# ==============================
# Section 11-7-5: Train with EarlyStopping
# ==============================
print("\n--- 11-7-5: Training Models ---")
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("\nTraining SimpleRNN...")
hist_rnn = model_rnn.fit(x_train, y_train, epochs=5, batch_size=batch_size,
                         validation_split=0.2, callbacks=[early_stop], verbose=1)

print("\nTraining LSTM...")
hist_lstm = model_lstm.fit(x_train, y_train, epochs=5, batch_size=batch_size,
                           validation_split=0.2, callbacks=[early_stop], verbose=1)

print("\nTraining GRU...")
hist_gru = model_gru.fit(x_train, y_train, epochs=5, batch_size=batch_size,
                         validation_split=0.2, callbacks=[early_stop], verbose=1)

# ==============================
# Section 11-7-6: Compare on Test Set
# ==============================
print("\n--- 11-7-6: Test Set Comparison ---")
test_loss_rnn, test_acc_rnn = model_rnn.evaluate(x_test, y_test, verbose=0)
test_loss_lstm, test_acc_lstm = model_lstm.evaluate(x_test, y_test, verbose=0)
test_loss_gru, test_acc_gru = model_gru.evaluate(x_test, y_test, verbose=0)

print("\n" + "="*50)
print("IMDB TEST SET COMPARISON")
print("="*50)
print(f"SimpleRNN - Loss: {test_loss_rnn:.4f}, Acc: {test_acc_rnn:.4f}")
print(f"LSTM      - Loss: {test_loss_lstm:.4f}, Acc: {test_acc_lstm:.4f}")
print(f"GRU       - Loss: {test_loss_gru:.4f}, Acc: {test_acc_gru:.4f}")

# ==============================
# Section 11-7-7: Plot Histories
# ==============================
print("\n--- 11-7-7: Training Plots ---")
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(history.history['accuracy'], label='Train Acc')
    ax1.plot(history.history['val_accuracy'], label='Val Acc')
    ax1.set_title(f'{title} - Accuracy'); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{title} - Loss'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()

plot_history(hist_rnn, 'SimpleRNN')
plot_history(hist_lstm, 'LSTM')
plot_history(hist_gru, 'GRU')

# ==============================
# Section 11-7-8: Advanced tips (Gradient Clipping & Recurrent Dropout)
# ==============================
print("\n--- 11-7-8: Advanced Tips ---")
optimizer_clip = Adam(learning_rate=0.001, clipnorm=1.0)
model_advanced = Sequential([
    Embedding(max_features, embedding_dim, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
model_advanced.compile(optimizer=optimizer_clip,
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
print("Advanced LSTM with gradient clipping and recurrent dropout created.")

print("\n=== End of Chapter 11 Group B (IMDB Sentiment) ===")