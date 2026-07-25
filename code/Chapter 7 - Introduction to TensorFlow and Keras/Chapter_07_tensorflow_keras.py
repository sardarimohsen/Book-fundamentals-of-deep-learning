# ============================================================
# Chapter 7: Introduction to TensorFlow and Keras
# Sections: 7-4-1, 7-4-2, 7-4-3
#           7-4-5, 7-5-1 to 7-5-5
#           7-6-2 to 7-6-9, 7-7-1
# ============================================================

import tensorflow as tf
import numpy as np
from tf.keras import Sequential, Model
from tf.keras.layers import Dense, Input, Dropout, Flatten, BatchNormalization
from tf.keras.layers import Conv2D, MaxPooling2D, Embedding, LSTM
from tf.keras.optimizers import Adam
from tf.keras.models import load_model

# ==============================
# Section 7-4-1: Scalar as Tensor
# ==============================
print("\n--- 7-4-1: Scalar Tensor ---")
x_scalar = tf.constant(10)
print("Scalar Tensor:", x_scalar)

# ==============================
# Section 7-4-2: Vector as Tensor
# ==============================
print("\n--- 7-4-2: Vector Tensor ---")
x_vector = tf.constant([10, 20, 30, 40])
print("Vector:", x_vector)
print("Vector shape:", x_vector.shape)

# ==============================
# Section 7-4-3: Matrix as Tensor
# ==============================
print("\n--- 7-4-3: Matrix Tensor ---")
x_matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
print("Matrix:\n", x_matrix)
print("Matrix shape:", x_matrix.shape)

# ==============================
# Section 7-4-5: Shape vs Rank
# ==============================
print("\n--- 7-4-5: Shape vs Rank ---")
x_rank = tf.constant([[1, 2], [3, 4]])
print("Tensor:\n", x_rank)
print("Rank:", tf.rank(x_rank).numpy())
print("Shape:", x_rank.shape)

# ==============================
# Section 7-5-1: Creating Tensors (zeros, ones)
# ==============================
print("\n--- 7-5-1: Creating Tensors ---")
zeros = tf.zeros((3, 4))
ones = tf.ones((2, 5))
print("Zeros shape:", zeros.shape)
print("Ones shape:", ones.shape)

# ==============================
# Section 7-5-2: Random Tensors
# ==============================
print("\n--- 7-5-2: Random Tensors ---")
rand_normal = tf.random.normal((3, 3))
rand_uniform = tf.random.uniform((3, 3))
print("Random normal sample (first row):", rand_normal[0])
print("Random uniform sample (first row):", rand_uniform[0])

# ==============================
# Section 7-5-3: Arithmetic Operations
# ==============================
print("\n--- 7-5-3: Arithmetic Ops ---")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# ==============================
# Section 7-5-4: Matrix Multiplication
# ==============================
print("\n--- 7-5-4: Matrix Multiplication ---")
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print("A * B =\n", C)

# ==============================
# Section 7-5-5: Reshaping Tensors
# ==============================
print("\n--- 7-5-5: Reshaping ---")
x_reshape = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
reshaped = tf.reshape(x_reshape, (8,))
expanded = tf.expand_dims(x_reshape, axis=0)
squeezed = tf.squeeze(expanded)
print("Original shape:", x_reshape.shape)
print("Reshaped to (8,):", reshaped.shape)
print("Expanded to (1,2,4):", expanded.shape)
print("Squeezed back:", squeezed.shape)

# ==============================
# Section 7-6-2: Sequential API
# ==============================
print("\n--- 7-6-2: Sequential API Model ---")
model_seq = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model_seq.summary()

# ==============================
# Section 7-6-3: Functional API
# ==============================
print("\n--- 7-6-3: Functional API Model ---")
inputs = Input(shape=(100,))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model_func = Model(inputs=inputs, outputs=outputs)
model_func.summary()

# ==============================
# Section 7-6-4: Common Layers (instantiation examples)
# ==============================
print("\n--- 7-6-4: Common Layers ---")
dense_layer = Dense(128, activation='relu')
dropout_layer = Dropout(0.5)
flatten_layer = Flatten()
bn_layer = BatchNormalization()
conv_layer = Conv2D(32, (3,3), activation='relu', padding='same')
pool_layer = MaxPooling2D((2,2))
embed_layer = Embedding(input_dim=10000, output_dim=128)
lstm_layer = LSTM(64)
print("All layer objects created successfully.")

# ==============================
# Section 7-6-5: Compile
# ==============================
print("\n--- 7-6-5: Compile ---")
model_seq.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model compiled.")

# ==============================
# Section 7-6-6: Fit (with dummy data)
# ==============================
print("\n--- 7-6-6: Fit (dummy data) ---")
X_dummy = np.random.rand(1000, 100)
y_dummy = np.random.randint(0, 10, (1000, 10))
history = model_seq.fit(X_dummy, y_dummy, epochs=2, batch_size=32, verbose=0)
print("Training history keys:", list(history.history.keys()))

# ==============================
# Section 7-6-7: Predict
# ==============================
print("\n--- 7-6-7: Predict ---")
y_pred = model_seq.predict(X_dummy[:5], verbose=0)
print("Predictions shape:", y_pred.shape)

# ==============================
# Section 7-6-8: Evaluate
# ==============================
print("\n--- 7-6-8: Evaluate ---")
loss, acc = model_seq.evaluate(X_dummy[:50], y_dummy[:50], verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# ==============================
# Section 7-6-9: Save & Load
# ==============================
print("\n--- 7-6-9: Save & Load ---")
model_seq.save('model_demo.keras')
loaded = load_model('model_demo.keras')
print("Model saved and loaded successfully.")

# ==============================
# Section 7-7-1: GPU Detection
# ==============================
print("\n--- 7-7-1: GPU Detection ---")
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
print("\n=== End of Chapter 7 script ===")