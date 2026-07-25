# ============================================================
# Chapter 10: CNN from scratch on CIFAR-10
# Sections: 10-5-1, 10-5-2, 10-5-3, 10-5-4
#           10-5-5, 10-5-6, 10-5-7
# ============================================================

import tensorflow as tf
from tf.keras import layers, models
from tf.keras.datasets import cifar10
from tf.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ==============================
# Section 10-5-1: Load & Preprocess CIFAR-10
# ==============================
print("\n--- 10-5-1: Load CIFAR-10 ---")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# ==============================
# Section 10-5-2: Base CNN Model (Sequential)
# ==============================
print("\n--- 10-5-2: Base CNN Model ---")
model_base = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model_base.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
model_base.summary()

# ==============================
# Section 10-5-3: Train Base Model
# ==============================
print("\n--- 10-5-3: Train Base Model ---")
history_base = model_base.fit(x_train, y_train,
                              epochs=20, batch_size=64,
                              validation_data=(x_test, y_test),
                              verbose=1)

# ==============================
# Section 10-5-4: Residual Block Model (Functional API)
# ==============================
print("\n--- 10-5-4: Residual Model ---")
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

inputs = layers.Input(shape=(32,32,3))
x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = residual_block(x, 32)
x = layers.MaxPooling2D((2,2))(x)
x = residual_block(x, 64)
x = layers.MaxPooling2D((2,2))(x)
x = residual_block(x, 128)
x = layers.MaxPooling2D((2,2))(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model_res = models.Model(inputs=inputs, outputs=outputs)
model_res.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
model_res.summary()

history_res = model_res.fit(x_train, y_train,
                            epochs=5, batch_size=64,
                            validation_data=(x_test, y_test),
                            verbose=1)

# ==============================
# Section 10-5-5: Evaluate & Compare
# ==============================
print("\n--- 10-5-5: Evaluation ---")
test_loss_base, test_acc_base = model_base.evaluate(x_test, y_test, verbose=0)
test_loss_res, test_acc_res = model_res.evaluate(x_test, y_test, verbose=0)
print(f"Base Model - Test Acc: {test_acc_base:.4f}")
print(f"Residual Model - Test Acc: {test_acc_res:.4f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history_base.history['accuracy'], label='Base Train Acc')
plt.plot(history_base.history['val_accuracy'], label='Base Val Acc')
plt.plot(history_res.history['accuracy'], label='Res Train Acc')
plt.plot(history_res.history['val_accuracy'], label='Res Val Acc')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history_base.history['loss'], label='Base Train Loss')
plt.plot(history_base.history['val_loss'], label='Base Val Loss')
plt.plot(history_res.history['loss'], label='Res Train Loss')
plt.plot(history_res.history['val_loss'], label='Res Val Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# Section 10-5-6: Data Augmentation
# ==============================
print("\n--- 10-5-6: Model with Augmentation ---")
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

model_aug = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model_aug.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history_aug = model_aug.fit(x_train, y_train,
                            epochs=5, batch_size=64,
                            validation_data=(x_test, y_test),
                            verbose=1)

# ==============================
# Section 10-5-7: Save & Load
# ==============================
print("\n--- 10-5-7: Save/Load ---")
model_base.save('cnn_cifar10.keras')
loaded = tf.keras.models.load_model('cnn_cifar10.keras')
print("Model saved and loaded successfully.")

print("\n=== End of Chapter 10 Group A (CIFAR-10) ===")
