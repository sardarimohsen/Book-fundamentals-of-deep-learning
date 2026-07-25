# ============================================================
# Chapter 8: Model Evaluation Metrics
# Sections: 8-3-1, 8-4-1, 8-7
#           8-11, 8-12-3 to 8-12-7
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tf.keras.datasets import mnist
from sklearn.metrics import (confusion_matrix, recall_score, f1_score,
                             roc_auc_score, top_k_accuracy_score,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

# Set plot style
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# ==============================
# Section 8-11: Load MNIST, build model, compute metrics
# (Also covers 8-3-1 and 8-4-1 as metrics in compile)
# ==============================
print("\n--- 8-11: MNIST Model & Metrics ---")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # This covers 8-3-1
model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute metrics
conf_matrix = confusion_matrix(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
top_3_acc = top_k_accuracy_score(y_test, y_pred_probs, k=3)

print("Confusion Matrix:\n", conf_matrix)
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1-Score: {f1:.4f}")
print(f"ROC AUC (OvR): {roc_auc:.4f}")
print(f"Top-3 Accuracy: {top_3_acc:.4f}")

# ==============================
# Section 8-7: R² Score (simple example)
# ==============================
print("\n--- 8-7: R² Score ---")
from sklearn.metrics import r2_score
y_true_r2 = np.array([10, 20, 30, 40, 50])
y_pred_r2 = np.array([12, 18, 33, 38, 52])
r2 = r2_score(y_true_r2, y_pred_r2)
print(f"R² Score: {r2:.4f}")


# ==============================
# Section 8-12-3: Loss & Accuracy curves
# ==============================
print("\n--- 8-12-3: Training Curves ---")
# Re-train with validation to get history
model_v2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model_v2.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
history = model_v2.fit(x_train, y_train, epochs=3, batch_size=32,
                       validation_data=(x_test, y_test), verbose=0)

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_loss)+1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, train_loss, 'b-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.title('Loss Curves'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs_range, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy')
plt.title('Accuracy Curves'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# Section 8-12-4: Confusion Matrix Heatmap
# ==============================
print("\n--- 8-12-4: Confusion Matrix Heatmap ---")
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Class'); plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

# ==============================
# Section 8-12-5: ROC Curve (multiclass)
# ==============================
print("\n--- 8-12-5: ROC Curve ---")
y_test_bin = label_binarize(y_test, classes=range(10))
fpr, tpr, roc_auc_dict = {}, {}, {}
n_classes = 10
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,8))
colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
for i, color in enumerate(colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc_dict[i]:.2f})')
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)'); plt.legend(loc="lower right"); plt.grid(True)
plt.show()

# ==============================
# Section 8-12-6: Precision-Recall Curve
# ==============================
print("\n--- 8-12-6: Precision-Recall Curve ---")
# Binary example: class 0 vs rest
y_test_binary = (y_test == 0).astype(int)
y_pred_binary = y_pred_probs[:, 0]
precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_binary)
avg_precision = average_precision_score(y_test_binary, y_pred_binary)

plt.figure(figsize=(8,6))
plt.plot(recall, precision, color='blue', lw=2,
         label=f'Precision-Recall (AP = {avg_precision:.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Class 0 vs Rest)')
plt.legend(loc="lower left"); plt.grid(True); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0])
plt.show()

# ==============================
# Section 8-12-7: Bar chart comparing models
# ==============================
print("\n--- 8-12-7: Model Comparison Bar Chart ---")
model_names = ['Model A', 'Model B', 'Model C', 'Model D']
accuracies = [0.85, 0.88, 0.92, 0.90]
precisions = [0.82, 0.87, 0.91, 0.89]
recalls = [0.88, 0.89, 0.93, 0.91]
f1_scores = [0.85, 0.88, 0.92, 0.90]

x = np.arange(len(model_names))
width = 0.2
plt.figure(figsize=(12,7))
plt.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='blue')
plt.bar(x - 0.5*width, precisions, width, label='Precision', color='green')
plt.bar(x + 0.5*width, recalls, width, label='Recall', color='orange')
plt.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='red')
plt.xlabel('Models'); plt.ylabel('Score')
plt.title('Comparison of Different Models')
plt.xticks(x, model_names); plt.ylim(0,1)
plt.legend(); plt.grid(True, axis='y')
plt.show()

print("\n=== End of Chapter 8 script ===")