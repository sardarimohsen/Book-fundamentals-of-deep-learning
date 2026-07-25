# ============================================================
# Chapter 9: Data Preprocessing
# Sections: 9-2-1, 9-3-2, 9-3-3, 9-4-2, 9-5-1, 9-5-1-1
#           9-5-2, 9-5-3, 9-6-1, 9-7, 9-9, 9-10-1
#           9-10-2, 9-11, 9-12-1, 9-13, 9-14-1 to 9-14-6
# ============================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats
import matplotlib.pyplot as plt
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.layers import (RandomRotation, RandomTranslation,
                                     RandomZoom, RandomFlip, RandomBrightness,
                                     RandomContrast, Normalization, Rescaling,
                                     StringLookup, CategoryEncoding)
from tf.keras.preprocessing.image import ImageDataGenerator

# ==============================
# Section 9-2-1: Feature Selection (RFE & Importance)
# ==============================
print("\n--- 9-2-1: Feature Selection ---")
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFE(estimator, n_features_to_select=10)
selector.fit(X, y)
print("Selected features (RFE):", selector.get_support())
print("Feature ranking:", selector.ranking_)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)
importances = model_rf.feature_importances_
for i, imp in enumerate(importances[:5]):
    print(f"Feature {i} importance: {imp:.4f}")

# ==============================
# Section 9-3-2: Train-Test Split & Shuffle
# ==============================
print("\n--- 9-3-2: Train-Test Split ---")
x_data = np.random.rand(100, 10)
y_data = np.random.randint(0, 2, 100)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42, shuffle=True)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# ==============================
# Section 9-3-3: Stratified Split
# ==============================
print("\n--- 9-3-3: Stratified Split ---")
x_train_str, x_test_str, y_train_str, y_test_str = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)
print("Stratified - y_train class distribution:", np.bincount(y_train_str))

# ==============================
# Section 9-4-2: K-Fold & Stratified K-Fold
# ==============================
print("\n--- 9-4-2: K-Fold ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(x_data):
    scores.append(np.random.rand())  # dummy score
print("K-Fold mean score:", np.mean(scores))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# skf.split(x_data, y_data) is ready to use
print("Stratified K-Fold ready.")

# ==============================
# Section 9-5-1: Missing Data (pandas)
# ==============================
print("\n--- 9-5-1: Missing Data (pandas) ---")
df = pd.DataFrame({
    'Age': [25, 30, np.nan, 40, 35],
    'Income': [50000, 60000, 55000, np.nan, 70000],
    'Gender': ['Male', 'Female', 'Male', 'Female', np.nan]
})
print("Missing values:\n", df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Income'] = df['Income'].fillna(df['Income'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
print("After imputation:\n", df)

# ==============================
# Section 9-5-1-1: KNN Imputer
# ==============================
print("\n--- 9-5-1-1: KNN Imputer ---")
X_missing = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X_missing)
print("KNN imputed:\n", X_imputed)

# ==============================
# Section 9-5-2: Duplicate Data
# ==============================
print("\n--- 9-5-2: Duplicate Data ---")
df_dup = pd.DataFrame({'A': [1,2,2,3], 'B': [4,4,4,5]})
print("Duplicates count:", df_dup.duplicated().sum())
df_dup = df_dup.drop_duplicates()
print("After drop:\n", df_dup)

# ==============================
# Section 9-5-3: Outlier Detection (IQR)
# ==============================
print("\n--- 9-5-3: Outlier IQR ---")
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (data < lower) | (data > upper)

data_sample = pd.Series([10,20,30,40,50,200])
outliers = detect_outliers_iqr(data_sample)
print("Outlier indices:", outliers[outliers].index.tolist())

# ==============================
# Section 9-6-1: Min-Max Normalization
# ==============================
print("\n--- 9-6-1: Min-Max Normalization ---")
def min_max_normalize(x, a=0, b=1):
    x_min, x_max = np.min(x), np.max(x)
    return a + (x - x_min) * (b - a) / (x_max - x_min)
data_norm = np.array([10,20,30,40,50])
normalized = min_max_normalize(data_norm)
print("Normalized:", normalized)

# ==============================
# Section 9-7: Standardization (Z-score)
# ==============================
print("\n--- 9-7: Standardization ---")
def standardize(x):
    mu, sigma = np.mean(x), np.std(x)
    return (x - mu) / sigma
std_data = standardize(data_norm)
print("Standardized:", std_data)
print("Mean:", np.mean(std_data), "Std:", np.std(std_data))

# ==============================
# Section 9-9: Box-Cox Transform
# ==============================
print("\n--- 9-9: Box-Cox Transform ---")
data_exp = np.random.exponential(scale=2, size=1000)
data_trans, lambda_opt = stats.boxcox(data_exp + 1e-6)  # add small epsilon to avoid zero
print("Optimal lambda:", lambda_opt)

# ==============================
# Section 9-10-1: Label Encoding
# ==============================
print("\n--- 9-10-1: Label Encoding ---")
le = LabelEncoder()
colors = ['Red', 'Blue', 'Green', 'Blue', 'Red']
encoded = le.fit_transform(colors)
print("Label encoded:", encoded)

# ==============================
# Section 9-10-2: One-Hot Encoding
# ==============================
print("\n--- 9-10-2: One-Hot Encoding ---")
colors_array = np.array([['Red'], ['Red'], ['Green'], ['Red'], ['Red']])
encoder = OneHotEncoder(sparse_output=False)
onehot = encoder.fit_transform(colors_array)
print("One-hot encoded (first 3 rows):\n", onehot[:3])

# ==============================
# Section 9-11: Padding
# ==============================
print("\n--- 9-11: Padding ---")
sequences = [[1,2,3,4], [5,6], [7,8,9,10,11,12,6]]
padded = pad_sequences(sequences, maxlen=6, padding='post', truncating='post')
print("Padded sequences:\n", padded)

# ==============================
# Section 9-12-1: Augmentation Layers
# ==============================
print("\n--- 9-12-1: Augmentation Layers ---")
rot = RandomRotation(factor=0.2)
trans = RandomTranslation(height_factor=0.2, width_factor=0.2)
zoom = RandomZoom(height_factor=0.2, width_factor=0.2)
flip = RandomFlip(mode='horizontal')
bright = RandomBrightness(factor=0.2)
contrast = RandomContrast(factor=0.2)
print("All augmentation layers instantiated.")

# ==============================
# Section 9-13: Imbalanced Data (SMOTE placeholder)
# ==============================
print("\n--- 9-13: Imbalanced Data (SMOTE) ---")
# Uncomment if imblearn is installed:
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X, y)
print("SMOTE and other resamplers available via imblearn library.")

# ==============================
# Section 9-14-1: Normalization Layer
# ==============================
print("\n--- 9-14-1: Normalization Layer ---")
norm_layer = Normalization()
dummy_data = np.random.rand(100, 5)
norm_layer.adapt(dummy_data)
print("Normalization layer adapted.")

# ==============================
# Section 9-14-2: Rescaling Layer
# ==============================
print("\n--- 9-14-2: Rescaling Layer ---")
rescaling = Rescaling(1./255)
print("Rescaling layer ready.")

# ==============================
# Section 9-14-3: Category Encoding Layers
# ==============================
print("\n--- 9-14-3: Category Encoding ---")
vocab = ['Red', 'Blue', 'Green']
lookup = StringLookup(vocabulary=vocab, output_mode='int')
encoding = CategoryEncoding(num_tokens=len(vocab)+1, output_mode='one_hot')
print("StringLookup and CategoryEncoding ready.")

# ==============================
# Section 9-14-4: ImageDataGenerator (placeholder path)
# ==============================
print("\n--- 9-14-4: ImageDataGenerator ---")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# train_generator = datagen.flow_from_directory('path/to/train', ...)
print("ImageDataGenerator configured. (Replace 'path/to/train' with actual path)")

# ==============================
# Section 9-14-5: Augmentation as Sequential model
# ==============================
print("\n--- 9-14-5: Augmentation Sequential ---")
data_aug_seq = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomTranslation(0.2, 0.2),
    RandomBrightness(0.2),
    RandomContrast(0.2)
])
print("Augmentation Sequential model ready.")

# ==============================
# Section 9-14-6: tf.data.Dataset (placeholder files)
# ==============================
print("\n--- 9-14-6: tf.data.Dataset ---")
def load_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Dummy placeholders - replace with actual file paths to run
file_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
labels = [0, 1, 0]
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
# dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
# dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
print("tf.data pipeline ready. (Uncomment and provide real image files to run)")

print("\n=== End of Chapter 9 script ===")