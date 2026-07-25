# ============================================================
# Chapter 10: Transfer Learning with VGG16 & ResNet50
# Sections: 10-6-3-1, 10-6-3-2, 10-6-3-3, 
#           10-6-3-4, 10-6-3-5
# ============================================================

import tensorflow as tf
from tf.keras import layers, models
from tf.keras.optimizers import Adam
from tf.keras.applications import VGG16, ResNet50, EfficientNetB0, InceptionV3
from tf.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS_FE = 2
EPOCHS_FT = 2
LR = 0.001

# ==============================
# 10-6-3-1: Data Preparation (Placeholder directories)
# ==============================
print("\n--- 10-6-3-1: Data Preparation ---")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# UNCOMMENT AND UPDATE PATHS TO RUN:
# train_generator = train_datagen.flow_from_directory(
#     'path/to/train', target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE, class_mode='categorical')
# val_generator = val_datagen.flow_from_directory(
#     'path/to/validation', target_size=(IMG_SIZE, IMG_SIZE),
#     batch_size=BATCH_SIZE, class_mode='categorical')

print("ImageDataGenerator ready. (Update 'path/to/train' and 'path/to/validation')")

# ==============================
# 10-6-3-2: Feature Extraction with VGG16
# ==============================
print("\n--- 10-6-3-2: VGG16 Feature Extraction ---")
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_vgg.trainable = False

model_vgg_fe = models.Sequential([
    base_vgg,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model_vgg_fe.compile(optimizer=Adam(learning_rate=LR),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
model_vgg_fe.summary()

# Uncomment to train:
# history_fe = model_vgg_fe.fit(train_generator, epochs=EPOCHS_FE,
#                               validation_data=val_generator, verbose=1)

# ==============================
# 10-6-3-3: Fine-Tuning VGG16
# ==============================
print("\n--- 10-6-3-3: VGG16 Fine-Tuning ---")
base_vgg.trainable = True
for layer in base_vgg.layers[:15]:
    layer.trainable = False
# Recompile with lower LR
model_vgg_fe.compile(optimizer=Adam(learning_rate=LR/10),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Uncomment to continue training:
# history_ft = model_vgg_fe.fit(train_generator,
#                               epochs=EPOCHS_FE + EPOCHS_FT,
#                               initial_epoch=EPOCHS_FE,
#                               validation_data=val_generator, verbose=1)

print("VGG16 fine-tuning ready (uncomment to run).")

# ==============================
# 10-6-3-4: Using ResNet50
# ==============================
print("\n--- 10-6-3-4: ResNet50 Transfer Learning ---")
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_resnet.trainable = False

model_resnet_fe = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model_resnet_fe.compile(optimizer=Adam(learning_rate=LR),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

# Fine-tune ResNet50: unfreeze conv5/conv4
base_resnet.trainable = True
for layer in base_resnet.layers:
    if 'conv5' in layer.name or 'conv4' in layer.name:
        layer.trainable = True
    else:
        layer.trainable = False
model_resnet_fe.compile(optimizer=Adam(learning_rate=LR/10),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
print("ResNet50 transfer learning ready.")

# ==============================
# 10-6-3-5: Helper function to load models
# ==============================
print("\n--- 10-6-3-5: Helper Function ---")
def load_pre_trained_model(model_name, input_shape=(224,224,3)):
    models_dict = {
        'vgg16': VGG16,
        'resnet50': ResNet50,
        'efficientnet': EfficientNetB0,
        'inceptionv3': InceptionV3
    }
    model_class = models_dict.get(model_name.lower())
    if model_class is None:
        raise ValueError("Model not supported.")
    return model_class(weights='imagenet', include_top=False, input_shape=input_shape)

base = load_pre_trained_model('resnet50', (224,224,3))
print("Helper function tested successfully.")

print("\n=== End of Chapter 10 Group B (Transfer Learning) ===")