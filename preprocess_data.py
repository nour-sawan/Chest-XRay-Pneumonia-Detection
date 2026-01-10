import tensorflow as tf

# Configuration
# =========================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Load datasets from folders
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "archive/chest_xray/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "archive/chest_xray/val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "archive/chest_xray/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Normalization layer
# =========================
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

# Data augmentation layers
# (Training only)
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),        # ~ ±10 degrees
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.1),
])


# Apply preprocessing
# =========================

# Training: normalization + augmentation
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(normalization_layer(x)), y),
    num_parallel_calls=AUTOTUNE
)

# Validation: normalization only
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
)

# Test: normalization only
test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE
)


# Performance optimization
# =========================
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Sanity check (optional)
#for images, labels in train_ds.take(1):
   ### print("Images shape:", images.shape)
   ### print("Labels shape:", labels.shape)
    ### print("Pixel range:", tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy())