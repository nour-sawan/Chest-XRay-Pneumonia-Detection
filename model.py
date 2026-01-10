import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

def build_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Builds a transfer learning model using DenseNet121.

    Args:
        input_shape (tuple): Shape of input images (H, W, C)
        num_classes (int): 1 for binary classification

    Returns:
        model (tf.keras.Model): Compiled Keras model
    """

    # Base model (pretrained)
    # =========================
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,        # remove ImageNet classifier
        input_shape=input_shape
    )

    # Freeze base model
    base_model.trainable = False

    # =========================
    # Custom classification head
    # =========================
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Binary output
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    # =========================
    # Build model
    # =========================
    model = models.Model(inputs=base_model.input, outputs=outputs)

    # =========================
    # Compile model
    # =========================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
