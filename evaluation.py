import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess_data import test_ds

# =========================
# Load trained model
# =========================
MODEL_PATH = "chest_xray_densenet_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# =========================
# Collect true labels & predictions
# =========================
y_true = []
y_pred = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)
    predictions = (probabilities >= 0.5).astype(int)

    y_true.extend(labels.numpy().astype(int).flatten())
    y_pred.extend(predictions.flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =========================
# Accuracy
# =========================
accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# =========================
# Precision / Recall / F1-score
# =========================
print("\nClassification Report (Precision / Recall / F1-score):")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4
    )
)

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
