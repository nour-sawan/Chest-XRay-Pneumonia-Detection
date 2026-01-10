import tensorflow as tf
import numpy as np
from collections import Counter

from preprocess_data import train_ds, val_ds
from model import build_model
import json
# =========================
# Step 1: Inspect class distribution
# =========================
labels = []

for _, y in train_ds:
    labels.extend(y.numpy().astype(int).flatten())

class_counts = Counter(labels)
print("Class distribution:", class_counts)

# =========================
# Step 2: Compute class weights
# =========================
total_samples = sum(class_counts.values())
num_classes = len(class_counts)

class_weight = {
    int(cls): total_samples / (num_classes * count)
    for cls, count in class_counts.items()
}

print("Class weights:", class_weight)


# =========================
# Step 3: Build the model
# =========================
model = build_model(input_shape=(224, 224, 3))

model.summary()

# =========================
# Step 4: Train the model
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight
)

# =========================
# Step 5: Save the model
# =========================
model.save("chest_xray_densenet_model.h5")
print("Model saved successfully.")

import json
with open("history.json", "w") as f:
    json.dump(history.history, f)
print("History saved to history.json")
