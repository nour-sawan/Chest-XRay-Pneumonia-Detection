##importing necessary libraries
from pathlib import Path
import matplotlib.pyplot as plt
import random

##defining data directory
DATA_DIR = Path("archive/chest_xray")
train_dir = DATA_DIR / "train"
normal_images = list((train_dir / "NORMAL").glob("*")) 
pneumonia_images = list((train_dir / "PNEUMONIA").glob("*"))

##displaying sample images from each category
plt.figure(figsize=(8, 8))
for i, img_path in enumerate(random.sample(normal_images, 2) +
                             random.sample(pneumonia_images, 2)):
    img = plt.imread(img_path)
    
    plt.subplot(2, 2, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(img_path.parent.name)
    plt.axis("off")

plt.show()

##calculating and printing class distribution
normal_count = len(normal_images)
pneumonia_count = len(pneumonia_images)
total = normal_count + pneumonia_count
print(f"Total training images: {total}")
print(f"NORMAL: {normal_count} ({normal_count/total:.2%})")
print(f"PNEUMONIA: {pneumonia_count} ({pneumonia_count/total:.2%})")

##visualizing class distribution
labels = ["NORMAL", "PNEUMONIA"]
counts = [normal_count, pneumonia_count]
plt.figure(figsize=(6, 4))
plt.bar(labels, counts)
plt.title("Class Distribution in Training Set")
plt.ylabel("Number of Images")
plt.xlabel("Class")
plt.show()

# Take a few images to inspect their shapes
DATA_DIR2 = Path("archive/chest_xray/train/NORMAL")
image_paths = list(DATA_DIR2.glob("*"))[:5]

for img_path in image_paths:
    img = plt.imread(img_path)
    print(img_path.name, "-> shape:", img.shape)

##results were : 
# IM-0115-0001.jpeg -> shape: (1858, 2090)
# IM-0117-0001.jpeg -> shape: (1152, 1422)
# IM-0119-0001.jpeg -> shape: (1434, 1810)
# IM-0122-0001.jpeg -> shape: (1279, 1618)
# IM-0125-0001.jpeg -> shape: (1125, 1600)
# They vary in size, so I will need to resize them  before training a CNN.
    