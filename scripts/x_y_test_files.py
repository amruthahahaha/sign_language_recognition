import os
import cv2
import numpy as np

# Correct path to your dataset
data_path = os.path.join('..', 'dataset', 'asl_alphabet_train')
labels = sorted(os.listdir(data_path))

x_data = []
y_labels = []

# Resize size (must match what model expects — 128 is typical)
IMG_SIZE = 128

print("⏳ Loading and processing images...")

for label in labels:
    label_path = os.path.join(data_path, label)
    if not os.path.isdir(label_path):
        continue

    for i, image_file in enumerate(os.listdir(label_path)):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(label_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x_data.append(image)
            y_labels.append(label)

            # Optional: Limit to 100 images per class
            if i >= 99:
                break

x_data = np.array(x_data)
y_labels = np.array(y_labels)

# Save .npy files for testing
np.save('x_data.npy', x_data)
np.save('y_labels.npy', y_labels)

print("✅ x_data.npy and y_labels.npy created successfully!")
print(f"x_data shape: {x_data.shape}")
print(f"y_labels shape: {y_labels.shape}")
