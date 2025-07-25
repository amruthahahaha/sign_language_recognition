import os
import cv2
import mediapipe as mp
import numpy as np

# ===== PATH =====
dataset_path =  "C:\\Users\\Naveen\\Desktop\\sign_language_project\\asl_alphabet_train\\asl_alphabet_train"

# ===== SETUP =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
labels = []
data = []

# ===== PROCESS =====
print("🟡 Starting landmark extraction...")
folders = os.listdir(dataset_path)

for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"🔍 Processing folder: {folder}")
    images = os.listdir(folder_path)[:300]  # Optional: limit to 300 per class
    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            flattened = []
            for lm in landmarks.landmark:
                flattened.extend([lm.x, lm.y, lm.z])
            data.append(flattened)
            labels.append(folder)

print(f"✅ Landmark data collected: {len(data)} samples")
output_file = "asl_landmarks_dataset.npy"
print("💾 Saving to:", output_file)

np.save(output_file, {'data': data, 'labels': labels})
print("🎉 Done! File saved as:", output_file)
