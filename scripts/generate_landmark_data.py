import os
import cv2
import numpy as np
import mediapipe as mp

base_path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(base_path, '../dataset/asl_alphabet_train')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
data = []
labels = []

for label in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(path):
        continue

    print(f"Processing: {label}")
    for img_name in os.listdir(path)[:300]:  # Limit to 300 per class
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmark = []
            for lm in hand.landmark:
                landmark.extend([lm.x, lm.y, lm.z])
            data.append(landmark)
            labels.append(label)

np.save(OUTPUT_FILE, {'data': data, 'labels': labels})
print(f"✅ Saved to {OUTPUT_FILE}")
