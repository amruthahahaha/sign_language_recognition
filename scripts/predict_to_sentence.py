import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
from collections import deque, Counter

# Load model and scaler
with open('sign_language_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
prev_time = 0
delay = 1.0  # seconds between accepted predictions

sentence = ""
engine = pyttsx3.init()

# Prediction buffer for stability
prediction_buffer = deque(maxlen=7)  # hold last 5 predictions

print("🔤 Controls: Enter=Add | Space=Space | S=Speak | C=Clear | Backspace=Delete | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_features = []
    stable_prediction = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_features.extend([lm.x, lm.y, lm.z])

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(landmark_features) == 63:
            scaled_features = scaler.transform([landmark_features])
            probs = model.predict_proba(scaled_features)[0]
            max_prob = np.max(probs)
            predicted_letter = model.classes_[np.argmax(probs)]

            # Confidence thresholding
            if max_prob > 0.85:
                prediction_buffer.append(predicted_letter)

            # Use most frequent prediction in buffer
            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common = Counter(prediction_buffer).most_common(1)[0]
                if most_common[1] >= 3:  # at least 3 matches
                    stable_prediction = most_common[0]

    # Display prediction
    if stable_prediction:
        cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Prediction: ...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

    # Display sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 13 and stable_prediction:  # Enter key
        sentence += stable_prediction
        prediction_buffer.clear()
    elif key == 32:  # Spacebar
        sentence += ' '
    elif key == ord('c'):  # Clear sentence
        sentence = ''
    elif key == 8:  # Backspace
        sentence = sentence[:-1]
    elif key == ord('s'):  # Speak sentence
        if sentence.strip():
            engine.say(sentence)
            engine.runAndWait()
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
