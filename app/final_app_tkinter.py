import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
from collections import deque
import warnings

warnings.filterwarnings("ignore")

# Load model and scaler
model = pickle.load(open("sign_language_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Label map
label_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 130)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Globals
sentence = ""
current_letter = ""
buffer = deque(maxlen=5)
show_guide = False
cap = cv2.VideoCapture(0)

# GUI setup
root = tk.Tk()
root.title("Sign Language Recognition")
root.geometry("950x700")
root.configure(bg="#EAF6F6")

# Guide image
guide_img = Image.open("gesture_guide.png").resize((500, 350), Image.Resampling.LANCZOS)
guide_photo = ImageTk.PhotoImage(guide_img)

# Layouts
# Top frame for buttons
top_frame = tk.Frame(root, bg="#EAF6F6")
top_frame.pack(side="top", pady=5)

# Frame for video + guide (centered)
main_frame = tk.Frame(root, bg="#EAF6F6")
main_frame.pack(pady=10)

# Frame to center webcam + guide
video_guide_frame = tk.Frame(main_frame, bg="#EAF6F6")
video_guide_frame.pack()

# Video label (center webcam)
video_label = tk.Label(video_guide_frame, bg="#FFFFFF", width=500, height=400)
video_label.pack(side="left", padx=5)

# Guide image (resized)
guide_img = Image.open("gesture_guide.png").resize((400, 400), Image.Resampling.LANCZOS)
guide_photo = ImageTk.PhotoImage(guide_img)

# Guide label (initially hidden)
guide_label = tk.Label(video_guide_frame, image=guide_photo, bg="#f0f0f0")
guide_label.image = guide_photo  # Prevent garbage collection
guide_label.pack_forget()

# Variables to display
prediction_var = tk.StringVar()
sentence_var = tk.StringVar()

# Prediction label (Comic Sans MS)
prediction_label = tk.Label(root, textvariable=prediction_var,
                            font=("Comic Sans MS", 18, "bold"), fg="#3C91E6", bg="#EAF6F6")
prediction_label.pack(pady=5)

# Sentence label (Times New Roman)
sentence_label = tk.Label(root, textvariable=sentence_var,
                          font=("Times New Roman", 20), fg="#FF6B6B", bg="#EAF6F6")
sentence_label.pack(pady=5)


# Button actions
def speak_sentence():
    if sentence:
        engine.say(sentence)
        engine.runAndWait()

def clear_sentence():
    global sentence
    sentence = ""
    sentence_var.set("sentence: ")

def add_letter():
    global sentence
    if current_letter:
        sentence += current_letter
        sentence_var.set(f"sentence: {sentence}")

def add_space():
    global sentence
    sentence += " "
    sentence_var.set(f"sentence: {sentence}")

def delete_last():
    global sentence
    sentence = sentence[:-1]
    sentence_var.set(f"sentence: {sentence}")

# Flag for guide visibility
guide_visible = True

# Toggle the gesture guide
def toggle_guide():
    global guide_visible
    if guide_visible:
        guide_label.pack_forget()
        guide_visible = False
    else:
        guide_label.pack(side=tk.RIGHT, padx=10)
        guide_visible = True

# Theme toggle support
dark_mode = False
def toggle_theme():
    global dark_mode
    if not dark_mode:
        root.configure(bg="#2e2e2e")
        prediction_label.configure(bg="#2e2e2e", fg="lightgreen")
        sentence_label.configure(bg="#2e2e2e", fg="lightblue")
    else:
        root.configure(bg="#f0f0f0")
        prediction_label.configure(bg="#f0f0f0", fg="darkgreen")
        sentence_label.configure(bg="#f0f0f0", fg="darkblue")
    dark_mode = not dark_mode

# Key bindings
def key_event(event):
    key = event.char.lower()
    if key == 's':
        speak_sentence()
    elif key == 'c':
        clear_sentence()
    elif key == ' ':
        add_space()
    elif key == '\r':
        add_letter()
    elif event.char.lower() == 't':
        toggle_theme()
    elif key == 'g':
        toggle_guide()
    elif event.keysym == 'BackSpace':
        delete_last()
    elif key == 'q':
        root.destroy()

root.bind("<Key>", key_event)

# Buttons
btn_style = {"font": ("Verdana", 10), "bg": "#FFDDC1", "fg": "#333", "padx": 5, "pady": 2}

buttons = [
    ("Add Letter (Enter)", add_letter),
    ("Space (Space)", add_space),
    ("Backspace", delete_last),
    ("Clear (C)", clear_sentence),
    ("Speak (S)", speak_sentence),
    ("Toggle Guide (G)", toggle_guide),
    ("Toggle Theme (T)", toggle_theme),
    ("Quit (Q)", lambda: root.destroy()),
]

for text, cmd in buttons:
    tk.Button(top_frame, text=text, command=cmd, **btn_style).pack(side="left", padx=5)

# Webcam loop
def update_video():
    global current_letter
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    letter = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) == 63:
            X = scaler.transform([landmarks])
            probs = model.predict_proba(X)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            if confidence > 0.85:
                letter = label_map[max_idx]

    buffer.append(letter)
    most_common = max(set(buffer), key=buffer.count)
    current_letter = most_common

    if current_letter:
        prediction_var.set(f"Prediction: {current_letter} ({confidence*100:.1f}%)")
    else:
        prediction_var.set("Prediction: ")

    img = Image.fromarray(rgb)
    img = img.resize((500, 400))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_video)

# Initialize sentence
sentence_var.set("Sentence: ")
update_video()
root.mainloop()
cap.release()
