# sign_language_recognition

A real-time American Sign Language (ASL) recognition system that converts hand gestures into text and speech using computer vision, machine learning, and a custom Tkinter GUI.

## Features
- Real-time ASL alphabet recognition using webcam input
- Accurate letter prediction using MediaPipe hand landmarks and a trained MLPClassifier
- Sentence formation with support for space, backspace, clear, and speech output
- Live prediction display with flicker-free performance
- Fully interactive and themed Tkinter GUI
- Toggleable ASL gesture guide displayed beside the webcam feed
- Key controls for sentence manipulation and system actions

## Structure
- `asl_landmarks_dataset.npy` – Full landmark dataset extracted from ASL alphabet images
- `extract_landmarks_from_asl_dataset.py` – Script to extract landmarks directly from the image dataset
- `final_app_tkinter.py` – Main GUI application for real-time sign language recognition
- `generate_landmark_data.py` – Extracts and saves hand landmarks and labels from images
- `gesture_guide.png` – ASL alphabet image displayed in the GUI as a reference
- `label_map.npy` – Mapping of label names to encoded numerical values
- `landmark_data.npy` – Numpy array of hand landmark features used for training
- `landmark_labels.npy` – Numpy array of labels corresponding to training data
- `predict_letters.py` – Live letter prediction using webcam feed
- `predict_to_sentence.py` – (Legacy) script for building sentences from predictions
- `scaler.pkl` – Scaler object used to normalize input features before prediction
- `sign_language_model.pkl` – Trained MLPClassifier model for gesture classification
- `test_model.py` – Tests model accuracy and performance on test data
- `train_model.py` – Trains the sign language model using landmark features and labels
- `x_data.npy` – Combined input features (training set)
- `X_test.npy` – Input features used for testing the model
- `x_y_test_files.py` – Script to split data and generate test files
- `y_labels.npy` – Labels used for training the model
- `y_test.npy` – Labels used for testing

## Installation Instructions
1. Clone the repository
2. Install required Python libraries listed in `requirements.txt`
3. Generate landmark data using `generate_landmark_data.py`
4. Train the model using `train_model.py`
5. Run the main GUI using `final_app_tkinter.py`

## Controls
- `Enter`: Add current prediction to sentence
- `Space`: Add a space
- `Backspace`: Delete the last character
- `C`: Clear the current sentence
- `S`: Speak the sentence aloud
- `G`: Toggle gesture guide display
- `Q`: Quit the application

## Model Information
- Input Features: 63 values from MediaPipe hand landmarks
- Model Type: MLPClassifier (from scikit-learn)
- Accuracy: Approximately 99% on test data
- Preprocessing: StandardScaler used for feature normalization
- Dataset: ASL alphabet image dataset

## Usage Guidelines
- Use consistent hand positioning and good lighting for optimal predictions.
- The application window displays the webcam feed with the current prediction and the sentence being formed.
- The ASL gesture guide can be toggled on or off for reference.
- All key controls work repeatedly and are supported by on-screen buttons.

## Author
Amrutha Girishkumar
