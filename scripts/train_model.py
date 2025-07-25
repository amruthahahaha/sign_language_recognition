import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load data
data = np.load('asl_landmarks_dataset.npy', allow_pickle=True).item()
X = np.array(data['data'])
y = np.array(data['labels'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
with open('sign_language_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("💾 Model & scaler saved.")
