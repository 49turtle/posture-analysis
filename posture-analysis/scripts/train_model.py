# train_model.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Function to load and parse JSON files
def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    keypoints = [item['keypoints'][0] for item in data]
    labels = [item['good_posture'] for item in data]
    return np.array(keypoints), np.array(labels)

# Load the data from the provided JSON files
bad_keypoints_01, labels_01 = load_keypoints('data/processed/bad_keypoints_01.json')
good_keypoints_01, labels_02 = load_keypoints('data/processed/good_keypoints_01.json')

# Combine the data
keypoints = np.concatenate((bad_keypoints_01, good_keypoints_01), axis=0)
labels = np.concatenate((labels_01, labels_02), axis=0)

# Normalize keypoints
keypoints = keypoints / np.max(keypoints)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(keypoints, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Flatten(input_shape=(keypoints.shape[1], keypoints.shape[2])),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# Save the model
model.save('models/posture_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
