# score_posture.py

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load and parse JSON files
def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    for item in data:
        if 'keypoints' in item and item['keypoints']:
            keypoints.append(item['keypoints'][0])
    return np.array(keypoints), data

# Load the unlabeled data
keypoints, original_data = load_keypoints('data/processed/temp_test_keypoints.json')

# Check if keypoints array is not empty
if keypoints.size == 0:
    raise ValueError("No keypoints found in the provided JSON file.")

# Normalize keypoints
keypoints = keypoints / np.max(keypoints)

# Load the trained model
model = load_model('models/posture_model.h5')

# Predict good_posture probabilities
predictions = model.predict(keypoints)

# Add predictions to the original data
for i, item in enumerate(original_data):
    item['good_posture_score'] = float(predictions[i])

# Save the results to a new JSON file
with open('data/processed/scored_keypoints.json', 'w') as f:
    json.dump(original_data, f, indent=4)

# Calculate the overall score
overall_score = np.mean(predictions)

print("Scoring complete. Results saved to data/processed/scored_keypoints.json")
print(f"Overall good posture score: {overall_score:.4f}")