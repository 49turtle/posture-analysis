import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load and parse JSON files
def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    keypoints = [item['keypoints'][0] for item in data]
    labels = [item['good_posture'] for item in data]
    return np.array(keypoints), np.array(labels)

# Load the evaluation data
eval_keypoints, eval_labels = load_keypoints('data/processed/bad_keypoints_02.json')

# Normalize keypoints
eval_keypoints = eval_keypoints / np.max(eval_keypoints)

# Load the trained model
model = load_model('models/posture_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(eval_keypoints, eval_labels)
print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
