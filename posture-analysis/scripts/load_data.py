import json
import numpy as np

def load_ai_hub_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints = []
    for annotation in data['annotations']:
        keypoints.append(annotation['keypoints'])
    return keypoints


def load_openpose_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints = []
    for frame in data:
        keypoints.extend(frame['keypoints'])
    return keypoints



def load_unified_keypoints(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



# def extract_features(data):
#     features = []
#     for frame_data in data:
#         keypoints = frame_data.get("keypoints", {})
#         if keypoints:
#             flattened_keypoints = []
#             for k, v in keypoints.items():
#                 flattened_keypoints.extend(v)
#             features.append(flattened_keypoints)
#     return np.array(features)

def load_labeled_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    keypoints = []
    labels = []
    for frame_data in data:
        frame_keypoints = frame_data['keypoints']
        keypoints.append(frame_keypoints)
        labels.append(1 if frame_data['good_posture'] else 0)
    return np.array(keypoints), np.array(labels)



def extract_features(keypoints):
    # Flatten the keypoints to create feature vectors
    return keypoints.reshape(keypoints.shape[0], -1)