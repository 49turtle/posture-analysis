import json
from scripts.load_data import load_ai_hub_json, load_openpose_json

# AI-Hub와 OpenPose keypoints 매핑
AI_HUB_KEYPOINTS_MAP = [
    "right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle",
    "pelvis", "thorax", "neck", "head_top", "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist"
]

OPENPOSE_KEYPOINTS_MAP = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
    "left_elbow", "left_wrist", "mid_hip", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear",
    "left_ear", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel"
]

def unify_keypoints_ai_hub(keypoints):
    unified_keypoints = []
    for person in keypoints:
        person_keypoints = {}
        for idx, label in enumerate(AI_HUB_KEYPOINTS_MAP):
            start_idx = idx * 3
            person_keypoints[label] = person[start_idx:start_idx+3]  # x, y, confidence
        unified_keypoints.append({
            "source": "ai_hub",
            "keypoints": person_keypoints
        })
    return unified_keypoints

def unify_keypoints_openpose(keypoints):
    unified_keypoints = []
    for person in keypoints:
        person_keypoints = {}
        for idx, label in enumerate(OPENPOSE_KEYPOINTS_MAP):
            if idx < len(person):
                person_keypoints[label] = person[idx]
            else:
                person_keypoints[label] = [0, 0, 0]  # 없는 경우 빈 값
        unified_keypoints.append({
            "source": "openpose",
            "keypoints": person_keypoints
        })
    return unified_keypoints

def save_unified_keypoints(unified_keypoints, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_keypoints, f, ensure_ascii=False, indent=4)

def main():
    # ai_hub_path = '../data/ai_hub/2-1_001-C01_2D.json'
    openpose_path = 'data/openpose/good_keypoints_01.json'
    output_path = 'data/processed/unified_good_keypoints_01.json'

    # ai_hub_keypoints = load_ai_hub_json(ai_hub_path)
    openpose_keypoints = load_openpose_json(openpose_path)

    # unified_ai_hub_keypoints = unify_keypoints_ai_hub(ai_hub_keypoints)
    unified_openpose_keypoints = unify_keypoints_openpose(openpose_keypoints)

    # unified_keypoints = unified_ai_hub_keypoints + unified_openpose_keypoints

    # save_unified_keypoints(unified_keypoints, output_path)
    save_unified_keypoints(unified_openpose_keypoints, output_path)

    print(f"Unified keypoints saved to {output_path}")

if __name__ == "__main__":
    main()
