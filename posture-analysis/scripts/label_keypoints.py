import json
import math
import numpy as np
from scripts.load_data import load_unified_keypoints


def calculate_angle(p1, p2, p3):
    if p1 == [0, 0] or p2 == [0, 0] or p3 == [0, 0]:
        return None

    a = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    b = (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2
    c = (p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2

    if a == 0 or b == 0:
        return None

    try:
        angle = math.acos((a + b - c) / (2 * math.sqrt(a) * math.sqrt(b)))
    except ValueError:
        return None

    return math.degrees(angle)

def is_within_threshold(angle, min_angle, max_angle):
    if angle is None:
        return True
    return min_angle <= angle <= max_angle


def label_frame(keypoints):
    labels = []

    def get_keypoint(keypoints, name):
        return keypoints.get(name, [0, 0])
    print("---------------프레임 분석 시작--------------")
    # 머리와 목 정렬
    neck = get_keypoint(keypoints, "neck")
    head_top = get_keypoint(keypoints, "head_top")
    head_neck_angle = calculate_angle(neck, [neck[0], neck[1]-1], head_top)
    if not is_within_threshold(head_neck_angle, 160, 180):
        labels.append('bad_head_neck_angle')

    # 오른쪽 어깨와 팔
    right_shoulder = get_keypoint(keypoints, "right_shoulder")
    right_elbow = get_keypoint(keypoints, "right_elbow")
    right_wrist = get_keypoint(keypoints, "right_wrist")
    right_arm_angle = calculate_angle(neck, right_shoulder, right_elbow)
    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    print("오른쪽 팔꿈치 각도: " + str(right_elbow_angle))
    # if not is_within_threshold(right_arm_angle, 60, 180):
    #     labels.append('bad_right_arm_angle')
    if not is_within_threshold(right_elbow_angle, 60, 180):
        labels.append('bad_right_elbow_angle')

    # 왼쪽 어깨와 팔
    left_shoulder = get_keypoint(keypoints, "left_shoulder")
    left_elbow = get_keypoint(keypoints, "left_elbow")
    left_wrist = get_keypoint(keypoints, "left_wrist")
    left_arm_angle = calculate_angle(neck, left_shoulder, left_elbow)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    print("왼쪽 팔꿈치 각도: " + str(left_elbow_angle))
    # if not is_within_threshold(left_arm_angle, 60, 180):
    #     labels.append('bad_left_arm_angle')
    if not is_within_threshold(left_elbow_angle, 60, 180):
        labels.append('bad_left_elbow_angle')

    # 골반 정렬
    # pelvis = get_keypoint(keypoints, "pelvis")
    right_hip = get_keypoint(keypoints, "right_hip")
    left_hip = get_keypoint(keypoints, "left_hip")
    # pelvis_angle = calculate_angle(right_hip, pelvis, left_hip)
    # if not is_within_threshold(pelvis_angle, 70, 110):
    #     labels.append('bad_pelvis_angle')

    # 오른쪽 다리
    right_knee = get_keypoint(keypoints, "right_knee")
    right_ankle = get_keypoint(keypoints, "right_ankle")
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
    print("오른쪽 무릎 각도: " + str(right_leg_angle))
    if not is_within_threshold(right_leg_angle, 140, 180):
        labels.append('bad_right_leg_angle')

    # 왼쪽 다리
    left_knee = get_keypoint(keypoints, "left_knee")
    left_ankle = get_keypoint(keypoints, "left_ankle")
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    print("왼쪽 무릎 각도: " + str(left_leg_angle))

    if not is_within_threshold(left_leg_angle, 140, 180):
        labels.append('bad_left_leg_angle')

    # 몸의 좌우 균형
    # if right_arm_angle is not None and left_arm_angle is not None and abs(right_arm_angle - left_arm_angle) > 20:
    #     labels.append('bad_arm_balance')
    # if right_leg_angle is not None and left_leg_angle is not None and abs(right_leg_angle - left_leg_angle) > 20:
    #     labels.append('bad_leg_balance')


    if len(labels) > 2:
        return 'bad_posture', labels
    else:
        return 'good_posture', []


def label_all_frames(keypoints_list):
    labeled_keypoints = []
    for keypoints in keypoints_list:
        label, details = label_frame(keypoints["keypoints"])
        labeled_keypoints.append({
            "source": keypoints["source"],
            "keypoints": keypoints["keypoints"],
            "label": label,
            "details": details
        })
    return labeled_keypoints


def save_labeled_keypoints(labeled_keypoints, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_keypoints, f, ensure_ascii=False, indent=4)


def main():
    input_path = '../data/processed/unified_keypoints_02.json'
    output_path = '../data/processed/labeled_keypoints_02.json'

    keypoints_list = load_unified_keypoints(input_path)

    labeled_keypoints = label_all_frames(keypoints_list)

    save_labeled_keypoints(labeled_keypoints, output_path)

    print(f"Labeled keypoints saved to {output_path}")


if __name__ == "__main__":
    main()
