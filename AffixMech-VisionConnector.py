import os
import json
import numpy as np
import time

def parse_json_file(file_path):
    """
    Parse a JSON file and extract the grasping data based on the best gripper to use.
    
    Args:
        file_path (str): Path to the JSON file.

    Returns:
        tuple:
            - np.array: PoseList containing all extracted poses.
            - list: A boolean list where True indicates Suction, and False indicates PJ.
            - str: The label if present, None otherwise.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    poses = []
    gripper_choices = []
    label = data.get("label", None)

    if "suction" in data and "pj" in data:
        best_gripper = data.get("bestGripperToUse", None)
        if best_gripper == "suction":
            suction = data["suction"]
            poses.append([
                suction["position"]["x"],
                suction["position"]["y"],
                suction["position"]["z"],
                suction["orientation"]["x"],
                suction["orientation"]["y"],
                suction["orientation"]["z"],
                suction["orientation"]["w"]
            ])
            gripper_choices.append(True)
        elif best_gripper == "pj":
            pj = data["pj"]
            poses.append([
                pj["position"]["x"],
                pj["position"]["y"],
                pj["position"]["z"],
                pj["orientation"]["x"],
                pj["orientation"]["y"],
                pj["orientation"]["z"],
                pj["orientation"]["w"]
            ])
            gripper_choices.append(False)
    else:
        if "suction" in data:
            suction = data["suction"]
            poses.append([
                suction["position"]["x"],
                suction["position"]["y"],
                suction["position"]["z"],
                suction["orientation"]["x"],
                suction["orientation"]["y"],
                suction["orientation"]["z"],
                suction["orientation"]["w"]
            ])
            gripper_choices.append(True)
        if "pj" in data:
            pj = data["pj"]
            poses.append([
                pj["position"]["x"],
                pj["position"]["y"],
                pj["position"]["z"],
                pj["orientation"]["x"],
                pj["orientation"]["y"],
                pj["orientation"]["z"],
                pj["orientation"]["w"]
            ])
            gripper_choices.append(False)

    return np.array(poses, dtype=np.float64), gripper_choices, label

def process():
    """
    Run AffixMain.py as a background process
    Modify this code to Output your desired data

    Returns:
        tuple:
            - np.array: PoseList containing all extracted poses.
            - list: A boolean list where True indicates Suction, and False indicates PJ.
            - list: A list of labels, where None indicates no label present.
    """
    
    directory="AffixData/Results"
    pose_list = []
    gripper_choices = []
    labels = []

    # Get list of JSON files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        poses, choices, label = parse_json_file(file_path)
        pose_list.extend(poses)
        gripper_choices.extend(choices)
        labels.append(label)

        # Delete the processed file
        os.remove(file_path)

    return np.array(pose_list, dtype=np.float64), gripper_choices, labels

def main():
    directory = "AffixData/Results"

    while True:
        if os.path.exists(directory):
            pose_list, gripper_choices, labels = watch_directory(directory)

            if pose_list.size > 0:
                print("PoseList:", pose_list)
                print("Gripper Choices:", gripper_choices)
                print("Labels:", labels)

        time.sleep(1)  # Polling interval

if __name__ == "__main__":
    main()
