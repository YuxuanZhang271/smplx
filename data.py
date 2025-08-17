import glob
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import torch


body_files = sorted(glob.glob('body/*_0.json'))
timestamps = [os.path.basename(p).split('_')[0] for p in body_files]

cam_t = None
focal_length = None
human_pose = []

idx = 0
for timestamp in timestamps: 
    with open(f'body/{timestamp}_0.json', 'r') as f: 
        data = json.load(f)
        global_orient = R.from_matrix(np.array(data['pred_smpl_params']['global_orient'])).as_rotvec()
        body_pose = R.from_matrix(np.array(data['pred_smpl_params']['body_pose'])).as_rotvec()
        betas = data['pred_smpl_params']['betas']
        if cam_t is None: 
            cam_t = np.array(data['pred_cam_t_full'][0])
            focal_length = data['scaled_focal_length']
    
    try: 
        with open(f'hand/{timestamp}_0.json', 'r') as f:
            hl = json.load(f)['pred_mano_params']
        hl_mat = np.array(hl['hand_pose'])
    except FileNotFoundError:
        hl_mat = np.tile(np.eye(3), (15, 1, 1))
    try: 
        with open(f'hand/{timestamp}_1.json', 'r') as f:
            hr = json.load(f)['pred_mano_params']
        hr_mat = np.array(hr['hand_pose'])
    except FileNotFoundError:
        hr_mat = np.tile(np.eye(3), (15, 1, 1))
    left_hand_pose = -R.from_matrix(hl_mat).as_rotvec().reshape(-1)
    right_hand_pose = R.from_matrix(hr_mat).as_rotvec().reshape(-1)

    frame = {
        "id": idx, 
        "timestamp": timestamp, 
        "global_orient": global_orient.tolist(), 
        "body_pose": body_pose.tolist(), 
        "betas": betas, 
        "left_hand_pose": left_hand_pose.tolist(), 
        "right_hand_pose": right_hand_pose.tolist()
    }
    human_pose.append(frame)
    idx += 1

human_pose_dataset = {
    "cam_t": cam_t.tolist(), 
    "focal_length": focal_length, 
    "human_pose": human_pose
}
with open("human_pose_dataset.json", "w") as f:
    json.dump(human_pose_dataset, f, indent=2)