import os
import glob
import time
import json
import torch
import smplx
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

mano_right = smplx.create(
    model_path='models',
    model_type='mano',
    use_pca=False,
    num_pca_comps=30,
    create_global_orient=True,
    create_hand_pose=True,
    create_transl=True,
    gender='right',
    ext='pkl'
)
mano_left = smplx.create(
    model_path='models',
    model_type='mano',
    use_pca=False,
    num_pca_comps=30,
    create_global_orient=True,
    create_hand_pose=True,
    create_transl=True,
    gender='left',
    ext='pkl'
)

# 2. load default body params
timestamp = 623307
vertices = []

# 3. load hand params
S = np.diag([1.0, -1.0, -1.0])
idx = 0
hands = []
with open(f'device_0_record_2/hand/{timestamp}_{idx}.json') as f: 
    hand_data   = json.load(f)
    hand        = hand_data['pred_mano_params']
    hand_transl = torch.tensor(hand_data['pred_cam_t'][0], dtype=torch.float32).unsqueeze(0)
    is_right    = hand_data['is_right']
hand_go_mats    = np.array(hand['global_orient'])
hp_mats         = np.array(hand['hand_pose'])

# if not is_right: 
#     hand_go_mats = S @ hand_go_mats @ S
#     hp_mats = S @ hp_mats @ S

hand_go     = torch.tensor(R.from_matrix(hand_go_mats).as_rotvec(), dtype=torch.float32)
hp          = torch.tensor(R.from_matrix(hp_mats).as_rotvec().reshape(-1), dtype=torch.float32).unsqueeze(0)
hand_betas  = torch.tensor(hand['betas'], dtype=torch.float32).unsqueeze(0)

print(f'Hand {idx}: {is_right}')
# mano        = mano_right if is_right else mano_left
mano        = mano_right
print(mano)
hand_output = mano(
    global_orient=hand_go,
    hand_pose=hp,
    betas=hand_betas, 
    transl=hand_transl
)
hand_joints = hand_output.joints[0].cpu().detach().numpy()
hand_wrist = hand_joints[0]
print(f'Hand {idx} Wrist: {hand_wrist}')
vertices.append(hand_output.vertices.detach().cpu().numpy().squeeze(0))

mano        = mano_left
print(mano)
hand_output = mano(
    global_orient=hand_go,
    hand_pose=hp,
    betas=hand_betas, 
    transl=hand_transl
)
vertices.append(hand_output.vertices.detach().cpu().numpy().squeeze(0))

# 4. visualization
hand_faces = np.asarray(mano.faces)        # (F_hand, 3)

geoms = []
for j in range(len(vertices)):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[j])
    mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 0.0, 0.0])
    geoms.append(mesh)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
geoms.append(axis)

img_w, img_h = 1920, 1080
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    width=img_w, 
    height=img_h,
    fx=37500, 
    fy=37500,
    cx=img_w/2, 
    cy=img_h/2
)
extrinsic = np.eye(4)
cam_params = o3d.camera.PinholeCameraParameters()
cam_params.intrinsic = intrinsic
cam_params.extrinsic = extrinsic

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='SMPL-X + MANO', width=img_w, height=img_h)
for g in geoms:
    vis.add_geometry(g)

ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(cam_params)

vis.run()
vis.destroy_window()