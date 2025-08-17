import os
import glob
import time
import json
import torch
import smplx
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


# ----------------------------
# 1) Load SMPL-X and MANO
# ----------------------------
model = smplx.create(
    model_path='models',
    model_type='smplx',
    gender='male',
    use_pca=False,
    ext='pkl',
    create_global_orient=True,
    create_body_pose=True,
    create_betas=True,
    create_left_hand_pose=True,
    create_right_hand_pose=True,
    create_jaw_pose=True,
    create_leye_pose=True,
    create_reye_pose=True,
    create_expression=True
)
faces = model.faces

mano_right = smplx.create(
    model_path='models',
    model_type='mano',
    is_rhand=True,
    use_pca=False,
    num_pca_comps=30,
    create_global_orient=True,
    create_hand_pose=True,
    create_transl=True,
    ext='pkl'
)
mano_left = smplx.create(
    model_path='models',
    model_type='mano',
    is_rhand=False,
    use_pca=False,
    num_pca_comps=30,
    create_global_orient=True,
    create_hand_pose=True,
    create_transl=True,
    ext='pkl'
)

# ----------------------------
# 2) Load body params with default hand params
# ----------------------------
timestamp = 624536
vertices = []

with open(f'device_0_record_2/body/{timestamp}_0.json') as f: 
    body0       = json.load(f)
bd0             = body0['pred_smpl_params']
transl0         = torch.tensor(body0['pred_cam_t_full'][0], dtype=torch.float32).unsqueeze(0)
focal_length    = body0['scaled_focal_length']
go0             = torch.tensor(R.from_matrix(np.array(bd0['global_orient'])).as_rotvec(), dtype=torch.float32).unsqueeze(0)
bp0             = torch.tensor(R.from_matrix(np.array(bd0['body_pose'])).as_rotvec().reshape(-1)[:63], dtype=torch.float32).unsqueeze(0)
betas0          = torch.tensor(bd0['betas'], dtype=torch.float32).unsqueeze(0)

with open(f'device_0_record_2/body/{timestamp}_1.json') as f: 
    body1       = json.load(f)
bd1             = body1['pred_smpl_params']
transl1         = torch.tensor(body1['pred_cam_t_full'][1], dtype=torch.float32).unsqueeze(0)
go1             = torch.tensor(R.from_matrix(np.array(bd1['global_orient'])).as_rotvec(), dtype=torch.float32).unsqueeze(0)
bp1             = torch.tensor(R.from_matrix(np.array(bd1['body_pose'])).as_rotvec().reshape(-1)[:63], dtype=torch.float32).unsqueeze(0)
betas1          = torch.tensor(bd1['betas'], dtype=torch.float32).unsqueeze(0)

lhp             = torch.zeros(1, 45, dtype=torch.float32)
rhp             = torch.zeros(1, 45, dtype=torch.float32)

with torch.no_grad():
    output0 = model(
        global_orient   = go0,
        body_pose       = bp0,
        betas           = betas0,
        left_hand_pose  = lhp,
        right_hand_pose = rhp, 
        transl          = transl0
    )
    joints0             = output0.joints[0].cpu().detach().numpy()
    print("Left Wrist: ", joints0[20], ", Right Wrist: ", joints0[21])
    vertices.append(output0.vertices.detach().cpu().numpy().squeeze(0))

    output1 = model(
        global_orient   = go1,
        body_pose       = bp1,
        betas           = betas1,
        left_hand_pose  = lhp,
        right_hand_pose = rhp,
        transl          = transl1
    )
    joints1             = output1.joints[0].cpu().detach().numpy()
    print("Left Wrist: ", joints1[20], ", Right Wrist: ", joints1[21])
    vertices.append(output1.vertices.detach().cpu().numpy().squeeze(0))

# ----------------------------
# 3) Load hand params
# ----------------------------
S = np.diag([-1.0, 1.0, 1.0])
idx = 0
hands = []
while True: 
    try: 
        with open(f'device_0_record_2/hand/{timestamp}_{idx}.json') as f: 
            hand_data       = json.load(f)
        hand                = hand_data['pred_mano_params']
        hand_transl         = torch.tensor(hand_data['pred_cam_t_full'][idx], dtype=torch.float32).unsqueeze(0)
        is_right            = hand_data['is_right']

        hand_go_mats        = np.array(hand['global_orient'])
        hp_mats             = np.array(hand['hand_pose'])

        if not is_right: 
            hand_go_mats    = S @ hand_go_mats @ S
            hp_mats         = S[None, :, :] @ hp_mats @ S

        hand_go             = torch.tensor(R.from_matrix(hand_go_mats).as_rotvec(), dtype=torch.float32)
        hp                  = torch.tensor(R.from_matrix(hp_mats).as_rotvec().reshape(-1), dtype=torch.float32).unsqueeze(0)
        hand_betas          = torch.tensor(hand['betas'], dtype=torch.float32).unsqueeze(0)
        
        print(f'Hand {idx}: {is_right}')
        mano = mano_right if is_right else mano_left
        print(mano)
        hand_output = mano(
            global_orient   = hand_go,
            hand_pose       = hp,
            betas           = hand_betas, 
            transl          = hand_transl
        )
        hand_joints         = hand_output.joints[0].cpu().detach().numpy()
        hand_wrist          = hand_joints[0]
        print(f'Hand {idx} Wrist: {hand_wrist}')
        vertices.append(hand_output.vertices.detach().cpu().numpy().squeeze(0))

        idx += 1
    except FileNotFoundError: 
        break

# ----------------------------
# 4) Visualization
# ----------------------------
body_faces = np.asarray(model.faces)
hand_faces = np.asarray(mano.faces)

geoms = []
for i in range(2):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
    mesh.triangles = o3d.utility.Vector3iVector(body_faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8])
    geoms.append(mesh)
for j in range(2, len(vertices)):
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
    fx=focal_length, 
    fy=focal_length,
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
ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

vis.run()
vis.destroy_window()