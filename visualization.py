import os
import glob
import time
import json
import torch
import smplx
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# ——— Settings ———
model_folder = 'models'        # SMPL-H model files (.npz or .pkl)
img_w, img_h = 1920, 1080
fps = 30                      # animation frames per second

def load_smplh_frame(ts, model):
    # Load body params
    with open(f'device_0_record_0/body/{ts}_0.json', 'r') as f:
        data = json.load(f)
    bd = data['pred_smpl_params']
    # Load hand params
    try: 
        with open(f'device_0_record_0/hand/{ts}_0.json', 'r') as f:
            hl = json.load(f)['pred_mano_params']
        hl_mat = np.array(hl['hand_pose'])
    except FileNotFoundError:
        hl_mat = np.tile(np.eye(3), (15, 1, 1))
    try: 
        with open(f'device_0_record_0/hand/{ts}_1.json', 'r') as f:
            hr = json.load(f)['pred_mano_params']
        hr_mat = np.array(hr['hand_pose'])
    except FileNotFoundError:
        hr_mat = np.tile(np.eye(3), (15, 1, 1))

    # Convert matrices to axis-angles
    go = R.from_matrix(np.array(bd['global_orient'])).as_rotvec()
    bp = R.from_matrix(np.array(bd['body_pose'])).as_rotvec().reshape(-1)[:63]
    lh = -R.from_matrix(hl_mat).as_rotvec().reshape(-1)
    rh = R.from_matrix(hr_mat).as_rotvec().reshape(-1)

    # Build torch tensors
    global_orient  = torch.tensor(go, dtype=torch.float32).unsqueeze(0)
    body_pose      = torch.tensor(bp, dtype=torch.float32).unsqueeze(0)
    left_hand_pose = torch.tensor(lh, dtype=torch.float32).unsqueeze(0)
    right_hand_pose= torch.tensor(rh, dtype=torch.float32).unsqueeze(0)
    betas          = torch.tensor(bd['betas'], dtype=torch.float32).unsqueeze(0)

    # Forward pass
    out = model(
        global_orient=global_orient,
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose
    )
    return out.vertices.detach().cpu().numpy().squeeze(0)

# 1. Initialize SMPL-H model
model = smplx.create(
    model_folder,
    model_type='smplx',
    gender='male',
    use_pca=False,
    ext='pkl',             # or 'npz'
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

# 2. Gather timestamps
body_files = sorted(glob.glob('device_0_record_0/body/*_0.json'))
timestamps = [os.path.basename(p).split('_')[0] for p in body_files]

# 3. Precompute vertex frames
vertex_frames = [load_smplh_frame(ts, model) for ts in timestamps]

# 4. Setup camera parameters from first frame
with open(f'device_0_record_0/body/{timestamps[0]}_0.json','r') as f:
    cam0 = json.load(f)
pred_cam_t   = cam0['pred_cam_t_full'][0]
focal_length = cam0['scaled_focal_length']

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=img_w, height=img_h,
                         fx=focal_length, fy=focal_length,
                         cx=img_w/2, cy=img_h/2)
extrinsic = np.eye(4)
extrinsic[:3,3] = pred_cam_t
cam_params = o3d.camera.PinholeCameraParameters()
cam_params.intrinsic = intrinsic
cam_params.extrinsic = extrinsic

# 5. Create mesh and visualizer
mesh = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(vertex_frames[0]),
    triangles=o3d.utility.Vector3iVector(faces)
)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.7,0.7,0.7])

vis = o3d.visualization.Visualizer()
vis.create_window(width=img_w, height=img_h)
vis.add_geometry(mesh)
vis.get_render_option().mesh_show_back_face = True
vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

# 6. Animation callback
frame_idx = 0
def animate(vis):
    global frame_idx
    if frame_idx >= len(vertex_frames):
        return False
    mesh.vertices = o3d.utility.Vector3dVector(vertex_frames[frame_idx])
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)
    frame_idx += 1
    time.sleep(1.0/fps)
    return True

vis.register_animation_callback(animate)
vis.run()
vis.destroy_window()
