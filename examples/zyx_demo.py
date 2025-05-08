import argparse
import cv2
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import smplx
import torch

def R2aa(rot): 
    return R.from_matrix(rot).as_rotvec()


def aa2R(aa):   
    return R.from_rotvec(aa).as_matrix()


def aa2R(aa, idx):
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while idx != -1:
        joint_rotation = R.from_rotvec(aa[idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        idx = parents[idx]
    return global_rotation


def Rs2aas(rots: np.ndarray) -> np.ndarray:
    aas = []
    for rot in rots:
        aas.append(R2aa(rot))
    return np.stack(aas, axis=0)


def main():
    with open('data/human/115463_0.json', 'r') as f: 
        body                = json.load(f)
        global_orient       = np.array(body['pred_smpl_params']['global_orient'][0])    # (1, 3, 3)
        body_pose           = np.array(body['pred_smpl_params']['body_pose'])[:21]      # (23, 3, 3) -> (21, 3, 3)
        # betas               = np.array(body['pred_smpl_params']['betas'])

        global_orient = R2aa(global_orient).reshape(1, 3)
        body_pose = Rs2aas(body_pose).reshape(21, 3)                                    # (21, 3)

    with open('data/hamer/115463_0.json', 'r') as f: 
        h0                  = json.load(f)
        is_right_h0         = h0['is_right']
        global_orient_h0    = np.array(h0['pred_mano_params']['global_orient'][0])
        body_pose_h0        = np.array(h0['pred_mano_params']['hand_pose'])             # (15, 3, 3)
        # betas_h0            = np.array(h0['pred_mano_params']['betas'])
        if is_right_h0: 
            rh_go = global_orient_h0
            rh_pose = body_pose_h0
        else: 
            lh_go = global_orient_h0
            lh_pose = body_pose_h0
    
    with open('data/hamer/115463_1.json', 'r') as f: 
        h1                  = json.load(f)
        is_right_h1         = h1['is_right']
        global_orient_h1    = np.array(h1['pred_mano_params']['global_orient'][0])
        body_pose_h1        = np.array(h1['pred_mano_params']['hand_pose'])             # (15, 3, 3)
        # betas_h1            = np.array(h1['pred_mano_params']['betas'])
        if is_right_h1: 
            rh_go = global_orient_h1
            rh_pose = body_pose_h1
        else: 
            lh_go = global_orient_h1
            lh_pose = body_pose_h1

    # lh_go = R2aa(np.vstack(lh_go)).reshape(1, 3)
    # rh_go = R2aa(np.vstack(rh_go)).reshape(1, 3)
    # lh_go = lh_go[0]
    # rh_go = rh_go[0]
    lh_pose  = Rs2aas(lh_pose).reshape(15, 3)                                # (15, 3)
    rh_pose = Rs2aas(rh_pose).reshape(15, 3)

    C = np.diag([1, -1, -1])
    # print(lh_go.shape)
    lh_go = C @ lh_go @ C
    # rh_go = C @ rh_go @ C

    full_body_pose = np.vstack([global_orient, body_pose])
    lh_elbow_R = aa2R(full_body_pose, 18)
    rh_elbow_R = aa2R(full_body_pose, 19)

    lh_wrist_R = lh_elbow_R.T @ lh_go
    rh_wrist_R = rh_elbow_R.T @ rh_go

    lh_wrist = R2aa(lh_wrist_R)
    rh_wrist = R2aa(rh_wrist_R)

    body_pose[19] = lh_wrist
    body_pose[20] = rh_wrist

    # lh_pose[0] = lh_wrist
    # rh_pose[0] = rh_wrist

    model = smplx.create(
        model_path="models", model_type="smplx", gender="male",
        use_face_contour=False, use_pca=False, num_betas=10, num_expression_coeffs=10
    )
    print(model)

    output = model(
        global_orient   = torch.tensor(global_orient.reshape(1, 3)).float(),            # (1 × 3)
        body_pose       = torch.tensor(body_pose.reshape(1, 21 * 3)).float(),           # (1 × 21 × 3)
        left_hand_pose  = torch.tensor(lh_pose.reshape(1, 15 * 3)).float(),             # (1 × 15 × 3)
        right_hand_pose = torch.tensor(rh_pose.reshape(1, 15 * 3)).float(),             # (1 × 15 × 3)
        jaw_pose        = torch.zeros(1,3),
        left_eye_pose   = torch.zeros(1,3),
        right_eye_pose  = torch.zeros(1,3),
        expression      = torch.zeros(1,10),
        betas           = torch.zeros(1,10),
        return_vertices = True
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    geometry = [mesh]

    joints_pcl = o3d.geometry.PointCloud()
    joints_pcl.points = o3d.utility.Vector3dVector(joints)
    joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
    geometry.append(joints_pcl)

    o3d.visualization.draw_geometries(geometry)


if __name__ == '__main__':
    main()
