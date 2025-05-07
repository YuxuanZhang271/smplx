import argparse
import cv2
import json
import numpy as np
import open3d as o3d
import smplx
import torch


def rotmats_to_axisangle(rot_mats: np.ndarray) -> np.ndarray:
    aa_list = []
    for R in rot_mats:
        rvec, _ = cv2.Rodrigues(R)   # rvec.shape == (3,1)
        aa_list.append(rvec.flatten())
    return np.stack(aa_list, axis=0)  # (K,3)


def main():
    model = smplx.create('models', 
                         model_type             = 'smplx', 
                         gender                 = 'male', 
                         ext                    = 'npz', 
                         use_pca                = False,
                         use_face_contour       = False,
                         num_betas              = 10,
                         num_expression_coeffs  = 10)
    print(model)

    with open('data/human/115463_0.json', 'r') as f: 
        body                = json.load(f)
        global_orient       = np.array(body['pred_smpl_params']['global_orient'])
        body_pose           = np.array(body['pred_smpl_params']['body_pose'])[:21]  # (23, 3, 3) -> (21, 3, 3)
        betas               = np.array(body['pred_smpl_params']['betas'])

        global_orient = rotmats_to_axisangle(global_orient).reshape(1, 3)
        body_pose = rotmats_to_axisangle(body_pose).reshape(1, 21 * 3)
    
    left_hand, right_hand = [], []
    with open('data/hamer/115463_0.json', 'r') as f: 
        h0                  = json.load(f)
        is_right_h0         = h0['is_right']
        global_orient_h0    = np.array(h0['pred_mano_params']['global_orient'])
        body_pose_h0        = np.array(h0['pred_mano_params']['hand_pose'])  # (15, 3, 3)
        betas_h0            = np.array(h0['pred_mano_params']['betas'])
        if is_right_h0: 
            right_hand.append(body_pose_h0)
        else: 
            left_hand.append(body_pose_h0)
    with open('data/hamer/115463_1.json', 'r') as f: 
        h1                  = json.load(f)
        is_right_h1         = h1['is_right']
        global_orient_h1    = np.array(h1['pred_mano_params']['global_orient'])
        body_pose_h1        = np.array(h1['pred_mano_params']['hand_pose'])  # (15, 3, 3)
        betas_h1            = np.array(h1['pred_mano_params']['betas'])
        if is_right_h1: 
            right_hand.append(body_pose_h1)
        else: 
            left_hand.append(body_pose_h1)
    if len(left_hand)==0:
        left_hand = [np.eye(3)[None].repeat(15,axis=0)]
    if len(right_hand)==0:
        right_hand = [np.eye(3)[None].repeat(15,axis=0)]
    left_hand  = rotmats_to_axisangle(np.vstack(left_hand)).reshape(1, 15 * 3)
    right_hand = rotmats_to_axisangle(np.vstack(right_hand)).reshape(1, 15 * 3)

    expression = np.zeros((1, model.num_expression_coeffs), dtype=np.float32)

    t_global_orient = torch.from_numpy(global_orient).float()
    t_body_pose     = torch.from_numpy(body_pose).float()
    t_betas         = torch.from_numpy(betas).float().unsqueeze(0)  
    t_expr          = torch.from_numpy(expression).float()
    t_left_hand     = torch.from_numpy(left_hand).float()
    t_right_hand    = torch.from_numpy(right_hand).float()

    output = model(
        betas           = t_betas,          # shape (1,10)
        expression      = t_expr,           # shape (1,10)
        global_orient   = t_global_orient,  # shape (1,3)
        body_pose       = t_body_pose,      # shape (1,69)
        left_hand_pose  = t_left_hand,      # shape (1,45)
        right_hand_pose = t_right_hand,     # shape (1,45)
        return_verts    = True
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
