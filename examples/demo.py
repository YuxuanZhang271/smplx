import os.path as osp
import argparse

import numpy as np
import torch

import smplx
import open3d as o3d
import json


def main(sample_shape=True,
         sample_expression=True,):

    model = smplx.create('models', model_type='smplx', gender='male', ext='npz', 
                         use_face_contour=False,
                         num_betas=10,
                         num_expression_coeffs=10)
    print(model)

    with open('data/human/115463_0.json', 'r') as f: 
        body                = json.load(f)
        global_orient       = body['pred_smpl_params']['global_orient']
        body_pose           = body['pred_smpl_params']['body_pose']  # (23, 3, 3)
        betas               = body['pred_smpl_params']['betas']
    
    with open('data/hamer/115463_0.json', 'r') as f: 
        h0                  = json.load(f)
        is_right_h0         = h0['is_right']
        global_orient_h0    = h0['pred_mano_params']['global_orient']
        body_pose_h0        = h0['pred_mano_params']['hand_pose']  # (15, 3, 3)
        betas_h0            = h0['pred_mano_params']['betas']

    with open('data/hamer/115463_1.json', 'r') as f: 
        h1                  = json.load(f)
        is_right_h1         = h1['is_right']
        global_orient_h1    = h1['pred_mano_params']['global_orient']
        body_pose_h1        = h1['pred_mano_params']['hand_pose']  # (15, 3, 3)
        betas_h1            = h1['pred_mano_params']['betas']

    betas, expression = None, None
    if sample_shape:
        betas = torch.randn([1, model.num_betas], dtype=torch.float32)
    if sample_expression:
        expression = torch.randn(
            [1, model.num_expression_coeffs], dtype=torch.float32)

    output = model(betas=betas, expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(model.faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.3, 0.3, 0.3])

    geometry = [mesh]

    joints_pcl = o3d.geometry.PointCloud()
    joints_pcl.points = o3d.utility.Vector3dVector(joints)
    joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
    geometry.append(joints_pcl)

    o3d.visualization.draw_geometries(geometry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')
    parser.add_argument('--sample-shape', default=True, dest='sample_shape',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random shape')
    parser.add_argument('--sample-expression', default=True, dest='sample_expression',
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Sample a random expression')
    args = parser.parse_args()

    sample_shape = args.sample_shape
    sample_expression = args.sample_expression

    main(sample_shape=sample_shape,
         sample_expression=sample_expression)
