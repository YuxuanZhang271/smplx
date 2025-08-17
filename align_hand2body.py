import glob
import json
import numpy as np
import open3d as o3d
import os
import smplx
import torch

from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R


IMG_WIDTH, IMG_HEIGHT = 1920, 1080
FOCAL_LENGTH = 37500
FPS = 30
YELLOW = "\033[93m"
RESET = "\033[0m"


def is_default_pose(x, default):
                if x is None:
                    return True
                x = np.asarray(x)
                d = np.asarray(default)
                return x.shape == d.shape and np.allclose(x, d, atol=1e-6)


def align_hand2body(root_folder, timestamp, 
                    smplx_model, mano_right, mano_left):
    """
    Returns a list[dict] of per-body params with left/right_hand_pose filled
    by the nearest MANO wrist (within match_thresh meters).
    """
    default_hand_pose = torch.zeros(1, 45, dtype=torch.float32)

    # ---------- 1) Load bodies and cache wrist positions ----------
    bodies = []
    i = 0
    while True:
        try:
            with open(os.path.join(root_folder, "body", f"{timestamp}_{i}.json")) as f:
                data = json.load(f)
        except FileNotFoundError:
            break

        cam_t = torch.tensor(data["pred_cam_t_full"][i], dtype=torch.float32).unsqueeze(0)
        go_mat = np.array(data["pred_smpl_params"]["global_orient"])
        bp_mat = np.array(data["pred_smpl_params"]["body_pose"])
        betas = torch.tensor(data["pred_smpl_params"]["betas"], dtype=torch.float32).unsqueeze(0)

        global_orient = torch.tensor(R.from_matrix(go_mat).as_rotvec(), dtype=torch.float32).unsqueeze(0)
        body_pose = torch.tensor(R.from_matrix(bp_mat).as_rotvec().reshape(-1)[:63], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = smplx_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                left_hand_pose=default_hand_pose,
                right_hand_pose=default_hand_pose,
                transl=cam_t
            )
            # Verify indices (commonly 20: left wrist, 21: right wrist)
            J = out.joints[0].cpu().numpy()
            left_wrist, right_wrist = J[20], J[21]

        bodies.append({
            "body_id": i,
            "cam_t": cam_t.tolist(),
            "global_orient": global_orient.tolist(),
            "body_pose": body_pose.tolist(),
            "betas": betas.tolist(),
            "left_hand_pose": default_hand_pose.tolist(),
            "right_hand_pose": default_hand_pose.tolist(),
            "left_wrist": left_wrist.tolist(),
            "right_wrist": right_wrist.tolist()
        })
        i += 1

    if not bodies:
        return bodies

    # ---------- 2) Load hands, mirror if needed, collect wrists & poses ----------
    S = np.diag([-1.0, 1.0, 1.0])  # mirror across X
    hand_body_pairs = []
    i = 0
    while True:
        try:
            with open(os.path.join(root_folder, "hand", f"{timestamp}_{i}.json")) as f:
                data = json.load(f)
        except FileNotFoundError:
            break

        transl = torch.tensor(data["pred_cam_t_full"][i], dtype=torch.float32).unsqueeze(0)
        go_mat = np.array(data["pred_mano_params"]["global_orient"])
        hp_mat = np.array(data["pred_mano_params"]["hand_pose"])
        betas = torch.tensor(data["pred_mano_params"]["betas"], dtype=torch.float32).unsqueeze(0)
        is_right = bool(data["is_right"])

        # If your JSON already stores true left-hand convention, remove this mirroring.
        if not is_right:
            go_mat = S @ go_mat @ S
            hp_mat = S @ hp_mat @ S

        global_orient = torch.tensor(R.from_matrix(go_mat).as_rotvec(), dtype=torch.float32)
        hand_pose = torch.tensor(R.from_matrix(hp_mat).as_rotvec().reshape(-1), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mano = mano_right if is_right else mano_left
            out = mano(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
                transl=transl
            )
            wrist_joint = out.joints[0].cpu().numpy()[0]  # MANO wrist/root
        
        for body in bodies:
            pair = {
                "hand_id": i,
                "body_id": body["body_id"],
                "is_right": is_right,
                "hand_pose": hand_pose.tolist(),
                "distance": np.linalg.norm(wrist_joint - body["right_wrist"] if is_right else body["left_wrist"])
            }
            hand_body_pairs.append(pair)
        
        i += 1
    
    hand_body_pairs = sorted(hand_body_pairs, key=lambda h: h["distance"])
    selected_hands = set()
    for pair in hand_body_pairs:
        hand_id = pair["hand_id"]
        body_id = pair["body_id"]
        is_right = pair["is_right"]
        distance = pair["distance"]
        print(f"Hand {hand_id} to Body {body_id}")
        print(f"\tIs Right Hand: {is_right}")
        print(f"\tDistance: {distance} meters")

        if hand_id in selected_hands:
            print(f"{YELLOW}\tHand {hand_id} already selected, skipping.{RESET}")
            continue
        hand_pose = "right_hand_pose" if is_right else "left_hand_pose"
        if not is_default_pose(bodies[body_id][hand_pose], default_hand_pose):
            print(f"{YELLOW}\tBody {body_id}'s {hand_pose} already set, skipping.{RESET}")
            continue
        bodies[body_id][hand_pose] = pair["hand_pose"]
        selected_hands.add(hand_id)
        print(f"{YELLOW}\tAssigned hand {hand_id} to body {body_id}'s {hand_pose}.{RESET}")
    
    return bodies


def main():
    smplx_model = smplx.create(
        model_path="models",
        model_type="smplx",
        gender="male",
        use_pca=False,
        ext="pkl",
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
    mano_right = smplx.create(
        model_path="models",
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        num_pca_comps=30,
        create_global_orient=True,
        create_hand_pose=True,
        create_transl=True,
        ext="pkl"
    )
    mano_left = smplx.create(
        model_path="models",
        model_type="mano",
        is_rhand=False,
        use_pca=False,
        num_pca_comps=30,
        create_global_orient=True,
        create_hand_pose=True,
        create_transl=True,
        ext="pkl"
    )

    root_folder = "device_0_record_2"
    # timestamps = [os.path.basename(p).split("_")[0] for p in sorted(glob.glob(os.path.join(root_folder, "body", "*_0.json")))]
    timestamps = ["669382"]
    print(timestamps)
    frames = []
    meshes = []
    faces = np.asarray(smplx_model.faces)
    idx = 0
    for timestamp in timestamps: 
        frame = align_hand2body(root_folder, timestamp, smplx_model, mano_right, mano_left)
        # print(frame)
        frames.append(frame)

        for body in frame: 
            cam_t           = torch.tensor(body['cam_t'], dtype=torch.float32)
            global_orient   = torch.tensor(body["global_orient"], dtype=torch.float32)
            body_pose       = torch.tensor(body["body_pose"], dtype=torch.float32)
            betas           = torch.tensor(body["betas"], dtype=torch.float32)
            left_hand_pose  = torch.tensor(body["left_hand_pose"], dtype=torch.float32)
            right_hand_pose = torch.tensor(body["right_hand_pose"], dtype=torch.float32)
            with torch.no_grad():
                output = smplx_model(
                    global_orient   = global_orient, 
                    body_pose       = body_pose, 
                    betas           = betas, 
                    left_hand_pose  = left_hand_pose, 
                    right_hand_pose = right_hand_pose, 
                    transl          = cam_t
                )
                vertice = output.vertices.detach().cpu().numpy().squeeze(0)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertice)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.8, 0.8, 0.8])
                meshes.append(mesh)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        meshes.append(axis)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width   = IMG_WIDTH, 
            height  = IMG_HEIGHT,
            fx      = FOCAL_LENGTH, 
            fy      = FOCAL_LENGTH,
            cx      = IMG_WIDTH/2, 
            cy      = IMG_HEIGHT/2
        )
        extrinsic = np.eye(4)
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsic
        cam_params.extrinsic = extrinsic

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='MANO2SMPLX', width=IMG_WIDTH, height=IMG_HEIGHT)
        for mesh in meshes:
            vis.add_geometry(mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        vis.run()
        vis.destroy_window()

        idx += 1

    dataset = {
        "name": root_folder, 
        "focal_length": 37500.0, 
        "frames": frames
    }

    with open(os.path.join(root_folder, "dataset.json"), "w") as f: 
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
