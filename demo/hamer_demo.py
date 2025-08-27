import argparse
import cv2
import json
import numpy as np
import os
from pathlib import Path
import torch

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

from detectron2 import model_zoo
from detectron2.config import get_cfg

from vitpose_model import ViTPoseModel


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main():
    parser  = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument("-i", "--img_folder", type=str, default="images",   help="Folder with input images")
    parser.add_argument("-o", "--out_folder", type=str, default="out_demo", help="Output folder to save rendered results")
    parser.add_argument("-b", "--batch_size", type=int, default=48,         help="Batch size for inference/fitting")
    args    = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)

    # Setup HaMeR model
    device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model   = model.to(device)
    model.eval()

    # Load detector
    detectron2_cfg  = model_zoo.get_config("new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh  = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh    = 0.4
    detector        = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)
    subfolder_path = os.path.join(args.out_folder, Path(args.img_folder).parent.name)
    os.makedirs(subfolder_path, exist_ok=False)
    print("subfolder: ", subfolder_path)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in ["*.jpg", "*.png"] for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img     = img_cv2.copy()[:, :, ::-1]

        det_instances   = det_out["instances"]
        valid_idx       = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes     = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores     = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

        bboxes      = []
        is_right    = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp  = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            # Rejecting not confident detections
            keyp    = left_hand_keyp
            valid   = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(), 
                    keyp[valid, 1].min(), 
                    keyp[valid, 0].max(), 
                    keyp[valid, 1].max()
                ]
                bboxes.append(bbox)
                is_right.append(0)

            keyp    = right_hand_keyp
            valid   = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [
                    keyp[valid, 0].min(), 
                    keyp[valid, 1].min(), 
                    keyp[valid, 0].max(), 
                    keyp[valid, 1].max()
                ]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset     = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
                # print("HaMeR output keys:", list(out.keys()))

            multiplier          = 2 * batch["right"] - 1
            pred_cam            = out["pred_cam"]
            pred_cam[:, 1]      = multiplier * pred_cam[:, 1]
            box_center          = batch["box_center"].float()
            box_size            = batch["box_size"].float()
            img_size            = batch["img_size"].float()
            multiplier          = 2 * batch["right"] - 1
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _   = os.path.splitext(os.path.basename(img_path))
                person_id   = int(batch["personid"][n])
                input_patch = batch["img"][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                # Add all verts and cams to list
                verts       = out["pred_vertices"][n].detach().cpu().numpy()
                is_right    = batch["right"][n].cpu().numpy()
                verts[:,0]  = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                output_params = {}
                if isinstance(out["pred_mano_params"], dict): 
                    person_mano = { 
                        key: value[n].detach().cpu().numpy().tolist() 
                        for key, value in out["pred_mano_params"].items() 
                    } 
                else: 
                    person_mano = out["pred_mano_params"][n].detach().cpu().numpy().tolist()

                output_params["pred_mano_params"]       = person_mano
                output_params["scaled_focal_length"]    = float(scaled_focal_length)
                output_params["pred_cam_t_full"]        = pred_cam_t_full.tolist()
                output_params["is_right"]               = bool(is_right)

                json_path = os.path.join(subfolder_path, f"{img_fn}_{person_id}.json")
                with open(json_path, "w") as f:
                    json.dump(output_params, f, indent=4)

        # Visualization for validation
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color = LIGHT_BLUE,
                scene_bg_color  = (1, 1, 1),
                focal_length    = scaled_focal_length
            )
            cam_view            = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            input_img           = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
            input_img           = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)    # Add alpha channel
            input_img_overlay   = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:,:,3:]
            cv2.imwrite(os.path.join(subfolder_path, f"{img_fn}_all.jpg"), 255 * input_img_overlay[:, :, ::-1])


if __name__ == "__main__":
    main()
