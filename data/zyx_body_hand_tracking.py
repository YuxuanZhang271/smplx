import argparse
import cv2
import json
import numpy as np
import os
from pathlib import Path
import torch
from typing import Dict, Optional

from human.hmr2.configs import CACHE_DIR_4DHUMANS
from human.hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from human.hmr2.utils import recursive_to
from human.hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from human.hmr2.utils.renderer import Renderer, cam_crop_to_full
from human.hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2 import model_zoo
from detectron2.config import get_cfg

from hamer.hamer.configs import CACHE_DIR_HAMER
from hamer.hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.hamer.utils import recursive_to
from hamer.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.vitpose_model import ViTPoseModel

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('-i', '--img_folder', type=str, default='example_data/images')
    parser.add_argument('-o', '--out_folder', type=str, default='demo_out')
    parser.add_argument('-b', '--batch_size', type=int, default=48)

    args = parser.parse_args()

    # load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # load hmr2 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # load detector & renderer
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    subfolder_path = os.path.join(args.out_folder, Path(args.img_folder).parent.name)
    os.makedirs(subfolder_path, exist_ok=False)
    print("subfolder: ", subfolder_path)

    # get all images
    img_paths = [img for end in ['*.jpg', '*.png'] for img in Path(args.img_folder).glob(end)]
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # run hmr2 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            # print("Model: ", out.keys())

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                # add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # save param json
                output_params = {}
                # smpl params
                pred_smpl_params = {}
                for subk, subv in out['pred_smpl_params'].items():
                    pred_smpl_params[subk] = subv[n].detach().cpu().numpy().tolist()
                output_params['pred_smpl_params'] = pred_smpl_params
                # scaled focal length
                output_params["scaled_focal_length"] = float(scaled_focal_length)
                # pred cam t
                output_params["pred_cam_t_full"] = pred_cam_t_full.tolist()

                json_path = os.path.join(subfolder_path, f'{img_fn}_{person_id}.json')
                with open(json_path, 'w') as f:
                    json.dump(output_params, f, indent=4)

        # save full frame image
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(subfolder_path, f'{img_fn}_full_frame.png'), 255 * input_img_overlay[:, :, ::-1])


if __name__ == '__main__':
    main()
