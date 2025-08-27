import argparse
import cv2
import glob
from multiprocessing import Pool, cpu_count
import numpy as np
import os
from pyorbbecsdk import Pipeline
from utils import frame_to_bgr_image


def _save_frame(task):
    depth_arr, color_arr, ts, rgb_dir, depth_dir = task
    if depth_arr is not None:
        cv2.imwrite(os.path.join(depth_dir, f"{ts}.png"), depth_arr)
    if color_arr is not None:
        cv2.imwrite(os.path.join(rgb_dir, f"{ts}.png"), color_arr)


def main(folder_path):
    bag_paths = glob.glob(os.path.join(folder_path, "*.bag"))
    for bag_path in bag_paths: 
        root    = os.path.dirname(bag_path)
        name, _ = os.path.splitext(os.path.basename(bag_path))
        out_dir = os.path.join(root, name)
        if os.path.isdir(out_dir): 
            print(f"Data {name} already processed. ")
            continue

        print(f"Processing {name}... ")
        rgb_dir   = os.path.join(out_dir, "rgb")
        depth_dir = os.path.join(out_dir, "depth")
        os.makedirs(out_dir,   exist_ok=True)
        os.makedirs(rgb_dir,   exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        pipeline    = Pipeline(bag_path)
        pipeline.start()
        pool        = Pool(processes=cpu_count())
        tasks       = []
        while True:
            frames = pipeline.wait_for_frames(5000)
            if frames is None:
                break

            depth       = frames.get_depth_frame()
            depth_arr   = None
            ts          = None
            if depth is not None:
                ts          = depth.get_timestamp()
                h, w        = depth.get_height(), depth.get_width()
                scale       = depth.get_depth_scale()
                depth_arr   = (np.frombuffer(depth.get_data(), np.uint16).reshape((h, w)).astype(np.float32) * scale).astype(np.uint16)

            color_arr   = None
            color       = frames.get_color_frame()
            if color is not None:
                if ts is None:
                    ts  = color.get_timestamp()
                color_arr   = frame_to_bgr_image(color)

            if ts is not None:
                tasks.append((depth_arr, color_arr, ts, rgb_dir, depth_dir))

        pipeline.stop()

        pool.map(_save_frame, tasks)
        pool.close()
        pool.join()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-p', '--folder_path', 
        type    = str, 
        default = "./20250623161011", 
        help    = "Set the root path of dataset. "
    )
    args = parser.parse_args()
    
    main(args.folder_path)
