import argparse
import cv2
import datetime as dt
import json
import numpy as np
import os
from queue import Queue
from typing import List

from pyorbbecsdk import *
from utils import frame_to_bgr_image


MAX_DEVICES     = 2
MAX_QUEUE_SIZE  = 5
ESC_KEY         = 27

curr_device_cnt = 0
stop_rendering  = False

color_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
depth_frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
has_color_sensor:   List[bool]  = [False for _ in range(MAX_DEVICES)]


def on_new_frame_callback(frames: FrameSet, index: int):
    global color_frames_queue, depth_frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is not None:
        if color_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            color_frames_queue[index].get()
        color_frames_queue[index].put(color_frame)
    if depth_frame is not None:
        if depth_frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            depth_frames_queue[index].get()
        depth_frames_queue[index].put(depth_frame)


def rendering_frames(pipelines: List[Pipeline], time):
    global color_frames_queue, depth_frames_queue, curr_device_cnt, stop_rendering
    is_recording    = False
    idx             = 0

    while not stop_rendering:
        for i in range(curr_device_cnt):
            color_frame = None
            depth_frame = None
            if not color_frames_queue[i].empty():
                color_frame = color_frames_queue[i].get()
            if not depth_frames_queue[i].empty():
                depth_frame = depth_frames_queue[i].get()
            if color_frame is None and depth_frame is None:
                continue

            color_image, depth_image    = None, None
            color_width, color_height   = 0,    0
            if color_frame is not None:
                color_width, color_height = color_frame.get_width(), color_frame.get_height()
                color_image = frame_to_bgr_image(color_frame)
            if depth_frame is not None:
                width           = depth_frame.get_width()
                height          = depth_frame.get_height()
                scale           = depth_frame.get_depth_scale()
                depth_format    = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue

                try:
                    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    depth_data = depth_data.reshape((height, width))
                except ValueError:
                    print("Failed to reshape depth data")
                    continue

                depth_data  = depth_data.astype(np.float32) * scale
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            if color_image is not None and depth_image is not None:
                window_size = (color_width // 2, color_height // 2)
                color_image = cv2.resize(color_image, window_size)
                depth_image = cv2.resize(depth_image, window_size)
                image       = np.hstack((color_image, depth_image))
            elif depth_image is not None and not has_color_sensor[i]:
                image       = depth_image
            else:
                continue

            cv2.imshow("Device {}".format(i), image)
            key = cv2.waitKey(1)
            if key == ord("r"): 
                is_recording = not is_recording
                if is_recording: 
                    index = 0
                    for pipeline in pipelines:
                        print("Starting device {} recording...".format(index))
                        pipeline.start_recording(os.path.join("records", time, f"device_{index}_record_{idx}.bag"))
                        index += 1
                else: 
                    index = 0
                    for pipeline in pipelines:
                        print("Stoping device {} recording...".format(index))
                        pipeline.stop_recording()
                        index += 1
                    idx += 1
            elif key == ESC_KEY:
                stop_rendering = True
                break
    cv2.destroyAllWindows()


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print("Starting device {}".format(index))
        pipeline.start(
            config, 
            lambda frame_set, 
            curr_index = index: on_new_frame_callback(frame_set, curr_index)
        )
        index += 1


def stop_streams(pipelines: List[Pipeline]):
    for pipeline in pipelines:
        pipeline.stop_recording()
        pipeline.stop()


def main():
    global curr_device_cnt, has_color_sensor, stop_rendering
    ctx             = Context()
    device_list     = ctx.query_devices()
    curr_device_cnt = device_list.get_count()
    if curr_device_cnt == 0:
        print("No device connected")
        return
    if curr_device_cnt > MAX_DEVICES:
        print("Too many devices connected")
        return
    
    pipelines:  List[Pipeline]  = []
    configs:    List[Config]    = []
    for i in range(device_list.get_count()):
        device      = device_list.get_device_by_index(i)
        pipeline    = Pipeline(device)
        config      = Config()
        try:
            profile_list        = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile       = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
            config.enable_stream(color_profile)
            has_color_sensor[i] = True

            color_intrinsics    = color_profile.get_intrinsic()
            print(color_intrinsics)
            K = np.array([
                [color_intrinsics.fx,   0,                      color_intrinsics.cx],
                [0,                     color_intrinsics.fy,    color_intrinsics.cy],
                [0,                     0,                      1]
            ])
        except OBError as e:
            print(e)
            has_color_sensor[i] = False
        
        profile_list    = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile   = profile_list.get_video_stream_profile(640, 576, OBFormat.Y16, 30)
        config.enable_stream(depth_profile)

        pipelines.append(pipeline)
        configs.append(config)

    time = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs("records", exist_ok=True)
    os.makedirs(os.path.join("records", time), exist_ok=True)
    np.savetxt(os.path.join("records", time, "cam_K.txt"), K, fmt="%.18e", delimiter=" ")

    cam_param   = {
        "cam_K":        K, 
        "depth_scale":  1.0
    }
    with open(os.path.join("records", time, "camera.json"), "w") as f:
        json.dump(cam_param, f, indent=4)

    start_streams(pipelines, configs)
    try:
        rendering_frames(pipelines, time)
    except KeyboardInterrupt:
        stop_rendering = True
    finally:
        stop_streams(pipelines)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
