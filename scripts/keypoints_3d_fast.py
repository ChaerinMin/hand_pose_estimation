import os
import sys
import cv2
import ujson
import torch
import shutil
import argparse
import tempfile
import platform
import numpy as np
# import xml.etree.cElementTree as ET

from tqdm import tqdm
from glob import glob
from natsort import natsorted

sys.path.append(".")
from src.utils.reader_v2 import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.fingers import FINGER_IDX, TIP_IDX
from src.triangulate import triangulate_joints, ransac_processor
from src.utils.filter import apply_one_euro_filter_3d

sys.path.append("./EasyMocap")
from myeasymocap.operations.triangulate import SimpleTriangulate


# -------------------- Arguments -------------------- #
parser = argparse.ArgumentParser(description='AlphaPose Keypoints Parser')
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--to_smooth", action="store_true", help="Whether to temporally smoothing the result")
parser.add_argument("--all_frames", default=False, action="store_true")
parser.add_argument("--easymocap", default=False, action="store_true", help='use Easymocap for triangulation')
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
parser.add_argument("--ignore_missing_tip", action="store_true", help="Should a missing fingertip be allowed")
args = parser.parse_args()


base_path = os.path.join(args.root_dir)
image_base = os.path.join(base_path, args.seq_path)
# image_base = os.path.join(base_path)
output_path = args.out_dir
# Loads the camera parameters
if args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"

params_path = os.path.join(output_path, params_txt)

if "stage1" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=True)
elif "stage2" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=False)
else:
    raise ValueError("Cannot determine whether to assume undistorted.")
cam_names = list(params[:]["cam_name"])
removed_camera_path = os.path.join(output_path, 'ignore_camera.txt')
if os.path.isfile(removed_camera_path):
    with open(removed_camera_path) as file:
        ignored_cameras = [line.rstrip() for line in file]
else:
    ignored_cameras = None
cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras)
for cam in cams_to_remove:
    if cam in cam_names:
        cam_names.remove(cam)

if args.ith == -1:
    total_video_idxs = 0
    max_folder_id = 0
    for fid, folder in enumerate(os.listdir(image_base)):
        if 'cam' in folder and folder not in cams_to_remove:
            length = len([file for file in os.listdir(os.path.join(image_base, folder)) if file.endswith('.mp4')])
            if length > total_video_idxs:
                total_video_idxs = length
                max_folder_id = fid
                anchor_camera_by_length = [p for p in os.listdir(image_base) if 'cam' in p][fid]
    # folder0 = os.listdir(image_base)[0]
    # folder0_path = os.path.join(image_base, folder0)
    # total_video_idxs = len(os.listdir(folder0_path))//2
    # anchor_camera_by_length = "brics-odroid-002_cam0"
    if args.start > 0:
        if args.end > 0:
            selected_vid_idxs = list(range(args.start, args.end))
        else:
            selected_vid_idxs = list(range(args.start, total_video_idxs))
    else:
        if args.end > 0:
            selected_vid_idxs = list(range(args.end))
        else:
            selected_vid_idxs = list(range(total_video_idxs))
else:       
    selected_vid_idxs = [args.ith]
    
for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')
    
    keypoints2d_dir_right = os.path.join(output_path, "keypoints_2d", "right", str(selected_vid_idx).zfill(3))
    keypoints2d_dir_left = os.path.join(output_path, "keypoints_2d", "left",  str(selected_vid_idx).zfill(3))

    cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)

    # Get files to process
    reader = Reader(args.input_type, image_base, cam_names=cam_names, cams_to_remove=cams_to_remove, ith=selected_vid_idx, anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera)
    if reader.frame_count <= 0:
        continue
        
    extra_cams_to_remove = reader.to_delete
    cur_cam_names = cam_names.copy()
    for cam in extra_cams_to_remove:
        if cam in cur_cam_names:
            cur_cam_names.remove(cam)
    print("Total Views:", len(cur_cam_names))
    print("Total frames", reader.frame_count)
    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cur_cam_names, cam_mapper, easymocap_format=True)
    
    keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(selected_vid_idx).zfill(3))
    try:
        shutil.rmtree(keypoints3d_dir)
    except FileNotFoundError:
        pass
    os.makedirs(keypoints3d_dir)


    if (args.all_frames):
        chosen_frames = range(0, reader.frame_count, 1)
    else:
        chosen_frames = range(args.start, args.end, args.stride)

    all_keypoints2d_left = []
    all_keypoints2d_right = []
    for cam in cur_cam_names:
        if cam in cam_mapper:
            keypoints2d_left = []
            keypoints2d_right = []
            ap_keypoints_path_left = os.path.join(keypoints2d_dir_left, f"{cam_mapper[cam]}.jsonl")
            ap_keypoints_path_right = os.path.join(keypoints2d_dir_right, f"{cam_mapper[cam]}.jsonl")
            with open(ap_keypoints_path_left, "r") as fl, open(ap_keypoints_path_right, "r") as fr:
                for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
                    if l_idx in chosen_frames:
                        keypoints2d_left.append(np.array(ujson.loads(linel)).reshape(-1, 3))
                        keypoints2d_right.append(np.array(ujson.loads(liner)).reshape(-1, 3))
            keypoints2d_left = np.asarray(keypoints2d_left)
            keypoints2d_right = np.asarray(keypoints2d_right)
            valid_left = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        (keypoints2d_left[:, :, 0] == 0).all(axis=1),
                        (keypoints2d_left[:, :, 1] == 0).all(axis=1)),
                    (keypoints2d_left[:, :, 2] == 1).all(axis=1)
                )
            )
            valid_right = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        (keypoints2d_right[:, :, 0] == 0).all(axis=1),
                        (keypoints2d_right[:, :, 1] == 0).all(axis=1)),
                    (keypoints2d_right[:, :, 2] == 1).all(axis=1)
                )
            )
            if args.to_smooth:
                if keypoints2d_left[:2][valid_left[:2]].shape[0] > 1:
                    keypoints2d_left[:2][valid_left[:2]] = apply_one_euro_filter_3d(keypoints2d_left[:2][valid_left[:2]], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
                if keypoints2d_right[:2][valid_right[:2]].shape[0] > 1:
                    keypoints2d_right[:2][valid_right[:2]] = apply_one_euro_filter_3d(keypoints2d_right[:2][valid_right[:2]], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            all_keypoints2d_left.append(keypoints2d_left)
            all_keypoints2d_right.append(keypoints2d_right)
    all_keypoints2d_left = np.asarray(all_keypoints2d_left)
    all_keypoints2d_right = np.asarray(all_keypoints2d_right)


    keypt_file_left = os.path.join(keypoints3d_dir, "left.jsonl")
    keypt_file_right = os.path.join(keypoints3d_dir, "right.jsonl")
    chosen_frames_left = []
    chosen_frames_right = []
    print(f"Writing 3D keypoints to {keypt_file_left}")
    print(f"Writing 3D keypoints to {keypt_file_right}")
    with open(keypt_file_left, "w") as fl, open(keypt_file_right, "w") as fr:
        for l_idx in tqdm(range(reader.frame_count), total=reader.frame_count):
            if l_idx in chosen_frames:
                keypoints2d_left = all_keypoints2d_left[:, l_idx, :, :]
                keypoints2d_right = all_keypoints2d_right[:, l_idx, :, :]
                valid_left = np.logical_not(
                    np.logical_and(
                        np.logical_and(
                            (keypoints2d_left[:, :, 0] == 0).all(axis=1),
                            (keypoints2d_left[:, :, 1] == 0).all(axis=1)),
                        (keypoints2d_left[:, :, 2] == 1).all(axis=1)
                    )
                )
                valid_right = np.logical_not(
                    np.logical_and(
                        np.logical_and(
                            (keypoints2d_right[:, :, 0] == 0).all(axis=1),
                            (keypoints2d_right[:, :, 1] == 0).all(axis=1)),
                        (keypoints2d_right[:, :, 2] == 1).all(axis=1)
                    )
                )
                if not args.easymocap:
                    keypoints3d_left, residuals = triangulate_joints(np.asarray(keypoints2d_left)[valid_left], np.asarray(projs)[valid_left], processor=ransac_processor, residual_threshold=10, min_samples=5)
                    print(f"Error: {residuals.mean()}")
                    keypoints3d_right, residuals = triangulate_joints(np.asarray(keypoints2d_right)[valid_right], np.asarray(projs)[valid_right], processor=ransac_processor, residual_threshold=10, min_samples=5)
                    print(f"Error: {residuals.mean()}")
                else:
                    triangulation = SimpleTriangulate("iterative")
                    valid_cameras = {}
                    for k_cam in cameras:
                        valid_cameras[k_cam] = cameras[k_cam][valid_left]
                    keypoints3d_left = triangulation(np.asarray(keypoints2d_left)[valid_left], valid_cameras)['keypoints3d']
                    valid_cameras = {}
                    for k_cam in cameras:
                        valid_cameras[k_cam] = cameras[k_cam][valid_right]
                    keypoints3d_right = triangulation(np.asarray(keypoints2d_right)[valid_right], valid_cameras)['keypoints3d']
                ujson.dump(keypoints3d_left.tolist(), fl)
                fl.write('\n')
                ujson.dump(keypoints3d_right.tolist(), fr)
                fr.write('\n')
            else:
                ujson.dump(np.zeros((21,4)).tolist(), fl)
                fl.write('\n')
                ujson.dump(np.zeros((21,4)).tolist(), fr)
                fr.write('\n')
            
            to_use_left = np.ones(1, dtype=bool)
            to_use_right = np.ones(1, dtype=bool)
            
            # Remove frames which have complete finger missing
            for idx in FINGER_IDX:
                to_use_left = np.logical_and(to_use_left, np.any(keypoints3d_left[idx,3], axis=0))
                to_use_right = np.logical_and(to_use_right, np.any(keypoints3d_right[idx,3], axis=0))
            
            # Remove frames which have any of the finger tips missing
            if not args.ignore_missing_tip:
                to_use_left = np.logical_and(to_use_left, np.all(keypoints3d_left[TIP_IDX,3], axis=0))
                to_use_right = np.logical_and(to_use_right, np.all(keypoints3d_right[TIP_IDX,3], axis=0))

            if np.any(to_use_left):
                chosen_frames_left.append(l_idx)
            if np.any(to_use_right):
                chosen_frames_right.append(l_idx)       
            
    chosen_path_left = os.path.join(keypoints3d_dir, f"chosen_frames_left.json")
    chosen_path_right = os.path.join(keypoints3d_dir, f"chosen_frames_right.json")
    with open(chosen_path_left, "w") as f:
        ujson.dump(chosen_frames_left, f, indent=2)
    with open(chosen_path_right, "w") as f:
        ujson.dump(chosen_frames_right, f, indent=2)
