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

sys.path.append("./EasyMocap")
from myeasymocap.operations.triangulate import SimpleTriangulate


# -------------------- Arguments -------------------- #
parser = argparse.ArgumentParser(description='AlphaPose Keypoints Parser')
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--all_frames", default=False, action="store_true")
parser.add_argument("--easymocap", default=False, action="store_true", help='use Easymocap for triangulation')
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
parser.add_argument("--ignore_missing_tip", action="store_true", help="Should a missing fingertip be allowed")
args = parser.parse_args()


base_path = os.path.join(args.root_dir)
image_base = os.path.join(base_path, args.seq_path)
output_path = args.out_dir
# Loads the camera parameters
if args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"

params_path = os.path.join(output_path, params_txt)

params = param_utils.read_params(params_path)
cam_names = list(params[:]["cam_name"])
cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam)
for cam in cams_to_remove:
    if cam in cam_names:
        cam_names.remove(cam)

keypoints2d_dir_right = os.path.join(output_path, "keypoints_2d", "right", str(args.ith).zfill(3))
keypoints2d_dir_left = os.path.join(output_path, "keypoints_2d", "left",  str(args.ith).zfill(3))

cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)
intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cam_names, cam_mapper, easymocap_format=True)
print("Total Views:", len(cam_mapper.keys()))


# Get files to process
reader = Reader(args.input_type, image_base, ith=args.ith)
print("Total frames",reader.frame_count)


keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(args.ith).zfill(3))
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
for cam in tqdm(cam_names, total=len(cam_names)):
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
        all_keypoints2d_left.append(np.asarray(keypoints2d_left))
        all_keypoints2d_right.append(np.asarray(keypoints2d_right))
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
            if not args.easymocap:
                keypoints3d_left, residuals = triangulate_joints(np.asarray(keypoints2d_left), np.asarray(projs), processor=ransac_processor, residual_threshold=10, min_samples=5)
                print(f"Error: {residuals.mean()}")
                keypoints3d_right, residuals = triangulate_joints(np.asarray(keypoints2d_right), np.asarray(projs), processor=ransac_processor, residual_threshold=10, min_samples=5)
                print(f"Error: {residuals.mean()}")
            else:
                triangulation = SimpleTriangulate("iterative")
                keypoints3d_left = triangulation(np.asarray(keypoints2d_left), cameras)['keypoints3d']
                keypoints3d_right = triangulation(np.asarray(keypoints2d_right), cameras)['keypoints3d']
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
