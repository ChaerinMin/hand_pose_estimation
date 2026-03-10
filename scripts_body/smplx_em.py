"""
SMPL-X fitting from 3D keypoints using EasyMocap.

This script fits SMPL-X body model to COCO-WholeBody 133 keypoints.
It supports:
- Full body fitting (body + hands + face + feet)
- Optional exclusion of feet joints
- Shape and pose optimization
- Temporal smoothing
- Visualization and mesh export

Input:
  - keypoints_3d/{vid_idx:03d}/wholebody.jsonl: 3D keypoints (133, 4)
  - keypoints_2d/{vid_idx:03d}/{cam_name}.jsonl: 2D keypoints (133*3,)

Output:
  - params/{vid_idx:03d}.json: SMPL-X parameters
  - Optional: mano/, repro_2d/, repro_3d/, meshes/ visualization
"""

import argparse
import glob
import json
import os
import sys
import cv2
import math

import numpy as np
import ujson
from tqdm import tqdm
import trimesh

import src.utils.params as param_utils
from src.utils.cameras import (get_projections, map_camera_names,
                               removed_cameras)
from src.utils.easymocap_utils import (load_model, projectN3, vis_repro,
                                       vis_smpl)
from src.utils.filter import apply_one_euro_filter_2d, apply_one_euro_filter_3d
from src.utils.parser import add_common_args
from src.utils.reader_v2 import Reader
from src.utils.video_handler import convert_video_ffmpeg, create_video_writer
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.pipeline import smpl_from_keypoints3d, smpl_from_keypoints3d2d
from easymocap.smplmodel import select_nf
from easymocap.smplmodel.body_model import SMPLlayer

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.system("module load ffmpeg")
sys.path.append(".")
sys.path.append("./third-party/EasyMocap")


# COCO-WholeBody to SMPL-X joint mapping
# COCO-WholeBody: 133 keypoints (0-16: body, 17-22: feet, 23-90: face, 91-111: left hand, 112-132: right hand)
# We'll use the body keypoints primarily for fitting
COCO_WHOLEBODY_TO_SMPLX = {
    'body': list(range(0, 17)),      # COCO body joints
    'feet': list(range(17, 23)),     # COCO feet joints
    'face': list(range(23, 91)),     # COCO face landmarks
    'lhand': list(range(91, 112)),   # Left hand
    'rhand': list(range(112, 133)),  # Right hand
}

# COCO17 to Body25 mapping (from EasyMocap)
# This maps COCO17[i] -> Body25[COCO17_IN_BODY25[i]]
COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def get_bodyhand_config(exclude_feet=False):
    """Get EasyMocap's bodyhand config for Body25+Hands (67 joints).

    Uses EasyMocap's predefined bodyhand format with:
    - Body25: 0-24 (25 joints)
    - Index 25: unused
    - Left hand: 26-45 (20 joints, no wrist root)
    - Index 46: unused
    - Right hand: 47-66 (20 joints, no wrist root)

    Args:
        exclude_feet: If True, exclude feet keypoints from fitting

    Returns:
        dict: Configuration for keypoint fitting
    """
    from easymocap.dataset import CONFIG

    # Get the bodyhand config from EasyMocap
    config = CONFIG['bodyhand'].copy()

    # If excluding feet, we need to filter out feet-related kintree entries
    if exclude_feet:
        # Remove kintree entries involving feet joints (19-24 in Body25)
        feet_joints = set(range(19, 25))
        filtered_kintree = []
        for edge in config['kintree']:
            if edge[0] not in feet_joints and edge[1] not in feet_joints:
                filtered_kintree.append(edge)
        config['kintree'] = filtered_kintree

    return config


def coco_wholebody_to_body25_hands(keypoints):
    """Convert COCO-WholeBody 133 keypoints to EasyMocap bodyhand format (67 joints).

    EasyMocap bodyhand format:
    - Body25: 0-24 (25 joints in OpenPose Body25 ordering)
    - Index 25: unused (set to zero)
    - Left hand: 26-45 (20 joints, MANO 1-20, no wrist root)
    - Index 46: unused (set to zero)
    - Right hand: 47-66 (20 joints, MANO 1-20, no wrist root)

    Args:
        keypoints: (n_frames, 133, 4) or (n_frames, n_views, 133, 3) array
            COCO-WholeBody format with (x, y, z, conf) or (x, y, conf)

    Returns:
        Converted keypoints with shape (n_frames, 67, 4) or (n_frames, n_views, 67, 3)
    """
    is_2d = (keypoints.ndim == 3 and keypoints.shape[-1] == 3)
    is_3d = (keypoints.ndim == 3 and keypoints.shape[-1] == 4)
    is_multi_view = (keypoints.ndim == 4)

    if is_multi_view:
        # (n_frames, n_views, 133, 3) -> process each view
        n_frames, n_views = keypoints.shape[:2]
        kpts_out = np.zeros((n_frames, n_views, 67, 3))
        for v in range(n_views):
            kpts_out[:, v, :, :] = coco_wholebody_to_body25_hands(keypoints[:, v, :, :])
        return kpts_out

    n_frames = keypoints.shape[0]
    n_coords = keypoints.shape[-1]  # 3 for 2D, 4 for 3D
    kpts_out = np.zeros((n_frames, 67, n_coords))

    # Step 1: Convert COCO17 body (0-16) to Body25 (0-24)
    # COCO17 joints: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
    #                l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
    #                l_knee, r_knee, l_ankle, r_ankle
    coco_body = keypoints[:, :17, :]  # (n_frames, 17, n_coords)

    # Map COCO17 to Body25 using COCO17_IN_BODY25
    # COCO17_IN_BODY25[i] tells us where COCO joint i goes in Body25
    for coco_idx in range(17):
        body25_idx = COCO17_IN_BODY25[coco_idx]
        kpts_out[:, body25_idx, :] = coco_body[:, coco_idx, :]

    # Compute missing joints by interpolation
    # Neck (Body25 index 1): midpoint of shoulders
    # COCO: l_shoulder=5, r_shoulder=6 -> Body25: LShoulder=5, RShoulder=2
    kpts_out[:, 1, :n_coords-1] = (kpts_out[:, 2, :n_coords-1] + kpts_out[:, 5, :n_coords-1]) / 2
    kpts_out[:, 1, n_coords-1] = np.minimum(kpts_out[:, 2, n_coords-1], kpts_out[:, 5, n_coords-1])

    # MidHip (Body25 index 8): midpoint of hips
    # Body25: LHip=12, RHip=9
    kpts_out[:, 8, :n_coords-1] = (kpts_out[:, 9, :n_coords-1] + kpts_out[:, 12, :n_coords-1]) / 2
    kpts_out[:, 8, n_coords-1] = np.minimum(kpts_out[:, 9, n_coords-1], kpts_out[:, 12, n_coords-1])

    # Body25 feet joints (19-24) from COCO-WholeBody feet (17-22)
    # COCO-WholeBody feet: 17-19 (left), 20-22 (right)
    # Body25 feet: 19-21 (left big toe, small toe, heel), 22-24 (right)
    if keypoints.shape[1] >= 23:
        kpts_out[:, 19:22, :] = keypoints[:, 17:20, :n_coords]  # Left foot
        kpts_out[:, 22:25, :] = keypoints[:, 20:23, :n_coords]  # Right foot

    # Step 2: Add hands
    # EasyMocap bodyhand format has empty slots at indices 25 and 46
    # The hand joints connect directly to body wrists (7 and 4) in the skeleton

    # Index 25: unused slot (set to zero with low confidence)
    kpts_out[:, 25, :n_coords-1] = 0
    kpts_out[:, 25, n_coords-1] = 0  # confidence = 0

    # Left hand: 26-45 (20 joints, COCO 92-111, excluding hand root at 91)
    kpts_out[:, 26:46, :] = keypoints[:, 92:112, :n_coords]

    # Index 46: unused slot (set to zero with low confidence)
    kpts_out[:, 46, :n_coords-1] = 0
    kpts_out[:, 46, n_coords-1] = 0  # confidence = 0

    # Right hand: 47-66 (20 joints, COCO 113-132, excluding hand root at 112)
    kpts_out[:, 47:67, :] = keypoints[:, 113:133, :n_coords]

    return kpts_out


def filter_keypoints(keypoints3d, keypoints2d, exclude_feet=False):
    """Filter keypoints to remove unwanted joints.

    Args:
        keypoints3d: (n_frames, 133, 4) array
        keypoints2d: (n_frames, n_views, 133, 3) array
        exclude_feet: If True, zero out feet keypoints

    Returns:
        Filtered keypoints with same shape
    """
    if not exclude_feet:
        return keypoints3d, keypoints2d

    # Zero out face keypoints (not used in body fitting) and feet if requested
    keypoints3d_filtered = keypoints3d.copy()
    keypoints2d_filtered = keypoints2d.copy() if keypoints2d is not None else None

    # Zero out face landmarks (23-90)
    keypoints3d_filtered[:, 23:91, 3] = 0  # Set confidence to 0

    # Zero out feet if requested (17-22)
    if exclude_feet:
        keypoints3d_filtered[:, 17:23, 3] = 0
        if keypoints2d_filtered is not None:
            keypoints2d_filtered[:, :, 17:23, 2] = 0

    if keypoints2d_filtered is not None:
        keypoints2d_filtered[:, :, 23:91, 2] = 0  # Face

    return keypoints3d_filtered, keypoints2d_filtered




def estimate_scale_from_keypoints(body_model, kp3ds, kintree=None, eps=1e-8):
    """Estimate scale factor between observed keypoints and model.

    Args:
        body_model: SMPL-X model
        kp3ds: (n_frames, n_joints, 4) observed keypoints
        kintree: list of limb connections
        eps: small value to avoid division by zero

    Returns:
        float: scale factor
    """
    kintree = np.array(kintree, dtype=int)
    src_idx = kintree[:, 0].astype(int)
    dst_idx = kintree[:, 1].astype(int)

    # Scale of observed keypoints
    vecs_obs = kp3ds[:, dst_idx, :3] - kp3ds[:, src_idx, :3]
    L_obs = np.linalg.norm(vecs_obs, axis=2)
    conf_obs = np.minimum(kp3ds[:, src_idx, 3], kp3ds[:, dst_idx, 3])

    # Scale of the model
    params0 = body_model.init_params(nFrames=1)
    kpts_model = body_model(return_verts=False, return_tensor=False, only_shape=True, **params0)[0]
    vecs_model = kpts_model[dst_idx, :3] - kpts_model[src_idx, :3]
    L_model = np.linalg.norm(vecs_model, axis=1)

    # Compute ratios
    ratios = []
    nLimbs = L_model.shape[0]
    for ts in range(kp3ds.shape[0]):
        for li in range(nLimbs):
            if L_obs[ts, li] > eps and L_model[li] > eps and conf_obs[ts, li] > 0.1:
                ratios.append(L_model[li] / (L_obs[ts, li] + eps))

    ratios = np.array(ratios)
    assert ratios.size > 0, "No valid limb ratios computed"
    s = float(np.median(ratios))
    assert np.isfinite(s) and s > 0, f'Invalid scale estimated: {s}'

    return s


def apply_scale_to_keypoints(kp3ds, s):
    """Apply scale factor to keypoints.

    Args:
        kp3ds: (n_frames, n_joints, 4) keypoints
        s: scale factor

    Returns:
        Scaled keypoints with same shape
    """
    coords = kp3ds[:, :, :3]
    centers = coords[:, 0:1, :]  # Use first joint as center
    scaled = (coords - centers) * float(s) + centers
    out = np.concatenate([scaled, kp3ds[..., 3:4]], axis=2)
    return out, centers


parser = argparse.ArgumentParser("SMPL-X Fitting Argument Parser")
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--to_smooth", action="store_true", help="Whether to temporally smooth the result")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
parser.add_argument('--exclude_feet', action='store_true', help='Exclude feet keypoints from fitting')

# EasyMocap args
parser.add_argument(
    '--body',
    type=str,
    default='wholebody',
    help='Body keypoint format'
)
parser.add_argument('--model', type=str, default='smplx', choices=['smpl', 'smplh', 'smplx'])
parser.add_argument("--optimize_bad_views", action="store_true", help="Whether to optimize extrinsics of bad views")
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
parser.add_argument('--save_origin', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--opts', help="Modify config options using the command-line",
    default={}, nargs='+')

recon = parser.add_argument_group('Reconstruction control')
recon.add_argument('--robust3d', action='store_true')

# Visualization
output = parser.add_argument_group('Output control')
output.add_argument('--write_smpl_full', action='store_true')
output.add_argument('--vis_2d_repro', action='store_true')
output.add_argument('--vis_3d_repro', action='store_true')
output.add_argument('--vis_smpl', action='store_true')
output.add_argument('--save_frame', action='store_true')
output.add_argument('--save_mesh', action='store_true')
output.add_argument("--confidence_thresh", type=float, default=None, help="Camera confidence")
args = parser.parse_args()


# Root paths
if args.optimize_bad_views:
    params_txt = "new_params.txt"
elif args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"
base_path = os.path.join(args.root_dir)
image_dir = os.path.join(base_path, args.seq_path)
output_path = args.out_dir
calib_dir = os.path.join(
    args.root_dir, args.seq_path, args.multisequence,
    "calib", f"stage{args.stage}", "sparse", "0"
)
params_path = os.path.join(calib_dir, params_txt)
assert os.path.exists(params_path), f"Params file not found: {params_path}"

# Filter out some cameras
if args.stage == 1:
    params = param_utils.read_params(params_path, distortion=True, args=args)
    use_parsed = False
elif args.stage == 2:
    params = param_utils.read_params(params_path, distortion=False, args=args)
    use_parsed = True
else:
    raise ValueError("Cannot determine whether to assume undistorted.")

cam_names = list(params[:]["cam_name"])
removed_camera_path = os.path.join(calib_dir, 'ignore_camera.txt')
if os.path.isfile(removed_camera_path):
    with open(removed_camera_path) as file:
        ignored_cameras = [line.rstrip() for line in file]
else:
    ignored_cameras = None

cams_to_remove = removed_cameras(
    remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras
)
for cam in cams_to_remove:
    if cam in cam_names:
        cam_names.remove(cam)

# Select videos
if args.ith == -1:
    total_video_idxs = 0
    max_folder_id = 0
    for fid, folder in enumerate(os.listdir(image_dir)):
        if 'cam' in folder and folder not in cams_to_remove:
            length = len([file for file in os.listdir(os.path.join(image_dir, folder)) if file.endswith('.mp4')])
            if length > total_video_idxs:
                total_video_idxs = length
                max_folder_id = fid
                anchor_camera_by_length = os.listdir(image_dir)[fid]
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

# Camera confidence
if args.confidence_thresh is not None:
    # conf_dir = args.out_dir[:args.out_dir.index("/stage")]
    # conf_path = os.path.join(conf_dir, "image_confidence.json")
    conf_path = os.path.join(
        args.root_dir, args.seq_path, args.multisequence, "calib", "image_confidence.json"
    )
    with open(conf_path, "r") as f:
        image_confidence = ujson.load(f)
    confident = {}
    for k, v in image_confidence.items():
        confident[k.replace(".jpg", "")] = v["num_visible_3D_points"] >= float(args.confidence_thresh)
else:
    confident = None

# Load SMPL-X model
print(f'Loading {args.model} model ({args.gender})...')
with Timer(f'Loading {args.model}, {args.gender}', not False):
    body_model: SMPLlayer = load_model(
        gender=args.gender,
        model_type=args.model,
        model_path="data/smplx",
        use_pose_blending=True,
        use_shape_blending=True,
        use_pca=False,
        use_flat_mean=False
    )

# Get EasyMocap's bodyhand config for Body25+hands keypoints (67 joints)
dataset_config = get_bodyhand_config(exclude_feet=args.exclude_feet)
print(f"Using EasyMocap bodyhand format with {dataset_config['nJoints']} joints for fitting (exclude_feet={args.exclude_feet})")

# Create config for COCO-WholeBody 2D visualization (original 133 keypoints)
def create_coco_wholebody_vis_config():
    """Simple COCO-WholeBody config for 2D keypoint visualization."""
    # COCO body skeleton
    body_kintree = [
        [0, 1], [0, 2], [1, 3], [2, 4],          # head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], # arms
        [5, 11], [6, 12], [11, 12],              # torso
        [11, 13], [13, 15], [12, 14], [14, 16],  # legs
        [15, 17], [15, 18], [15, 19],            # left foot
        [16, 20], [16, 21], [16, 22],            # right foot
    ]
    # Hand skeleton
    hand_kintree = [
        [0, 1], [1, 2], [2, 3], [3, 4],         # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],         # index
        [0, 9], [9, 10], [10, 11], [11, 12],    # middle
        [0, 13], [13, 14], [14, 15], [15, 16],  # ring
        [0, 17], [17, 18], [18, 19], [19, 20],  # pinky
    ]
    lhand_kintree = [[i + 91, j + 91] for i, j in hand_kintree]
    rhand_kintree = [[i + 112, j + 112] for i, j in hand_kintree]
    hand_connections = [[9, 91], [10, 112]]  # wrists to hands

    total_kintree = body_kintree + hand_connections + lhand_kintree + rhand_kintree
    total_colors = [[255, 0, 0]] * len(body_kintree) + [[0, 255, 0]] * 2 + [[255, 0, 255]] * (len(hand_kintree) * 2)

    return {
        'nJoints': 133,
        'kintree': total_kintree,
        'colors': total_colors,
    }

vis_config_2d = create_coco_wholebody_vis_config()

for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')

    if args.video_dir:
        video_dir = os.path.join(args.video_dir, args.seq_path)
    else:
        video_dir = image_dir

    # Read video
    reader = Reader(
        "video",
        video_dir,
        cam_names=cam_names,
        cams_to_remove=cams_to_remove,
        ith=selected_vid_idx,
        anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera
    )
    if reader.frame_count <= 0:
        continue

    # Keypoint paths
    keypoints2d_dir = os.path.join(output_path, "keypoints_2d", str(selected_vid_idx).zfill(3))
    keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(selected_vid_idx).zfill(3))
    keypt3d_file = os.path.join(keypoints3d_dir, "wholebody.jsonl")

    if not os.path.exists(keypt3d_file):
        print(f"Warning: 3D keypoints not found at {keypt3d_file}, skipping...")
        continue

    # Filter frames
    if args.use_filtered:
        chosen_path = os.path.join(keypoints3d_dir, "chosen_frames.json")
        if os.path.exists(chosen_path):
            with open(chosen_path, "r") as f:
                chosen_frames = list(set(json.load(f)))
        else:
            chosen_frames = list(range(args.start, args.end, args.stride))
    else:
        chosen_frames = list(range(args.start, args.end, args.stride))

    chosen_frames = sorted(chosen_frames)
    print(f"Total valid frames {len(chosen_frames)}/{reader.frame_count}")

    # Match camera names
    cam_mapper = map_camera_names(keypoints2d_dir, cam_names)
    extra_cams_to_remove = reader.to_delete
    cur_cam_names = cam_names.copy()
    for cam in extra_cams_to_remove:
        if cam in cur_cam_names:
            cur_cam_names.remove(cam)

    # Load camera parameters
    intrs, projs, dist_intrs, dists, cameras = get_projections(
        args, params, cur_cam_names, cam_mapper, easymocap_format=True
    )

    # Load 2D keypoints
    all_keypoints2d = []
    for cam in cur_cam_names:
        if cam in cam_mapper:
            keypoints2d = []
            kp_path = os.path.join(keypoints2d_dir, f"{cam_mapper[cam]}.jsonl")
            with open(kp_path, "r") as f:
                for l_idx, line in enumerate(f):
                    if l_idx in chosen_frames:
                        kp = np.array(ujson.loads(line)).reshape(-1, 3)
                        keypoints2d.append(kp)
            all_keypoints2d.append(np.asarray(keypoints2d))

    all_keypoints2d = np.asarray(all_keypoints2d)
    all_keypoints2d = np.swapaxes(all_keypoints2d, 0, 1)  # (n_frames, n_views, n_kpts, 3)

    # Undistort 2D keypoints
    for nf in range(all_keypoints2d.shape[0]):
        all_keypoints2d[nf, :, :, :2] = param_utils.undistort_points(
            all_keypoints2d[nf, :, :, :2], intrs, dists, dist_intrs
        )

    # Load 3D keypoints
    keypoints3d = []
    with open(keypt3d_file, "r") as f:
        for l_idx, line in enumerate(f):
            if l_idx in chosen_frames:
                keypoints3d.append(np.array(ujson.loads(line)).reshape(-1, 4))
    keypoints3d = np.asarray(keypoints3d)

    print(f"Loaded keypoints: 3D={keypoints3d.shape}, 2D={all_keypoints2d.shape}")

    # Convert COCO-WholeBody 133 to Body25+Hands 67 format
    print("Converting COCO-WholeBody to Body25+Hands format...")
    keypoints3d_selected = coco_wholebody_to_body25_hands(keypoints3d)
    all_keypoints2d_selected = coco_wholebody_to_body25_hands(all_keypoints2d)

    print(f"Converted keypoints for fitting: 3D={keypoints3d_selected.shape}, 2D={all_keypoints2d_selected.shape}")

    # Debug: Find a valid frame and verify conversion
    valid_frame_idx = -1
    for i in range(len(keypoints3d)):
        if keypoints3d[i, 0, 3] > 0 and keypoints3d[i, 11, 3] > 0:
            valid_frame_idx = i
            break

    # if valid_frame_idx >= 0:
    #     print(f"\nDEBUG - Keypoint conversion (frame {valid_frame_idx}):")
    #     print(f"  COCO nose [0]: {keypoints3d[valid_frame_idx, 0, :3]} -> Body25 [0]: {keypoints3d_selected[valid_frame_idx, 0, :3]}")
    #     print(f"  COCO l_shoulder [5]: {keypoints3d[valid_frame_idx, 5, :3]} -> Body25 [5]: {keypoints3d_selected[valid_frame_idx, 5, :3]}")
    #     print(f"  COCO r_shoulder [6]: {keypoints3d[valid_frame_idx, 6, :3]} -> Body25 [2]: {keypoints3d_selected[valid_frame_idx, 2, :3]}")
    #     print(f"  COCO l_hip [11]: {keypoints3d[valid_frame_idx, 11, :3]} -> Body25 [12]: {keypoints3d_selected[valid_frame_idx, 12, :3]}")
    #     print(f"  COCO r_hip [12]: {keypoints3d[valid_frame_idx, 12, :3]} -> Body25 [9]: {keypoints3d_selected[valid_frame_idx, 9, :3]}")
    #     print(f"  Computed Neck [1]: {keypoints3d_selected[valid_frame_idx, 1, :3]}")
    #     print(f"  Computed MidHip [8]: {keypoints3d_selected[valid_frame_idx, 8, :3]}")
    # else:
    #     print("\nWARNING: No valid frames with keypoint detections!")

    # Smooth keypoints if requested
    if args.to_smooth and len(keypoints3d_selected) > 3:
        print('Smoothing 3D keypoints...')
        keypoints3d_selected = apply_one_euro_filter_3d(
            keypoints3d_selected, mincutoff=0.5, beta=0.0, dcutoff=1.0
        )

    # Estimate scale and apply
    print("Estimating scale...")
    scale = estimate_scale_from_keypoints(
        body_model, keypoints3d_selected, kintree=dataset_config.get('kintree', None)
    )
    print(f"Estimated scale: {scale:.4f}")
    keypoints3d_scaled, root = apply_scale_to_keypoints(keypoints3d_selected, scale)
    final_scale = 1.0 / scale

    # Fit SMPL-X
    print("Fitting SMPL-X model...")
    weight_pose = {
        'k3d': 1e2, 'k2d': 2e-3,
        'reg_poses': 5e-5, 'smooth_body': 1e1, 'smooth_poses': 5.0,
    }

    # Choose between 3D-only or 3D+2D fitting
    fit_3d2d = False  # Can be made an argument if needed

    if fit_3d2d:
        # Create fake bboxes for 3D+2D fitting
        all_bboxes = np.ones((all_keypoints2d_selected.shape[0], all_keypoints2d_selected.shape[1], 5))
        params_body = smpl_from_keypoints3d2d(
            body_model, keypoints3d_scaled, all_keypoints2d_selected, all_bboxes, projs,
            config=dataset_config, args=args,
            weight_shape={'s3d': 1e5, 'reg_shapes': 5e3},
            weight_pose=weight_pose
        )
    else:
        params_body = smpl_from_keypoints3d(
            body_model, keypoints3d_scaled,
            config=dataset_config, args=args,
            weight_shape={'s3d': 1e5, 'reg_shapes': 1e2},
            weight_pose=weight_pose
        )

    # Smooth SMPL-X parameters if requested
    if args.to_smooth and len(params_body['Rh']) > 3:
        print('Smoothing SMPL-X parameters...')
        params_body['Rh'] = apply_one_euro_filter_2d(params_body['Rh'], mincutoff=0.5, beta=0.0, dcutoff=1.0)
        params_body['Th'] = apply_one_euro_filter_2d(params_body['Th'], mincutoff=0.5, beta=0.0, dcutoff=1.0)
        params_body['poses'] = apply_one_euro_filter_2d(params_body['poses'], mincutoff=0.5, beta=0.0, dcutoff=1.0)

    # Save parameters
    params_list = {}
    for key in params_body:
        params_list[key] = params_body[key].tolist()

    out_params_path = os.path.join(output_path, 'params', f'{str(selected_vid_idx).zfill(3)}.json')
    os.makedirs(os.path.dirname(out_params_path), exist_ok=True)
    with open(out_params_path, "w") as f:
        ujson.dump(params_list, f)
    print(f"Saved parameters to {out_params_path}")

    # Visualization and mesh export
    if args.vis_smpl or args.save_mesh or args.vis_2d_repro or args.vis_3d_repro:
        # Setup output directories
        if args.vis_smpl:
            out_smpl_path = os.path.join(output_path, 'smpl', str(selected_vid_idx).zfill(3))
            os.makedirs(out_smpl_path, exist_ok=True)
        if args.vis_2d_repro:
            out_2d_path = os.path.join(output_path, 'repro_2d', str(selected_vid_idx).zfill(3))
            os.makedirs(out_2d_path, exist_ok=True)
        if args.vis_3d_repro:
            out_3d_path = os.path.join(output_path, 'repro_3d', str(selected_vid_idx).zfill(3))
            os.makedirs(out_3d_path, exist_ok=True)

        out_joint_path = os.path.join(output_path, 'regress_joints', str(selected_vid_idx).zfill(3))
        os.makedirs(out_joint_path, exist_ok=True)

        nf = 0
        if not use_parsed:
            generator = reader(chosen_frames)

        for abs_idx, chosen_f in tqdm(enumerate(chosen_frames)):
            if use_parsed:
                # multiseq_dir = args.out_dir[:args.out_dir.index("calib")]
                parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")
                timestamp_dir = os.path.join(parsed_dir, f"timestamp_{chosen_f}", "images")
                frames = {}
                for cam in cur_cam_names:
                    if cam in cam_mapper:
                        frame_path = os.path.join(timestamp_dir, f"{cam}.jpg")
                        if os.path.exists(frame_path):
                            image_s2 = cv2.imread(frame_path)
                        else:
                            image_s2 = np.ones((params["height"][0], params["width"][0], 3), dtype=np.uint8) * 255
                        frames[cam_mapper[cam]] = image_s2
            else:
                (frames, idx) = next(generator)

            # Undistort images
            images = []
            c_idx = 0
            for cam in cur_cam_names:
                if cam in cam_mapper:
                    image = frames[cam_mapper[cam]]
                    if args.undistort and not (dists[c_idx] == 0).all():
                        image = param_utils.undistort_image(intrs[c_idx], dist_intrs[c_idx], dists[c_idx], image)
                    c_idx += 1
                    images.append(image)

            param_frame = select_nf(params_body, nf)

            if abs_idx % args.stride == 0:
                # Visualize SMPL-X mesh
                if args.vis_smpl:
                    vertices = body_model(return_verts=True, return_tensor=False, **param_frame)

                    # Scale vertices
                    # Use MidHip as root (average of left_hip and right_hip in COCO-WholeBody)
                    # COCO-WholeBody: left_hip=11, right_hip=12
                    # root = (keypoints3d[abs_idx][11, :3] + keypoints3d[abs_idx][12, :3]) / 2
                    vertices_scaled = (vertices - root[abs_idx:abs_idx+1]) * final_scale + root[abs_idx:abs_idx+1]
                    vertices_scaled = vertices_scaled.squeeze(0)

                    image_vis, render_results = vis_smpl(
                        args, vertices=vertices_scaled, faces=body_model.faces,
                        images=images, nf=nf, cameras=cameras, add_back=True,
                        out_dir=out_smpl_path, confident=confident
                    )
                    if abs_idx == 0:
                        out_smpl = create_video_writer(
                            out_smpl_path + ".mp4",
                            (image_vis.shape[1], image_vis.shape[0]), fps=30
                        )
                    out_smpl.write(image_vis)

                # Save mesh
                if args.save_mesh:
                    mesh = trimesh.Trimesh(vertices=vertices_scaled, faces=body_model.faces)
                    outdir = os.path.join(output_path, f'meshes/{str(selected_vid_idx).zfill(3)}')
                    os.makedirs(outdir, exist_ok=True)
                    outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                    mesh.export(outname)

                # Visualize 3D keypoint reprojection (Body25+Hands 67 keypoints)
                if args.vis_3d_repro:
                    # Use converted Body25+hands keypoints for 3D visualization
                    kpts_repro = projectN3(keypoints3d_selected[abs_idx], projs)
                    kpts_repro[:, :, 2] = 0.5  # Set all confidences to 0.5 for vis
                    image_vis = vis_repro(
                        args, images, kpts_repro, config=dataset_config,
                        nf=nf, mode='repro_smpl', outdir=out_3d_path,
                        cameras=cameras, confident=confident
                    )
                    if abs_idx == 0:
                        out_3d = create_video_writer(
                            out_3d_path + ".mp4",
                            (image_vis.shape[1], image_vis.shape[0]), fps=30
                        )
                    out_3d.write(image_vis)

                # Visualize regressed joints
                joints = body_model(return_verts=False, return_tensor=False, **param_frame)
                joints_scaled = (joints - root[abs_idx:abs_idx+1]) * final_scale + root[abs_idx:abs_idx+1]
                joints_scaled = joints_scaled.squeeze(0)
                joints_repro = projectN3(joints_scaled, projs)
                joints_repro[:, :, 2] = 0.5

                if args.vis_smpl:
                    image_vis = vis_repro(
                        args, render_results, joints_repro, config=dataset_config,
                        nf=nf, mode='repro_smpl', outdir=out_joint_path,
                        cameras=cameras, confident=confident
                    )
                else:
                    image_vis = vis_repro(
                        args, images, joints_repro, config=dataset_config,
                        nf=nf, mode='repro_smpl', outdir=out_joint_path,
                        cameras=cameras, confident=confident
                    )

                if abs_idx == 0:
                    out_joint = create_video_writer(
                        out_joint_path + ".mp4",
                        (image_vis.shape[1], image_vis.shape[0]), fps=30
                    )
                out_joint.write(image_vis)

                # Visualize 2D keypoints (original COCO-WholeBody 133 keypoints)
                if args.vis_2d_repro:
                    kpts_repro = all_keypoints2d[abs_idx]
                    image_vis = vis_repro(
                        args, images, kpts_repro, config=vis_config_2d,
                        nf=nf, mode='repro_smpl', outdir=out_2d_path,
                        cameras=cameras, confident=confident
                    )
                    if abs_idx == 0:
                        out_2d = create_video_writer(
                            out_2d_path + ".mp4",
                            (image_vis.shape[1], image_vis.shape[0]), fps=30
                        )
                    out_2d.write(image_vis)

            nf += 1

        # Release video writers
        if args.vis_smpl:
            out_smpl.release()
            convert_video_ffmpeg(out_smpl_path + ".mp4")
            print('SMPL video saved')
        if args.vis_2d_repro:
            out_2d.release()
            convert_video_ffmpeg(out_2d_path + ".mp4")
            print('2D repro video saved')
        if args.vis_3d_repro:
            out_3d.release()
            convert_video_ffmpeg(out_3d_path + ".mp4")
            print('3D repro video saved')

        out_joint.release()
        convert_video_ffmpeg(out_joint_path + ".mp4")
        print('Joint video saved')

print("SMPL-X fitting complete!")
