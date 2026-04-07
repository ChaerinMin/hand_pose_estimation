import argparse
import json
import os
import sys
import cv2

import numpy as np
import ujson
from tqdm import tqdm
import trimesh

import src.utils.params as param_utils
from src.utils.cameras import (get_projections, map_camera_names,
                               removed_cameras)
from src.utils.easymocap_utils import (load_model, projectN3, vis_repro,
                                       vis_smpl)
from src.utils.filter import apply_one_euro_filter_2d, apply_one_euro_filter_3d, apply_savgol_filter_2d, apply_savgol_filter_3d, apply_savgol_filter_rotvec, reject_outliers_median_2d, reject_outliers_median_3d, canonicalize_rotvec_sequence, reject_rotation_outliers
from src.utils.parser import add_common_args
from src.utils.reader_v2 import Reader
from src.utils.video_handler import convert_video_ffmpeg, create_video_writer
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.pipeline import smpl_from_keypoints3d, smpl_from_keypoints3d2d
from easymocap.pyfitting import optimizeShape
from easymocap.smplmodel import select_nf
from easymocap.smplmodel.body_model import SMPLlayer

os.environ['PYOPENGL_PLATFORM'] = 'egl' 
os.system("module load ffmpeg")
sys.path.append(".")
sys.path.append("./third-party/EasyMocap")


def estimate_scale_from_keypoints(body_model, kp3ds, kintree=None, eps=1e-8):
    # model
    kintree = np.array(kintree, dtype=int)
    src_idx = kintree[:, 0].astype(int)
    dst_idx = kintree[:, 1].astype(int)

    # scale of our keypoints
    vecs_obs = kp3ds[:, dst_idx, :3] - kp3ds[:, src_idx, :3]
    L_obs = np.linalg.norm(vecs_obs, axis=2)

    # scale of the model
    params0 = body_model.init_params(nFrames=1)
    kpts_model = body_model(return_verts=False, return_tensor=False, only_shape=True, **params0)[0]
    vecs_model = kpts_model[dst_idx, :3] - kpts_model[src_idx, :3]
    L_model = np.linalg.norm(vecs_model, axis=1)

    # ratio
    ratios = []
    nLimbs = L_model.shape[0]
    for ts in range(kp3ds.shape[0]):
        for li in range(nLimbs):
            if L_obs[ts, li] > eps and L_model[li] > eps:
                ratios.append(L_model[li] / (L_obs[ts, li] + eps))
    ratios = np.array(ratios)
    assert ratios.size > 0, " No valid limb ratios computed"
    s = float(np.median(ratios))
    assert np.isfinite(s) and s > 0, f'Invalid scale estimated: {s}'
    return s


def apply_scale_to_keypoints(kp3ds, s):
    coords = kp3ds[:, :, :3]
    centers = coords[:, 0:1, :]
    scaled = (coords - centers) * float(s) + centers
    out = np.concatenate([scaled, kp3ds[..., 3:4]], axis=2)
    return out



parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--to_smooth", action="store_true", help="Whether to temporally smoothing the result")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
# Easy Mocap
parser.add_argument(
    '--body',
    type=str,
    default='body25',
    choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'handlr', 'total']
)
parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
parser.add_argument("--optimize_bad_views", action="store_true", help="Whether to optimize extrinsics of bad views")
parser.add_argument("--outlier_rejection", action="store_true", default=False, help="Reject outliers before smoothing (requires --to_smooth)")
parser.add_argument("--outlier_window", type=int, default=5, help="Sliding window size for outlier rejection")
parser.add_argument("--outlier_threshold", type=float, default=0.5, help="MAD multiplier threshold for outlier rejection")
parser.add_argument("--savgol", action=argparse.BooleanOptionalAction, default=True, help="Use zero-phase Savitzky-Golay filter instead of One Euro filter (requires --to_smooth)")
parser.add_argument("--savgol_window", type=int, default=11, help="Window length for Savitzky-Golay filter (must be odd)")
parser.add_argument("--savgol_polyorder", type=int, default=3, help="Polynomial order for Savitzky-Golay filter")
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
parser.add_argument("--refine_shape_with_mask", action="store_true", help="Refine MANO shape parameters using hand masks")
parser.add_argument(
    "--subject_name", type=str, default=None,
    help="If refine_shape_with_mask, save beta with subject_name. " \
        "If not refine_shape_with_mask, load beta with subject_name. " \
        "If not provided, do not load beta"
)
parser.add_argument('--save_origin', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--opts', help="Modify config options using the command-line", 
    default={}, nargs='+')
recon = parser.add_argument_group('Reconstruction control')
recon.add_argument('--robust3d', action='store_true')
# visualization
output = parser.add_argument_group('Output control')
output.add_argument('--write_smpl_full', action='store_true')
output.add_argument('--vis_2d_repro', action='store_true')
output.add_argument('--vis_3d_repro', action='store_true')
output.add_argument('--vis_smpl', action='store_true')
output.add_argument('--save_frame', action='store_true')
output.add_argument('--save_mesh', action='store_true')
output.add_argument("--confidence_thresh", type=float, default=None, help="camera conficence")
args = parser.parse_args()
args.out_dir = os.path.join(args.out_dir, "hand")


# root paths
if args.optimize_bad_views: 
    params_txt = "new_params.txt"
elif args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"
base_path = os.path.join(args.root_dir)
image_dir = os.path.join(base_path, args.seq_path)
calib_dir = os.path.join(
    args.root_dir, args.seq_path, args.multisequence,
    "calib", f"stage{args.stage}", "sparse", "0"
)
params_path = os.path.join(calib_dir, params_txt)
assert os.path.exists(params_path)

# filter out some cameras
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

# select videos in brics
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

# camera confidence
if args.confidence_thresh is not None:
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

if args.video_dir:
    video_dir = os.path.join(args.video_dir, args.seq_path)
else:
    video_dir = image_dir

for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')
    # read video
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
    
    # 2d/3d keypoint paths
    keypoints2d_dir_right = os.path.join(args.out_dir, "intermediate", "keypoints_2d", "right", str(selected_vid_idx).zfill(3))
    keypoints2d_dir_left = os.path.join(args.out_dir, "intermediate", "keypoints_2d", "left",  str(selected_vid_idx).zfill(3))
    bboxes_dir_right = os.path.join(args.out_dir, "intermediate", "bboxes", "right",  str(selected_vid_idx).zfill(3))
    bboxes_dir_left = os.path.join(args.out_dir, "intermediate", "bboxes", "left",  str(selected_vid_idx).zfill(3))
    keypoints3d_dir = os.path.join(args.out_dir, "intermediate", "keypoints_3d", str(selected_vid_idx).zfill(3))
    keypt3d_file_left = os.path.join(keypoints3d_dir, "left.jsonl")
    keypt3d_file_right = os.path.join(keypoints3d_dir, "right.jsonl")

    # Always read chosen_frames_* for per-hand validity check
    chosen_path_left = os.path.join(keypoints3d_dir, "chosen_frames_left.json")
    chosen_path_right = os.path.join(keypoints3d_dir, "chosen_frames_right.json")
    with open(chosen_path_right, "r") as f:
        chosen_frames_right = json.load(f)
    with open(chosen_path_left, "r") as f:
        chosen_frames_left = json.load(f)

    # fileter out some frames
    if args.use_filtered:
        chosen_frames = list(set(chosen_frames_right) | set(chosen_frames_left))
    else:
        chosen_frames = range(args.start, args.end, args.stride)
    chosen_frames = sorted(chosen_frames)
    print(f"Total valid frames {len(chosen_frames)}/{reader.frame_count}")
    
    # match camera anmes with keypoints
    cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)
    extra_cams_to_remove = reader.to_delete
    cur_cam_names = cam_names.copy()
    for cam in extra_cams_to_remove:
        if cam in cur_cam_names:
            cur_cam_names.remove(cam)

    # load camera parameters
    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cur_cam_names, cam_mapper, easymocap_format=True)

    # load 2d keypoints
    all_keypoints2d_left, all_keypoints2d_right = [], []
    all_bboxes_left, all_bboxes_right = [], []
    for cam in cur_cam_names:
        if cam in cam_mapper:
            keypoints2d_left = []
            keypoints2d_right = []
            bboxes_left = []
            bboxes_right = []
            ap_keypoints_path_left = os.path.join(keypoints2d_dir_left, f"{cam_mapper[cam]}.jsonl")
            ap_keypoints_path_right = os.path.join(keypoints2d_dir_right, f"{cam_mapper[cam]}.jsonl")
            bboxes_path_left = os.path.join(bboxes_dir_right, f"{cam_mapper[cam]}.jsonl")
            bboxes_path_right = os.path.join(bboxes_dir_left, f"{cam_mapper[cam]}.jsonl")
            with open(ap_keypoints_path_left, "r") as fl, open(ap_keypoints_path_right, "r") \
                as fr, open(bboxes_path_left, "r") as fbl, open(bboxes_path_right, "r") as fbr:
                for l_idx, (linel, liner, linebl, linebr) in enumerate(zip(fl, fr, fbl, fbr)):
                    if l_idx in chosen_frames:
                        keypoints2d_left.append(np.array(ujson.loads(linel)).reshape(-1, 3))
                        keypoints2d_right.append(np.array(ujson.loads(liner)).reshape(-1, 3))
                        bboxes_left.append(np.array(ujson.loads(linebl) + [1.0]))
                        bboxes_right.append(np.array(ujson.loads(linebr) + [1.0]))
            all_keypoints2d_left.append(np.asarray(keypoints2d_left))
            all_keypoints2d_right.append(np.asarray(keypoints2d_right))
            all_bboxes_left.append(np.asarray(bboxes_left))
            all_bboxes_right.append(np.asarray(bboxes_right))
    all_keypoints2d_left = np.asarray(all_keypoints2d_left)
    all_keypoints2d_right = np.asarray(all_keypoints2d_right)
    all_bboxes_left = np.asarray(all_bboxes_left)
    all_bboxes_right = np.asarray(all_bboxes_right)
    all_keypoints2d_left = np.swapaxes(all_keypoints2d_left, 0, 1)
    all_keypoints2d_right = np.swapaxes(all_keypoints2d_right, 0, 1)
    all_bboxes_left = np.swapaxes(all_bboxes_left, 0, 1)
    all_bboxes_right = np.swapaxes(all_bboxes_right, 0, 1)
    for nf in range(all_keypoints2d_left.shape[0]):
        all_keypoints2d_left[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_left[nf, :, :, :2], intrs, dists, dist_intrs)
    for nf in range(all_keypoints2d_right.shape[0]):
        all_keypoints2d_right[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_right[nf, :, :, :2], intrs, dists, dist_intrs)
        
    # load 3d keypoints
    # For visualization: load for the full union of chosen frames
    # For MANO fitting: load only each hand's own quality-checked frames so that
    # frames where the hand is absent (zeros / garbage) are never passed to the fitter.
    chosen_frames_right_set = set(chosen_frames_right)
    chosen_frames_left_set = set(chosen_frames_left)
    keypoints3d_right, keypoints3d_left = [], []
    keypoints3d_right_mano, keypoints3d_left_mano = [], []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                kp_left = np.array(ujson.loads(linel)).reshape(-1, 4)
                kp_right = np.array(ujson.loads(liner)).reshape(-1, 4)
                keypoints3d_left.append(kp_left)
                keypoints3d_right.append(kp_right)
            if l_idx in chosen_frames_left_set:
                keypoints3d_left_mano.append(np.array(ujson.loads(linel)).reshape(-1, 4))
            if l_idx in chosen_frames_right_set:
                keypoints3d_right_mano.append(np.array(ujson.loads(liner)).reshape(-1, 4))
    keypoints3d_left = np.asarray(keypoints3d_left)
    keypoints3d_right = np.asarray(keypoints3d_right)
    keypoints3d_left_mano = np.asarray(keypoints3d_left_mano) if keypoints3d_left_mano else np.zeros((0, 21, 4))
    keypoints3d_right_mano = np.asarray(keypoints3d_right_mano) if keypoints3d_right_mano else np.zeros((0, 21, 4))

    # A hand is valid only if it has quality-checked frames to fit MANO to.
    right_valid = len(chosen_frames_right) > 0
    left_valid = len(chosen_frames_left) > 0
    print(f"Hand validity — right: {right_valid} ({len(chosen_frames_right)} frames), left: {left_valid} ({len(chosen_frames_left)} frames)")

    # load hand masks if refine_shape_with_mask is enabled
    hand_masks = None
    seg_status = {}
    if args.refine_shape_with_mask:
        mask_dir = os.path.join(args.out_dir, "intermediate", "mask_2d", str(selected_vid_idx).zfill(3))
        mask_path = os.path.join(mask_dir, "hand_masks.npz")
        if os.path.exists(mask_path):
            print(f"Loading hand masks from {mask_path}")
            mask_data = np.load(mask_path)
            hand_masks = {}
            # Load masks for each camera
            for cam in cur_cam_names:
                if cam in cam_mapper and cam in mask_data:
                    hand_masks[cam] = mask_data[cam]
                    # Load segmentation status
                    seg_status[f"{cam}_left"] = bool(mask_data.get(f"{cam}_left", False))
                    seg_status[f"{cam}_right"] = bool(mask_data.get(f"{cam}_right", False))
            print(f"  - Loaded masks for {len(hand_masks)} cameras")
            print(f"  - Segmentation status: {sum(seg_status.values())} successful segmentations")

            # Assert that all views have the first frame available
            if use_parsed:
                multiseq_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence)
                parsed_dir = os.path.join(multiseq_dir, "parsed")
                first_frame = chosen_frames[0]
                timestamp_dir = os.path.join(parsed_dir, f"timestamp_{first_frame}", "images")
                for cam in cur_cam_names:
                    if cam in cam_mapper:
                        frame_path = os.path.join(timestamp_dir, f"{cam}.jpg")
                        assert os.path.exists(frame_path), f"Is this bootstarp sequence? Missing frame detected."
                print(f"  - Verified all {len(cur_cam_names)} cameras have first frame available")
        else:
            raise FileNotFoundError(f"Hand masks not found at {mask_path}. Please run mask_2d.py first.")
    
    # mano model
    with Timer('Loading {}, {}'.format(args.model, args.gender), not False):
        body_model_right: SMPLlayer = load_model(
            gender=args.gender,
            model_type=args.model, 
            model_path="data/smplx",
            num_pca_comps=6,
            use_pose_blending=True,
            use_shape_blending=True,
            use_pca=False,
            use_flat_mean=False
        )
    with Timer('Loading {}, {}'.format(args.model, args.gender), not False):
        body_model_left: SMPLlayer = load_model(
            gender=args.gender,
            model_type=args.model.replace('r', 'l'),
            model_path="data/smplx",
            num_pca_comps=6,
            use_pose_blending=True,
            use_shape_blending=True,
            use_pca=False,
            use_flat_mean=False
        )

    # fit mano
    dataset_config = CONFIG[args.body]
    if right_valid or left_valid:
        # smooth keypoints
        if args.to_smooth:
            print('Smoothing Keypoints 3D...')
            kp3d_left = reject_outliers_median_3d(keypoints3d_left[:, :, :3], window=args.outlier_window, threshold=args.outlier_threshold) if args.outlier_rejection else keypoints3d_left[:, :, :3]
            keypoints3d_left[:, :, :3] = apply_savgol_filter_3d(kp3d_left, window=args.savgol_window, polyorder=args.savgol_polyorder) if args.savgol else apply_one_euro_filter_3d(kp3d_left, mincutoff=0.5, beta=0.0, dcutoff=1.0)
            kp3d_right = reject_outliers_median_3d(keypoints3d_right[:, :, :3], window=args.outlier_window, threshold=args.outlier_threshold) if args.outlier_rejection else keypoints3d_right[:, :, :3]
            keypoints3d_right[:, :, :3] = apply_savgol_filter_3d(kp3d_right, window=args.savgol_window, polyorder=args.savgol_polyorder) if args.savgol else apply_one_euro_filter_3d(kp3d_right, mincutoff=0.5, beta=0.0, dcutoff=1.0)

        # keypoints -> mano parameters
        weight_pose = {
            'k3d': 1e2, 'k2d': 2e-3,
            # 'reg_poses': 1e-3, 'smooth_body': 1e2, 'smooth_poses': 1e2,
            'reg_poses': 5e-5, 'smooth_body': 1e1, 'smooth_poses': 5.0,
            # 'reg_poses': 0.0, 'smooth_body': 0.0, 'smooth_poses': 0.0
        }
        # Estimate global scale from per-hand quality-checked frames only,
        # but apply scale to the full union array so params have N_total frames
        # and select_nf works correctly in the visualization loop.
        if right_valid:
            s_right = estimate_scale_from_keypoints(body_model_right, keypoints3d_right_mano, kintree=dataset_config.get('kintree', None))
            keypoints3d_right_mano_scaled = apply_scale_to_keypoints(keypoints3d_right_mano, s_right)
            final_scale_right = 1.0 / s_right
        else:
            final_scale_right = 1.0
        if left_valid:
            s_left = estimate_scale_from_keypoints(body_model_left, keypoints3d_left_mano, kintree=dataset_config.get('kintree', None))
            keypoints3d_left_mano_scaled = apply_scale_to_keypoints(keypoints3d_left_mano, s_left)
            final_scale_left = 1.0 / s_left
        else:
            final_scale_left = 1.0
        # Load personalized shape parameters if subject_name is given without refine_shape_with_mask
        init_shapes_right = None
        init_shapes_left = None
        if not args.refine_shape_with_mask and args.subject_name is not None:
            shape_save_path = os.path.join(args.root_dir, "personlized_shapes.json")
            if os.path.exists(shape_save_path):
                with open(shape_save_path, "r") as f:
                    all_shapes = json.load(f)
                if args.subject_name in all_shapes:
                    subject_shapes = all_shapes[args.subject_name]
                    if right_valid and "right" in subject_shapes:
                        init_shapes_right = np.array(subject_shapes["right"]).reshape(1, -1)
                    if left_valid and "left" in subject_shapes:
                        init_shapes_left = np.array(subject_shapes["left"]).reshape(1, -1)
                    print(f"Loaded personalized shapes for subject '{args.subject_name}'")
                else:
                    print(f"Warning: Subject '{args.subject_name}' not found in {shape_save_path}")
            else:
                print(f"Warning: Shape file not found at {shape_save_path}")

        CHUNK_SIZE = 300
        SHAPE_SAMPLE = 100 

        def smpl_from_keypoints3d_chunked(body_model, kp3ds, weight_shape, weight_pose, init_shapes):
            nFrames = kp3ds.shape[0]
            if nFrames <= CHUNK_SIZE:
                return smpl_from_keypoints3d(body_model, kp3ds,
                    config=dataset_config, args=args,
                    weight_shape=weight_shape, weight_pose=weight_pose,
                    init_shapes=init_shapes)
            print(f"  [chunked] {nFrames} frames > {CHUNK_SIZE}, splitting into chunks")
            if init_shapes is None:
                sample_idx = np.linspace(0, nFrames - 1, min(SHAPE_SAMPLE, nFrames), dtype=int)
                kp3ds_sample = kp3ds[sample_idx]
                params_init = body_model.init_params(nFrames=1)
                params_shape = optimizeShape(body_model, params_init, kp3ds_sample,
                    weight_loss=weight_shape, kintree=dataset_config['kintree'])
                shapes = params_shape['shapes']
                print(f"  [chunked] shape estimated from {len(sample_idx)} frames")
            else:
                shapes = init_shapes
            rh_list, th_list, poses_list = [], [], []
            for chunk_start in range(0, nFrames, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, nFrames)
                chunk_kp3d = kp3ds[chunk_start:chunk_end]
                print(f"  [chunked] frames {chunk_start}-{chunk_end-1}")
                chunk_params = smpl_from_keypoints3d(body_model, chunk_kp3d,
                    config=dataset_config, args=args,
                    weight_shape=weight_shape, weight_pose=weight_pose,
                    init_shapes=shapes)
                rh_list.append(chunk_params['Rh'])
                th_list.append(chunk_params['Th'])
                poses_list.append(chunk_params['poses'])
            return {
                'Rh': np.concatenate(rh_list, axis=0),
                'Th': np.concatenate(th_list, axis=0),
                'poses': np.concatenate(poses_list, axis=0),
                'shapes': shapes,
            }

        fit_3d2d = False
        params_right = None
        params_left = None
        if fit_3d2d:
            if right_valid:
                params_right = smpl_from_keypoints3d2d(
                    body_model_right, keypoints3d_right_mano_scaled, all_keypoints2d_right, all_bboxes_right, projs,
                    config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose
                )
            if left_valid:
                params_left = smpl_from_keypoints3d2d(
                    body_model_left, keypoints3d_left_mano_scaled, all_keypoints2d_left, all_bboxes_left, projs,
                    config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose
                )
        else:
            if right_valid:
                params_right = smpl_from_keypoints3d_chunked(
                    body_model_right, keypoints3d_right_mano_scaled,
                    weight_shape={'s3d': 1e5, 'reg_shapes': 1e2}, weight_pose=weight_pose,
                    init_shapes=init_shapes_right)
            if left_valid:
                params_left = smpl_from_keypoints3d_chunked(
                    body_model_left, keypoints3d_left_mano_scaled,
                    weight_shape={'s3d': 1e5, 'reg_shapes': 1e2}, weight_pose=weight_pose,
                    init_shapes=init_shapes_left)

        # hand masks --> shape (beta)
        if args.refine_shape_with_mask and hand_masks is not None:
            print('Refining MANO shape parameters with hand masks...')
            from src.utils.mask_optimize import refine_shape_with_mask

            # Right hand's shape (beta)
            if right_valid:
                params_right = refine_shape_with_mask(
                    body_model_right, params_right, hand_masks, cameras,
                    cur_cam_names, cam_mapper, intrs, final_scale_right, keypoints3d_right_mano[0], hand_side="right",
                    # weight_loss={'mask': 1e4, 'reg_shapes': 1e1, 'init_shape': 5e1},  # 1e3 1e2 5e2
                    weight_loss={'mask': 1e4, 'reg_shapes': 0.5, 'init_shape': 0.0},  # 1e3 1e2 5e2
                    max_iter=20, verbose=True
                )

            # Left hand's shape (beta)
            if left_valid:
                params_left = refine_shape_with_mask(
                    body_model_left, params_left, hand_masks, cameras,
                    cur_cam_names, cam_mapper, intrs, final_scale_left, keypoints3d_left_mano[0], hand_side="left",
                    weight_loss={'mask': 1e4, 'reg_shapes': 0.5, 'init_shape': 0.0},
                    max_iter=20, verbose=True
                )

            # Save shape (beta) with subject's name
            shape_save_path = os.path.join(args.root_dir, "personlized_shapes.json")
            os.makedirs(os.path.dirname(shape_save_path), exist_ok=True)
            if os.path.exists(shape_save_path):
                with open(shape_save_path, "r") as f:
                    current_persons = json.load(f)
            else:
                current_persons = {}
            current_persons[args.subject_name] = {}
            if left_valid:
                current_persons[args.subject_name]["left"] = params_left['shapes'].squeeze(0).tolist()
            if right_valid:
                current_persons[args.subject_name]["right"] = params_right['shapes'].squeeze(0).tolist()
            with open(shape_save_path, "w") as f:
                json.dump(current_persons, f, indent=4)

        # smooth mano parameters
        if args.to_smooth:
            print('Smoothing Manos...')
            def smooth_params_per_segment(params, frame_indices):
                """Apply smoothing within each contiguous temporal segment independently.
                Prevents savgol from blending across gaps where the hand was absent.
                Rh (global rotation) is canonicalized in quaternion space first to remove
                axis-angle π-singularity flip artifacts before smoothing."""
                frames = np.array(frame_indices)
                diffs = np.diff(frames)
                boundaries = np.where(diffs > 1)[0] + 1
                seg_starts = np.concatenate([[0], boundaries])
                seg_ends = np.concatenate([boundaries, [len(frames)]])
                for start, end in zip(seg_starts, seg_ends):
                    for key in ('Rh', 'Th', 'poses'):
                        seg = params[key][start:end]
                        if len(seg) < 2:
                            continue
                        if key == 'Rh':
                            seg = canonicalize_rotvec_sequence(seg)
                            # 0.5 rad (~29°) geodesic threshold: catches bad MANO local minima
                            # without over-rejecting legitimate fast hand rotations
                            seg = reject_rotation_outliers(seg, window=args.outlier_window, threshold=0.5)
                            if args.savgol:
                                seg = apply_savgol_filter_rotvec(seg, window=args.savgol_window, polyorder=args.savgol_polyorder)
                            else:
                                seg = apply_one_euro_filter_2d(seg, mincutoff=0.5, beta=0.0, dcutoff=1.0)
                        else:
                            if args.outlier_rejection:
                                seg = reject_outliers_median_2d(seg, window=args.outlier_window, threshold=args.outlier_threshold)
                            if args.savgol:
                                seg = apply_savgol_filter_2d(seg, window=args.savgol_window, polyorder=args.savgol_polyorder)
                            else:
                                seg = apply_one_euro_filter_2d(seg, mincutoff=0.5, beta=0.0, dcutoff=1.0)
                        params[key][start:end] = seg
            if right_valid:
                smooth_params_per_segment(params_right, chosen_frames_right)
            if left_valid:
                smooth_params_per_segment(params_left, chosen_frames_left)

        # json dump mano
        manos_params = {}
        if left_valid:
            manos_params['left'] = {key: params_left[key].tolist() for key in params_left}
        if right_valid:
            manos_params['right'] = {key: params_right[key].tolist() for key in params_right}
        outhand_mano_params_path = f'{args.out_dir}/mano_params/{str(selected_vid_idx).zfill(3)}.json'
        os.makedirs(os.path.dirname(outhand_mano_params_path), exist_ok=True)
        with open(outhand_mano_params_path, "w") as f:
            ujson.dump(manos_params, f)
        
        if args.vis_smpl or args.save_mesh or args.vis_2d_repro or args.vis_3d_repro:
            # save paths
            # if args.vis_smpl:
                # if not args.save_frame:
                #     os.makedirs(f'{args.out_dir}/mano', exist_ok=True)
                #     outhand_mano_path = f'{args.out_dir}/vis/mano/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                # outhand_mano_path = f'{args.out_dir}/vis/mano/{str(selected_vid_idx).zfill(3)}'
                # os.makedirs(outhand_mano_path, exist_ok=True)
            if args.vis_2d_repro:
                # if not args.save_frame:
                    # os.makedirs(f'{args.out_dir}/repro_2d', exist_ok=True)
                    # outhand_2d_path = f'{args.out_dir}/vis/repro_2d/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                outhand_2d_path = f'{args.out_dir}/vis/repro_2d/{str(selected_vid_idx).zfill(3)}'
                os.makedirs(outhand_2d_path, exist_ok=True)
            if args.vis_3d_repro:
                # if not args.save_frame:
                #     os.makedirs(f'{args.out_dir}/repro_3d', exist_ok=True)
                #     outhand_3d_path = f'{args.out_dir}/vis/repro_3d/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                outhand_3d_path = f'{args.out_dir}/vis/repro_3d/{str(selected_vid_idx).zfill(3)}'
                os.makedirs(outhand_3d_path, exist_ok=True)
            # if not args.save_frame:
            #     os.makedirs(f'{args.out_dir}/regress_joints', exist_ok=True)
            #     outjoint_3d_path = f'{args.out_dir}/vis/regress_joints/{str(selected_vid_idx).zfill(3)}.mp4'
            # else:
            outjoint_3d_path = f'{args.out_dir}/vis/regress_joints/{str(selected_vid_idx).zfill(3)}'
            os.makedirs(outjoint_3d_path, exist_ok=True)

            # scale
            # nf = 0
            # final_scale_right = math.inf
            # final_scale_left = math.inf
            # for abs_idx in range(len(chosen_frames)):
            #     param_right = select_nf(params_right, nf)
            #     param_left = select_nf(params_left, nf)
            #     joints_right = body_model_right(return_verts=False, return_tensor=False, **param_right)
            #     joints_left = body_model_left(return_verts=False, return_tensor=False, **param_left)
            #     if abs_idx % args.stride == 0:
            #         scale_right = compute_scale_factor(keypoints3d_right[abs_idx], joints_right[0])
            #         scale_left = compute_scale_factor(keypoints3d_left[abs_idx], joints_left[0])
            #         if abs(scale_right - 1.0) < abs(final_scale_right - 1.0):
            #             final_scale_right = scale_right
            #         if abs(scale_left - 1.0) < abs(final_scale_left - 1.0):
            #             final_scale_left = scale_left
            #     nf += 1
                    
            # Map absolute frame index → position in per-hand params
            # (params_right/left have N_per_hand frames, not N_total)
            frame_to_right_nf = {f: i for i, f in enumerate(chosen_frames_right)} if right_valid else {}
            frame_to_left_nf = {f: i for i, f in enumerate(chosen_frames_left)} if left_valid else {}

            nf = 0
            if not use_parsed:
                generator = reader(chosen_frames)
            for abs_idx, chosen_f in tqdm(enumerate(chosen_frames)):
                if use_parsed:
                    multiseq_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence)
                    parsed_dir = os.path.join(multiseq_dir, "parsed")
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
                
                # undistort images for visualization
                images = []
                c_idx = 0
                for cam in cur_cam_names:
                    if cam in cam_mapper:
                        image = frames[cam_mapper[cam]]
                        if args.undistort and not (dists[c_idx] == 0).all():
                            image = param_utils.undistort_image(intrs[c_idx], dist_intrs[c_idx], dists[c_idx], image)
                        c_idx += 1
                        images.append(image)

                nf_right = frame_to_right_nf.get(chosen_f)
                nf_left = frame_to_left_nf.get(chosen_f)
                param_right = select_nf(params_right, nf_right) if right_valid and nf_right is not None else None
                param_left = select_nf(params_left, nf_left) if left_valid and nf_left is not None else None
                root_right = keypoints3d_right[abs_idx][0, :3] if param_right is not None else None
                root_left = keypoints3d_left[abs_idx][0, :3] if param_left is not None else None
                if abs_idx % args.stride == 0:
                    # visualize mano
                    if args.vis_smpl:
                        # mano parameters -> mesh
                        if param_right is not None:
                            vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)
                            vertices_right_scaled = (vertices_right[0] - root_right) * final_scale_right + root_right
                        if param_left is not None:
                            vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)
                            vertices_left_scaled = (vertices_left[0] - root_left) * final_scale_left + root_left
                        # project the mesh to image
                        if param_right is not None and param_left is not None:
                            vertices = np.concatenate((vertices_left_scaled, vertices_right_scaled), axis=0)
                            faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left_scaled.shape[0]), axis=0)
                        elif param_right is not None:
                            vertices = vertices_right_scaled
                            faces = body_model_right.faces
                        else:
                            vertices = vertices_left_scaled
                            faces = body_model_left.faces
                        image_vis, render_results = vis_smpl(
                            args, vertices=vertices, faces=faces, images=images,
                            nf=nf, cameras=cameras, add_back=True, out_dir="",
                            confident=confident, save_frames=False
                        )
                        # if args.vis_smpl:
                        #     if abs_idx == 0:
                        #         outhand_mano = create_video_writer(outhand_mano_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        #     outhand_mano.write(image_vis)

                    # save the mesh as obj
                    if args.save_mesh:
                        if param_right is not None and param_left is not None:
                            vertices = np.concatenate((vertices_left_scaled, vertices_right_scaled), axis=0)
                            faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left_scaled.shape[0]), axis=0)
                        elif param_right is not None:
                            vertices = vertices_right_scaled
                            faces = body_model_right.faces
                        else:
                            vertices = vertices_left_scaled
                            faces = body_model_left.faces
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        outdir = os.path.join(args.out_dir, f'vis/meshes/{str(selected_vid_idx).zfill(3)}')
                        os.makedirs(outdir, exist_ok=True)
                        outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                        mesh.export(outname)

                    # project the 3D keypoints to image
                    vis_config = CONFIG['handlr']
                    if args.vis_3d_repro:
                        if nf_right is not None and nf_left is not None:
                            keypoints = np.concatenate((keypoints3d_right[abs_idx], keypoints3d_left[abs_idx]), axis=0)
                        elif nf_right is not None:
                            keypoints = keypoints3d_right[abs_idx]
                        else:
                            keypoints = keypoints3d_left[abs_idx]
                        kpts_repro = projectN3(keypoints, projs)
                        kpts_repro[:, :, 2] = 0.5
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path, cameras=cameras, confident=confident)
                        # image_vis = cv2.addWeighted(image_vis, 0.7, image_kps, 0.3, 0)
                        if abs_idx == 0:
                            outhand_3d = create_video_writer(outhand_3d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_3d.write(image_vis)

                    # if args.vis_regressed_joints:
                    if param_right is not None:
                        joints_right = body_model_right(return_verts=False, return_tensor=False, **param_right)
                        joints_right = (joints_right[0] - root_right) * final_scale_right + root_right
                    if param_left is not None:
                        joints_left = body_model_left(return_verts=False, return_tensor=False, **param_left)
                        joints_left = (joints_left[0] - root_left) * final_scale_left + root_left
                    if param_right is not None and param_left is not None:
                        joints = np.concatenate((joints_left, joints_right), axis=0)
                    elif param_right is not None:
                        joints = joints_right
                    else:
                        joints = joints_left
                    joints_repro = projectN3(joints, projs)
                    joints_repro[:, :, 2] = 0.5
                    image_vis = vis_repro(args, render_results, joints_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outjoint_3d_path, cameras=cameras, confident=confident)
                    if abs_idx == 0:
                        outjoint_3d = create_video_writer(outjoint_3d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                    outjoint_3d.write(image_vis)

                    # overlay the 2D keypoints to image
                    if args.vis_2d_repro:
                        if right_valid and left_valid:
                            keypoints2d = np.concatenate((all_keypoints2d_right[abs_idx], all_keypoints2d_left[abs_idx]), axis=1)
                        elif right_valid:
                            keypoints2d = all_keypoints2d_right[abs_idx]
                        else:
                            keypoints2d = all_keypoints2d_left[abs_idx]
                        kpts_repro = keypoints2d
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_2d_path, cameras=cameras, confident=confident)
                        if abs_idx == 0:
                            outhand_2d = create_video_writer(outhand_2d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_2d.write(image_vis)
                nf += 1

            # save as video
            # if not args.save_frame:
            # if args.vis_smpl:
            #     outhand_mano.release()
            #     convert_video_ffmpeg(outhand_mano_path+".mp4")
            #     print('Video Handler Released')
            if args.vis_2d_repro:
                outhand_2d.release()
                convert_video_ffmpeg(outhand_2d_path+".mp4")
                print('Video Handler Released')
            if args.vis_3d_repro:
                outhand_3d.release()
                convert_video_ffmpeg(outhand_3d_path+".mp4")
                print('Video Handler Released')
            outjoint_3d.release()
            convert_video_ffmpeg(outjoint_3d_path+".mp4")
            print('Video Handler Released')
