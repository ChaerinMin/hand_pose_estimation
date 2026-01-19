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
from src.utils.fingers import compute_scale_factor
from easymocap.dataset import CONFIG
from easymocap.mytools import Timer
from easymocap.pipeline import smpl_from_keypoints3d, smpl_from_keypoints3d2d
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
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
parser.add_argument("--refine_shape_with_mask", action="store_true", help="Refine MANO shape parameters using hand masks")
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


# root paths
if args.optimize_bad_views: 
    params_txt = "new_params.txt"
elif args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"
base_path = os.path.join(args.root_dir)
image_dir = os.path.join(base_path, args.seq_path)
output_path = args.out_dir
params_path = os.path.join(output_path, params_txt)
assert os.path.exists(params_path)

# filter out some cameras
if "stage1" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=True)
    use_parsed = False
elif "stage2" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=False)
    use_parsed = True
else:
    raise ValueError("Cannot determine whether to assume undistorted.")
cam_names = list(params[:]["cam_name"])
removed_camera_path = os.path.join(output_path, 'ignore_camera.txt')
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
    conf_dir = args.out_dir[:args.out_dir.index("/stage")]
    conf_path = os.path.join(conf_dir, "image_confidence.json")
    with open(conf_path, "r") as f:
        image_confidence = ujson.load(f)
    confident = {}
    for k, v in image_confidence.items():
        confident[k.replace(".jpg", "")] = v["num_visible_3D_points"] >= float(args.confidence_thresh)
else:
    confident = None

for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')
    # read video
    reader = Reader(
        "video",
        image_dir,
        cam_names=cam_names,
        cams_to_remove=cams_to_remove,
        ith=selected_vid_idx,
        anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera
    )
    if reader.frame_count <= 0:
        continue
    
    # 2d/3d keypoint paths
    keypoints2d_dir_right = os.path.join(output_path, "keypoints_2d", "right", str(selected_vid_idx).zfill(3))
    keypoints2d_dir_left = os.path.join(output_path, "keypoints_2d", "left",  str(selected_vid_idx).zfill(3))
    bboxes_dir_right = os.path.join(output_path, "bboxes", "right",  str(selected_vid_idx).zfill(3))
    bboxes_dir_left = os.path.join(output_path, "bboxes", "left",  str(selected_vid_idx).zfill(3))
    keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(selected_vid_idx).zfill(3))
    keypt3d_file_left = os.path.join(keypoints3d_dir, "left.jsonl")
    keypt3d_file_right = os.path.join(keypoints3d_dir, "right.jsonl")

    # fileter out some frames
    if args.use_filtered:
        chosen_path_left = os.path.join(keypoints3d_dir, "chosen_frames_left.json")
        chosen_path_right = os.path.join(keypoints3d_dir, "chosen_frames_right.json")
        with open(chosen_path_right, "r") as f:
            chosen_frames_right = set(json.load(f))
        with open(chosen_path_left, "r") as f:
            chosen_frames_left = set(json.load(f))
        chosen_frames =list(set(chosen_frames_right | chosen_frames_left))
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
    keypoints3d_right, keypoints3d_left = [], []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                keypoints3d_left.append(np.array(ujson.loads(linel)).reshape(-1, 4))
                keypoints3d_right.append(np.array(ujson.loads(liner)).reshape(-1, 4))
    keypoints3d_left = np.asarray(keypoints3d_left)
    keypoints3d_right = np.asarray(keypoints3d_right)

    # load hand masks if refine_shape_with_mask is enabled
    hand_masks = None
    seg_status = {}
    if args.refine_shape_with_mask:
        mask_dir = os.path.join(output_path, "mask_2d", str(selected_vid_idx).zfill(3))
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
                multiseq_dir = args.out_dir[:args.out_dir.index("calib")]
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
    if len(keypoints3d_right.shape) == 3:
        # smooth keypoints
        if args.to_smooth:
            print('Smoothing Keypoints 3D...')
            keypoints3d_left[:3] = apply_one_euro_filter_3d(keypoints3d_left[:3], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            keypoints3d_right[:3] = apply_one_euro_filter_3d(keypoints3d_right[:3], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)

        # keypoints -> mano parameters
        weight_pose = {
            'k3d': 1e2, 'k2d': 2e-3,
            # 'reg_poses': 1e-3, 'smooth_body': 1e2, 'smooth_poses': 1e2,
            'reg_poses': 5e-5, 'smooth_body': 1e1, 'smooth_poses': 5.0,
            # 'reg_poses': 0.0, 'smooth_body': 0.0, 'smooth_poses': 0.0
        }
        # Estimate global scale for observed keypoints -> model units and create scaled copies
        s_right = estimate_scale_from_keypoints(body_model_right, keypoints3d_right, kintree=dataset_config.get('kintree', None))
        s_left = estimate_scale_from_keypoints(body_model_left, keypoints3d_left, kintree=dataset_config.get('kintree', None))
        keypoints3d_right_scaled = apply_scale_to_keypoints(keypoints3d_right, s_right)
        keypoints3d_left_scaled = apply_scale_to_keypoints(keypoints3d_left, s_left)
        final_scale_right = 1.0 / s_right
        final_scale_left = 1.0 / s_left
        fit_3d2d = False
        if fit_3d2d:
            params_right = smpl_from_keypoints3d2d(
                body_model_right, keypoints3d_right_scaled, all_keypoints2d_right, all_bboxes_right, projs, 
                config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose
            )
            params_left = smpl_from_keypoints3d2d(
                body_model_left, keypoints3d_left_scaled, all_keypoints2d_left, all_bboxes_left, projs, 
                config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose
            )
        else:
            params_right = smpl_from_keypoints3d(body_model_right, keypoints3d_right_scaled,
                config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 1e2}, weight_pose=weight_pose) # 5e3
            params_left = smpl_from_keypoints3d(body_model_left, keypoints3d_left_scaled,
                config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 1e2}, weight_pose=weight_pose)

        # refine shape with hand masks if enabled
        if args.refine_shape_with_mask and hand_masks is not None:
            print('Refining MANO shape parameters with hand masks...')
            from src.utils.mask_optimize import refine_shape_with_mask

            # Refine right hand
            params_right = refine_shape_with_mask(
                body_model_right, params_right, hand_masks, cameras,
                cur_cam_names, cam_mapper, intrs, final_scale_right, keypoints3d_right[0], hand_side="right",
                # weight_loss={'mask': 1e4, 'reg_shapes': 1e1, 'init_shape': 5e1},  # 1e3 1e2 5e2
                weight_loss={'mask': 1e4, 'reg_shapes': 0.5, 'init_shape': 0.0},  # 1e3 1e2 5e2
                max_iter=20, verbose=True
            )

            # Refine left hand
            params_left = refine_shape_with_mask(
                body_model_left, params_left, hand_masks, cameras,
                cur_cam_names, cam_mapper, intrs, final_scale_left, keypoints3d_left[0], hand_side="left",
                weight_loss={'mask': 1e4, 'reg_shapes': 0.5, 'init_shape': 0.0},
                max_iter=20, verbose=True
            )

        # smooth mano parameters
        if args.to_smooth:
            print('Smoothing Manos...')
            params_right['Rh'] = apply_one_euro_filter_2d(params_right['Rh'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['Rh'] = apply_one_euro_filter_2d(params_left['Rh'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_right['Th'] = apply_one_euro_filter_2d(params_right['Th'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['Th'] = apply_one_euro_filter_2d(params_left['Th'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_right['poses'] = apply_one_euro_filter_2d(params_right['poses'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['poses'] = apply_one_euro_filter_2d(params_left['poses'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)

        # json dump mano
        manos_params = {}
        params_left_list = {}
        params_right_list = {}
        for key in params_left:
            params_left_list[key] = params_left[key].tolist()
        for key in params_right:
            params_right_list[key] = params_right[key].tolist()
        manos_params['left'] = params_left_list
        manos_params['right'] = params_right_list
        outhand_mano_params_path = f'{output_path}/params/{str(selected_vid_idx).zfill(3)}.json'
        os.makedirs(os.path.dirname(outhand_mano_params_path), exist_ok=True)
        with open(outhand_mano_params_path, "w") as f:
            ujson.dump(manos_params, f)
            
        if args.vis_smpl or args.save_mesh or args.vis_2d_repro or args.vis_3d_repro:
            # save paths
            if args.vis_smpl:
                # if not args.save_frame:
                #     os.makedirs(f'{output_path}/mano', exist_ok=True)
                #     outhand_mano_path = f'{output_path}/mano/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                outhand_mano_path = f'{output_path}/mano/{str(selected_vid_idx).zfill(3)}'
                os.makedirs(outhand_mano_path, exist_ok=True)
            if args.vis_2d_repro:
                # if not args.save_frame:
                    # os.makedirs(f'{output_path}/repro_2d', exist_ok=True)
                    # outhand_2d_path = f'{output_path}/repro_2d/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                outhand_2d_path = f'{output_path}/repro_2d/{str(selected_vid_idx).zfill(3)}'
                os.makedirs(outhand_2d_path, exist_ok=True)
            if args.vis_3d_repro:
                # if not args.save_frame:
                #     os.makedirs(f'{output_path}/repro_3d', exist_ok=True)
                #     outhand_3d_path = f'{output_path}/repro_3d/{str(selected_vid_idx).zfill(3)}.mp4'
                # else:
                outhand_3d_path = f'{output_path}/repro_3d/{str(selected_vid_idx).zfill(3)}'
                os.makedirs(outhand_3d_path, exist_ok=True)
            # if not args.save_frame:
            #     os.makedirs(f'{output_path}/regress_joints', exist_ok=True)
            #     outjoint_3d_path = f'{output_path}/regress_joints/{str(selected_vid_idx).zfill(3)}.mp4'
            # else:
            outjoint_3d_path = f'{output_path}/regress_joints/{str(selected_vid_idx).zfill(3)}'
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
                    
            nf = 0
            if not use_parsed:
                generator = reader(chosen_frames)
            for abs_idx, chosen_f in tqdm(enumerate(chosen_frames)):
                if use_parsed:
                    multiseq_dir = args.out_dir[:args.out_dir.index("calib")]
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

                param_right = select_nf(params_right, nf)
                param_left = select_nf(params_left, nf)
                if abs_idx % args.stride == 0:
                    # visualize mano
                    if args.vis_smpl:
                        # mano parameters -> mesh
                        vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)
                        vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)
                        # scaling
                        # root_right = param_right['Th'].reshape(3)
                        # root_left = param_left['Th'].reshape(3)
                        root_right = keypoints3d_right[abs_idx][0, :3]
                        root_left = keypoints3d_left[abs_idx][0, :3]
                        vertices_right_scaled = (vertices_right[0] - root_right) * final_scale_right + root_right
                        vertices_left_scaled = (vertices_left[0] - root_left) * final_scale_left + root_left
                        # vertices_right_scaled = vertices_right[0]
                        # vertices_left_scaled = vertices_left[0]
                        # project the mesh to image
                        vertices = np.concatenate((vertices_left_scaled, vertices_right_scaled), axis=0)
                        faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left_scaled.shape[0]), axis=0)
                        image_vis, render_results = vis_smpl(
                            args, vertices=vertices, faces=faces, images=images, 
                            nf=nf, cameras=cameras, add_back=True, out_dir=outhand_mano_path,
                            confident=confident
                        )
                        if abs_idx == 0:
                            outhand_mano = create_video_writer(outhand_mano_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_mano.write(image_vis)

                    # save the mesh as obj
                    if args.save_mesh:
                        vertices = np.concatenate((vertices_left_scaled, vertices_right_scaled), axis=0)
                        faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left_scaled.shape[0]), axis=0)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        outdir = os.path.join(output_path, f'meshes/{str(selected_vid_idx).zfill(3)}')
                        os.makedirs(outdir, exist_ok=True)
                        outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                        mesh.export(outname)

                    # project the 3D keypoints to image
                    vis_config = CONFIG['handlr']
                    if args.vis_3d_repro:
                        keypoints = np.concatenate((keypoints3d_right[abs_idx], keypoints3d_left[abs_idx]), axis=0)
                        kpts_repro = projectN3(keypoints, projs)
                        kpts_repro[:, :, 2] = 0.5
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path, cameras=cameras, confident=confident)
                        # image_vis = cv2.addWeighted(image_vis, 0.7, image_kps, 0.3, 0)
                        if abs_idx == 0:
                            outhand_3d = create_video_writer(outhand_3d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_3d.write(image_vis)

                    vis_regressed_joints = True
                    if vis_regressed_joints:
                        joints_right = body_model_right(return_verts=False, return_tensor=False, **param_right)
                        joints_left = body_model_left(return_verts=False, return_tensor=False, **param_left)
                        joints_right = (joints_right[0] - root_right) * final_scale_right + root_right
                        joints_left = (joints_left[0] - root_left) * final_scale_left + root_left
                        joints = np.concatenate((joints_left, joints_right), axis=0)
                        joints_repro = projectN3(joints, projs)
                        joints_repro[:, :, 2] = 0.5
                        image_vis = vis_repro(args, render_results, joints_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outjoint_3d_path, cameras=cameras, confident=confident)
                        if abs_idx == 0:
                            outjoint_3d = create_video_writer(outjoint_3d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outjoint_3d.write(image_vis)

                    # overlay the 2D keypoints to image
                    if args.vis_2d_repro:
                        keypoints2d = np.concatenate((all_keypoints2d_right[abs_idx], all_keypoints2d_left[abs_idx]), axis=1)
                        kpts_repro = keypoints2d
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_2d_path, cameras=cameras, confident=confident)
                        if abs_idx == 0:
                            outhand_2d = create_video_writer(outhand_2d_path+".mp4", (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_2d.write(image_vis)
                nf += 1

            # save as video
            # if not args.save_frame:
            if args.vis_smpl:
                outhand_mano.release()
                convert_video_ffmpeg(outhand_mano_path+".mp4")
                print('Video Handler Released')
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
