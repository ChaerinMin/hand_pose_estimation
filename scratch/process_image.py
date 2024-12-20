import os
import cv2
import sys
import json
import ujson
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils.reader_v2 import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.easymocap_utils import vis_smpl, projectN3, vis_repro, load_model
from src.utils.filter import apply_one_euro_filter_2d, apply_one_euro_filter_3d
from src.utils.video_handler import create_video_writer, convert_video_ffmpeg
sys.path.append("./third-party/EasyMocap")
from easymocap.mytools import Timer
from easymocap.dataset import CONFIG
from easymocap.pipeline import smpl_from_keypoints3d2d, smpl_from_keypoints3d
from easymocap.smplmodel import check_keypoints, select_nf

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'





# -------------------- Visualization Functions -------------------- #
# parser = load_parser()
parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--to_smooth", action="store_true", help="Whether to temporally smoothing the result")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
# Easy Mocap Arguments
parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'handlr', 'total'])
parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
parser.add_argument('--save_origin', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--opts', help="Modify config options using the command-line", 
    default={}, nargs='+')
# optimization control
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
args = parser.parse_args()

params_txt = "optim_params.txt"


base_path = os.path.join(args.root_dir)
image_dir = os.path.join(base_path, args.seq_path)
output_path = args.out_dir
params_path = os.path.join(output_path, params_txt)

params = param_utils.read_params(params_path)
cam_names = list(params[:]["cam_name"])
removed_camera_path = os.path.join(output_path, 'ignore_camera.txt')
if os.path.isfile(removed_camera_path):
    with open(removed_camera_path) as file:
        ignored_cameras = [line.rstrip() for line in file]
else:
    ignored_cameras = None
cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras)
cams_to_remove += ['brics-odroid-001_cam0', 'brics-odroid-001_cam1', 'brics-odroid-002_cam0', 'brics-odroid-007_cam0', 'brics-odroid-007_cam1', 'brics-odroid-008_cam0', 'brics-odroid-010_cam0', 'brics-odroid-010_cam1', 'brics-odroid-011_cam0', 'brics-odroid-012_cam1', 'brics-odroid-014_cam1', 'brics-odroid-015_cam0', 'brics-odroid-015_cam1', 'brics-odroid-016_cam0', 'brics-odroid-017_cam0', 'brics-odroid-017_cam1', 'brics-odroid-019_cam1', 'brics-odroid-020_cam0', 'brics-odroid-020_cam1', 'brics-odroid-021_cam0', 'brics-odroid-021_cam1', 'brics-odroid-022_cam0', 'brics-odroid-022_cam1', 'brics-odroid-023_cam0', 'brics-odroid-024_cam0', 'brics-odroid-024_cam1', 'brics-odroid-025_cam0', 'brics-odroid-026_cam0', 'brics-odroid-026_cam1', 'brics-odroid-027_cam0', 'brics-odroid-027_cam1', 'brics-odroid-028_cam0', 'brics-odroid-029_cam0', 'brics-odroid-029_cam1', 'brics-odroid-030_cam0', 'brics-odroid-030_cam1']
cams_to_remove += ['brics-odroid-012_cam0', 'brics-odroid-006_cam0']

for cam in cams_to_remove:
    if cam in cam_names:
        cam_names.remove(cam)


selected_vid_idxs = [args.ith]

for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')
    reader = Reader("video", image_dir, cams_to_remove=cams_to_remove, ith=selected_vid_idx, anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera)
    if reader.frame_count <= 0:
        continue
    
    keypoints2d_dir_right = os.path.join(output_path, "keypoints_2d", "right", str(selected_vid_idx).zfill(3))
    keypoints2d_dir_left = os.path.join(output_path, "keypoints_2d", "left",  str(selected_vid_idx).zfill(3))
    bboxes_dir_right = os.path.join(output_path, "bboxes", "right",  str(selected_vid_idx).zfill(3))
    bboxes_dir_left = os.path.join(output_path, "bboxes", "left",  str(selected_vid_idx).zfill(3))
    keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(selected_vid_idx).zfill(3))
    keypt3d_file_left = os.path.join(keypoints3d_dir, "left.jsonl")
    keypt3d_file_right = os.path.join(keypoints3d_dir, "right.jsonl")

    # loads the selected frames
    chosen_path_left = os.path.join(keypoints3d_dir, f"chosen_frames_left.json")
    chosen_path_right = os.path.join(keypoints3d_dir, f"chosen_frames_right.json")
    with open(chosen_path_right, "r") as f:
        chosen_frames_right = set(json.load(f))
    with open(chosen_path_left, "r") as f:
        chosen_frames_left = set(json.load(f))
    chosen_frames =list(set(chosen_frames_right | chosen_frames_left))

    chosen_frames = sorted(chosen_frames)
    print(f"Total valid frames {len(chosen_frames)}/{reader.frame_count}")
    
    cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)

    # cam_mapper = {
    #     'brics-odroid-006_cam0': 'brics-odroid-006_cam0_1718718123013140',
    #     'brics-odroid-012_cam0': 'brics-odroid-012_cam0_1718718123013412',
    #     'brics-odroid-025_cam1': 'brics-odroid-025_cam1_1718718123056094'
    # }
    extra_cams_to_remove = reader.to_delete
    cur_cam_names = cam_names.copy()
    for cam in extra_cams_to_remove:
        if cam in cur_cam_names:
            cur_cam_names.remove(cam)
    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cur_cam_names, cam_mapper, easymocap_format=True)

    # loads 2d & 3d keypoints
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
            with open(ap_keypoints_path_left, "r") as fl, open(ap_keypoints_path_right, "r") as fr, open(bboxes_path_left, "r") as fbl, open(bboxes_path_right, "r") as fbr:
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
        all_keypoints2d_left[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_left[nf, :, :, :2], cameras)
    for nf in range(all_keypoints2d_right.shape[0]):
        all_keypoints2d_right[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_right[nf, :, :, :2], cameras)
        
    keypoints3d_right, keypoints3d_left = [], []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                keypoints3d_left.append(np.array(ujson.loads(linel)).reshape(-1, 4))
                keypoints3d_right.append(np.array(ujson.loads(liner)).reshape(-1, 4))
    keypoints3d_left = np.asarray(keypoints3d_left)
    keypoints3d_right = np.asarray(keypoints3d_right)
    
    # fits the mano model
    dataset_config = CONFIG[args.body]

    output_path = '/users/rfu7/data/code/24Text2Action/data_analysis/video_visualization/render_video'
    outhand_2d_path = f'{output_path}/repro_2d/{str(selected_vid_idx).zfill(3)}'
    outhand_3d_path = f'{output_path}/repro_3d/{str(selected_vid_idx).zfill(3)}'
    os.makedirs(outhand_2d_path, exist_ok=True)
    os.makedirs(outhand_3d_path, exist_ok=True)
    
    if len(keypoints3d_right.shape) == 3:
        nf = 0
        for abs_idx, (frames, idx) in tqdm(enumerate(reader(chosen_frames)), total=len(chosen_frames)):
            images = []
            c_idx = 0
            for cam in cur_cam_names:
                if cam in cam_mapper:
                    image = frames[cam_mapper[cam]]
                    if args.undistort:
                        image = param_utils.undistort_image(intrs[c_idx], dist_intrs[c_idx], dists[c_idx], image)
                        c_idx += 1
                    images.append(image)
        
                    
            # visualize keypoint reprojection
            vis_config = CONFIG['handlr']
            if args.vis_2d_repro:
                keypoints2d = np.concatenate((all_keypoints2d_right[abs_idx], all_keypoints2d_left[abs_idx]), axis=1)
                kpts_repro = keypoints2d
                if nf == 120:
                    image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_2d_path)
                    outname = os.path.join(outhand_2d_path, 'v3_2d.png')
                    cv2.imwrite(outname, image_vis)
                
            if args.vis_3d_repro:
            #     keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
                keypoints = np.concatenate((keypoints3d_right[abs_idx], keypoints3d_left[abs_idx]), axis=0)
                kpts_repro = projectN3(keypoints, projs)
                kpts_repro[:, :, 2] = 0.5
                
                if nf == 120:
                    image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path)
                    outname = os.path.join(outhand_2d_path, 'v3_3d.png')
                    cv2.imwrite(outname, image_vis)
            nf += 1