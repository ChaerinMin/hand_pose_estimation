"""
Immediate proceeder: keypoints_2d_yolo_vitpose.py
This file: get hand mask, conditioned on 2d keypoints and bbox.
    This will be used to optimize MANO shape parameters at mano_em.py
Immediate successor: keypoints_3d_fast.py
"""

import argparse
import gc
import glob
import os
import shutil
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
from tqdm import tqdm
import ujson

from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.parser import add_common_args
import src.utils.params as param_utils
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess
from easymocap.mytools.vis_base import merge, get_row_col

def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")
    
parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument(
    "--to_smooth", action="store_true",
    help="Whether to temporally smoothing the result"
)
parser.add_argument("--all_frames", default=False, action="store_true")
parser.add_argument(
    "--easymocap", default=False, action="store_true",
    help='use Easymocap for triangulation'
)
parser.add_argument(
    '--remove_side_cam', type=bool, default=True, help='Remove Side Cameras'
)
parser.add_argument(
    '--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras'
)
parser.add_argument("--ignore_missing_tip", action="store_true", help="Should a missing fingertip be allowed")
parser.add_argument(
    "--sam_path",
    type=str,
    default="/oscar/data/ssrinath/datasets/sam3/sam3.pt",
    help="Path to sam.pt"
)
parser.add_argument(
    "--v_idx", type=int, default=None,
    help="If you want to process a specific video index"
)
parser.add_argument(
    "--collage_only", action="store_true",
    help="If true, only create the collage video from existing per-camera videos"
)
args = parser.parse_args()

# paths
base_path = os.path.join(args.root_dir)
image_base = os.path.join(base_path, args.seq_path)
output_path = args.out_dir

# load cameras
if args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"
params_path = os.path.join(args.out_dir, params_txt)
if "stage1" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=True)
    use_parsed = False
elif "stage2" in args.out_dir:
    params = param_utils.read_params(params_path, distortion=False)
    use_parsed = True
else:
    raise ValueError("Cannot determine whether to assume undistorted.")

# remove cameras
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

# which video
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
    keypoints2d_dir_right = os.path.join(
        output_path, "keypoints_2d", "right", str(selected_vid_idx).zfill(3)
    )
    keypoints2d_dir_left = os.path.join(
        output_path, "keypoints_2d", "left",  str(selected_vid_idx).zfill(3)
    )
    cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)

    # image paths
    reader = Reader(
        args.input_type, image_base, cam_names=cam_names,
        cams_to_remove=cams_to_remove, ith=selected_vid_idx,
        anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera
    )
    if reader.frame_count <= 0:
        continue

    # delete paths
    extra_cams_to_remove = reader.to_delete
    cur_cam_names = cam_names.copy()
    for cam in extra_cams_to_remove:
        if cam in cur_cam_names:
            cur_cam_names.remove(cam)
    print("Total Views:", len(cur_cam_names))
    print("Total frames:", reader.frame_count)
    
    # intrinsics / extrinsics
    intrs, projs, dist_intrs, dists, cameras = get_projections(
        args, params, cur_cam_names, cam_mapper, easymocap_format=True
    )
    print("Total cams:", len(intrs), len(dist_intrs), len(dists))
    print("Reader Length", len(reader.vids))

    # output dirs
    mask_left_dir = os.path.join(output_path, "mask_2d", "left", str(selected_vid_idx).zfill(3))
    mask_right_dir = os.path.join(output_path, "mask_2d", "right", str(selected_vid_idx).zfill(3))
    vis_dir = os.path.join(output_path, "mask_2d", "vis_"+str(selected_vid_idx).zfill(3))
    os.makedirs(mask_left_dir, exist_ok=True)
    os.makedirs(mask_right_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids)):
        if args.v_idx is not None and v_idx != args.v_idx:
            continue
        if args.collage_only:
            break
        exist_ids = []

        # read images
        im_names, orig_imgs, im_h, im_w = frame_preprocess(
            input_video_path, use_parsed, args, intrs[v_idx], dist_intrs[v_idx], dists[v_idx]
        )
        orig_images = []
        for l_idx, orig_img in enumerate(orig_imgs):
            if orig_img is not None:
                exist_ids.append(l_idx)
                orig_images.append(Image.fromarray(orig_img))

        # read 2D keypoints
        camname = input_video_path.split('/')[-2]
        ap_keypoints_path_left = os.path.join(
            keypoints2d_dir_left, f"{cam_mapper[camname]}.jsonl"
        )
        ap_keypoints_path_right = os.path.join(
            keypoints2d_dir_right, f"{cam_mapper[camname]}.jsonl"
        )
        keypoints2d_left = []
        keypoints2d_right = []
        with open(ap_keypoints_path_left, "r") as fl, open(ap_keypoints_path_right, "r") as fr:
            for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
                if l_idx in set(exist_ids):
                    keypoints2d_left.append(np.array(ujson.loads(linel)).reshape(-1, 3))
                    keypoints2d_right.append(np.array(ujson.loads(liner)).reshape(-1, 3))
        keypoints2d_left = np.array(keypoints2d_left)  # (frames, 21, 3)
        keypoints2d_right = np.array(keypoints2d_right)

        # keypoint validity
        valid_left = np.logical_not(
            (keypoints2d_left[:, :, 0] == 0) &\
                 (keypoints2d_left[:, :, 1] == 0) &\
                     (keypoints2d_left[:, :, 2] == 1)
        )
        valid_left = np.logical_and(
            valid_left,
            (keypoints2d_left[:, :, 0] < im_w) & (keypoints2d_left[:, :, 0] >= 0) & 
            (keypoints2d_left[:, :, 1] < im_h) & (keypoints2d_left[:, :, 1] >= 0)
        ) # (frames, 21)
        valid_right = np.logical_not(
            (keypoints2d_right[:, :, 0] == 0) &\
                 (keypoints2d_right[:, :, 1] == 0) &\
                     (keypoints2d_right[:, :, 2] == 1)
        )
        valid_right = np.logical_and(
            valid_right,
            (keypoints2d_right[:, :, 0] < im_w) & (keypoints2d_right[:, :, 0] >= 0) & 
            (keypoints2d_right[:, :, 1] < im_h) & (keypoints2d_right[:, :, 1] >= 0)
        ) # (frames, 21)

        # save paths
        mask_left_path = os.path.join(mask_left_dir, f"{cam_mapper[camname]}.npz")
        mask_right_path = os.path.join(mask_right_dir, f"{cam_mapper[camname]}.npz")

        # when to give prompt
        prompt_frame_l = None
        prompt_frame_r = None
        for i in range(len(orig_images)):
            if valid_left[i].sum() > 0:
                prompt_frame_l = i
                break
        for i in range(len(orig_images)):
            if valid_right[i].sum() > 0:
                prompt_frame_r = i
                break

        # sam3: first frame
        video_predictor = build_sam3_video_predictor(
            checkpoint_path=args.sam_path
        )
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=orig_images,  # a JPEG folder, an MP4 video file, or a list of PIL Image objects
            )
        )
        session_id = response["session_id"]
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text="background",
            )
        )
        outputs_per_frame = propagate_in_video(video_predictor, session_id)
        if prompt_frame_l is not None:
            points_abs = keypoints2d_left[prompt_frame_l][valid_left[prompt_frame_l]][:, :2]  # (n_points, 2)
            xmin, ymin = points_abs.min(axis=0)
            xmax, ymax = points_abs.max(axis=0)
            padding_x = int(0.1 * (xmax - xmin))
            padding_y = int(0.1 * (ymax - ymin))
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(im_w - 1, xmax + padding_x)
            ymax = min(im_h - 1, ymax + padding_y)
            box = [xmin, ymin, xmax - xmin, ymax - ymin]
            points_tensor_l = torch.tensor(
                abs_to_rel_coords(points_abs.tolist(), im_w, im_h, coord_type="point"),
                dtype=torch.float32,
            )
            box_tensor_l = torch.tensor(
                abs_to_rel_coords([box], im_w, im_h, coord_type="box"),
                dtype=torch.float32,
            )
            labels = np.array([1] * points_tensor_l.shape[0])
            # if valid_left[prompt_frame_l][0]:
            #     labels[0] = 0  # wrist as negative point
            points_labels_tensor_l = torch.tensor(labels, dtype=torch.int32)
            box_labels = np.array([1] * box_tensor_l.shape[0])
            box_labels_tensor_l = torch.tensor(box_labels, dtype=torch.int32)
        if prompt_frame_r is not None:
            points_abs = keypoints2d_right[prompt_frame_r][valid_right[prompt_frame_r]][:, :2]
            xmin, ymin = points_abs.min(axis=0)
            xmax, ymax = points_abs.max(axis=0)
            padding_x = int(0.1 * (xmax - xmin))
            padding_y = int(0.1 * (ymax - ymin))
            xmin = max(0, xmin - padding_x)
            ymin = max(0, ymin - padding_y)
            xmax = min(im_w - 1, xmax + padding_x)
            ymax = min(im_h - 1, ymax + padding_y)
            box = [xmin, ymin, xmax - xmin, ymax - ymin]
            points_tensor_r = torch.tensor(
                abs_to_rel_coords(points_abs.tolist(), im_w, im_h, coord_type="point"),
                dtype=torch.float32,
            )
            box_tensor_r = torch.tensor(
                abs_to_rel_coords([box], im_w, im_h, coord_type="box"),
                dtype=torch.float32,
            )
            labels = np.array([1] * points_tensor_r.shape[0])
            points_labels_tensor_r = torch.tensor(labels, dtype=torch.int32)
            # if valid_right[prompt_frame_r][0]:
            #     labels[0] = 0  # wrist as negative point
            box_labels = np.array([1] * box_tensor_r.shape[0])
            box_labels_tensor_r = torch.tensor(box_labels, dtype=torch.int32)
        if prompt_frame_l is not None:
            if prompt_frame_r is not None:
                points_tensor = torch.cat([points_tensor_l, points_tensor_r], dim=0)
                neg_labels = torch.tensor([0] * points_tensor_r.shape[0], dtype=torch.int32)
                points_labels_tensor = torch.cat([points_labels_tensor_l, neg_labels], dim=0)
            else:
                points_tensor = points_tensor_l
                points_labels_tensor = points_labels_tensor_l
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_l,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    # bounding_boxes=box_tensor_l,
                    # bounding_box_labels=box_labels_tensor_l,
                    # text="hand"
                    obj_id=1,
                )
            )
        if prompt_frame_r is not None:
            if prompt_frame_l is not None:
                points_tensor = torch.cat([points_tensor_r, points_tensor_l], dim=0)
                neg_labels = torch.tensor([0] * points_tensor_l.shape[0], dtype=torch.int32)
                points_labels_tensor = torch.cat([points_labels_tensor_l, neg_labels], dim=0)
            else:
                points_tensor = points_tensor_r
                points_labels_tensor = points_labels_tensor_r
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_r,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    # bounding_boxes=box_tensor_r,
                    # bounding_box_labels=box_labels_tensor_r,
                    # text="hand",
                    obj_id=2,
                )
            )

        # sam3: propagate
        outputs_per_frame = propagate_in_video(video_predictor, session_id)
        vis_per_frame = prepare_masks_for_visualization(outputs_per_frame)

        # save
        save_mask_left = np.zeros((len(orig_imgs), im_h, im_w), dtype=bool)
        save_mask_right = np.zeros((len(orig_imgs), im_h, im_w), dtype=bool)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vis_path = os.path.join(vis_dir, f"{cam_mapper[camname]}.mp4")
        video_writer = cv2.VideoWriter(vis_path, fourcc, 30, (600, 400))
        n_written = 0
        for frame_idx in tqdm(range(len(outputs_per_frame)), desc="Saving "):
            obj_ids = list(outputs_per_frame[frame_idx].keys())
            if 1 in obj_ids:
                mask_left = outputs_per_frame[frame_idx][1]
                save_mask_left[exist_ids[frame_idx], ...] = mask_left
            if 2 in obj_ids:
                mask_right = outputs_per_frame[frame_idx][2]
                save_mask_right[exist_ids[frame_idx], ...] = mask_right
            # visualize
            fig = visualize_formatted_frame_output(
                frame_idx,
                orig_images,
                outputs_list=[vis_per_frame],
                figsize=(6, 4),
            )
            while n_written < exist_ids[frame_idx]:
                video_writer.write(np.ones((400, 600, 3), dtype=np.uint8)*255)
                n_written += 1
            fig.canvas.draw()
            video_writer.write(np.array(fig.canvas.buffer_rgba())[..., :3][..., ::-1])
            fig.clf()
            plt.close(fig)
            n_written += 1
        np.savez_compressed(mask_left_path, *save_mask_left)
        np.savez_compressed(mask_right_path, *save_mask_right)
        print(f"Saved {mask_left_path} and {mask_right_path}")

        # clean up
        _ = video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        del video_predictor
        gc.collect()
        torch.cuda.empty_cache()
        video_writer.release()
    
    # collage
    if args.v_idx is None or args.collage_only:
        vis_paths = sorted(glob.glob(os.path.join(vis_dir, "brics*.mp4")))
        video_handlers = [cv2.VideoCapture(pth) for pth in vis_paths]
        for v_idx, v_handler in enumerate(video_handlers):
            if not v_handler.isOpened():
                raise IOError(f"Cannot open video {vis_paths[v_idx]}")
        collage_path = os.path.join(
            output_path, "mask_2d", "vis_"+str(selected_vid_idx).zfill(3)+".mp4"
        )
        row, col = get_row_col(len(video_handlers), square=False)
        im_h = 400
        im_w = 600
        if im_h > im_w and len(video_handlers) == 3:
            row, col = 1, 3
        collage_h = im_h * row
        collage_w = im_w * col
        min_height = 1000
        if collage_h > min_height:
            scale = min_height / collage_h
            collage_h = int(collage_h * scale)
            collage_w = int(collage_w * scale)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        collage_handler = cv2.VideoWriter(collage_path, fourcc, 30, (collage_w, collage_h))
        flag = True
        while True:
            imgs = []
            for v_handler in video_handlers:
                ret, frame = v_handler.read()
                if not ret:
                    flag = False
                    break
                imgs.append(frame)
            if not flag:
                break
            collaged = merge(imgs, resize=True)
            collage_handler.write(collaged)
        collage_handler.release()
        print(f"Saved {collage_path}")

        # delete individual videos
        for v_handler in video_handlers:
            v_handler.release()
        time.sleep(0.1)
        shutil.rmtree(vis_dir)
