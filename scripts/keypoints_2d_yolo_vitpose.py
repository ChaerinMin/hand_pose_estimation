from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import ujson
import time
from ultralytics import YOLO, checks

from typing import Dict, Optional
from collections import defaultdict

import cv2
import sys
sys.path.append(".")
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess, create_video_writer, convert_video_ffmpeg
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.vitpose_wrapper import ViTPoseModel
from src.hamer_wrapper import HAMER_CKPT_PATH, ViTDetDataset, recursive_clear

HAND_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
LEFT_HAND_COLOR = (255, 50, 50)
RIGHT_HAND_COLOR = (50, 50, 255)


def draw_hand_keypoints_on_image(image, left_kps, right_kps, conf_thresh=0.3):
    """Draw left (blue) and right (red) hand keypoints on image.
    left_kps, right_kps: (21, 3) arrays [x, y, conf]
    """
    vis = image.copy() if isinstance(image, np.ndarray) else np.array(image)
    for kps, color in [(left_kps, LEFT_HAND_COLOR), (right_kps, RIGHT_HAND_COLOR)]:
        for i, j in HAND_SKELETON:
            if kps[i, 2] > conf_thresh and kps[j, 2] > conf_thresh:
                cv2.line(vis, (int(kps[i, 0]), int(kps[i, 1])),
                         (int(kps[j, 0]), int(kps[j, 1])), color, 2, cv2.LINE_AA)
        for k in range(21):
            if kps[k, 2] > conf_thresh:
                cv2.circle(vis, (int(kps[k, 0]), int(kps[k, 1])), 3, color, -1, cv2.LINE_AA)
    return vis


# ------------------------------ Alpha Pose Helpers ------------------------------ #
os.system("module load ffmpeg")

def process_hand_keypoints_batch(keypoints, validity_threshold, min_valid_keypoints):
    valid_mask = keypoints[:, :, 2] > validity_threshold
    valid_counts = np.sum(valid_mask, axis=1)
    
    # Initialize storage for bounding boxes and reshape keypoints
    bboxes = np.zeros((keypoints.shape[0], 4))
    reshaped_keypoints = np.zeros((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2]))

    # Process only those with enough valid keypoints
    sufficient_valid = valid_counts > min_valid_keypoints
    if np.any(sufficient_valid):
        filtered_keypoints = np.where(valid_mask[:,:,None], keypoints, np.nan)  # Replace invalid keypoints with NaN for min/max operations
        bboxes[sufficient_valid, 0] = np.nanmin(filtered_keypoints[sufficient_valid, :, 0], axis=1)
        bboxes[sufficient_valid, 1] = np.nanmin(filtered_keypoints[sufficient_valid, :, 1], axis=1)
        bboxes[sufficient_valid, 2] = np.nanmax(filtered_keypoints[sufficient_valid, :, 0], axis=1)
        bboxes[sufficient_valid, 3] = np.nanmax(filtered_keypoints[sufficient_valid, :, 1], axis=1)
        reshaped_keypoints[sufficient_valid] = keypoints[sufficient_valid].reshape(-1, keypoints.shape[1] * keypoints.shape[2])
        
    return bboxes, reshaped_keypoints, valid_counts

def prorcess_all_hamerposes(noframe_buffer, hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f):
    boxes = hamer_batch['boxes']
    box_center = hamer_batch["box_center"].float()
    box_size = hamer_batch["box_size"].float()
    
    batch_size = hamer_batch['img'].shape[0]
    for n in range(batch_size):
        is_right = hamer_batch['right'][n]
        if noframe_buffer[n]:
            box = [0.0, 0.0, 0.0, 0.0]
            joints = [0.0, 0.0, 1.0] * 21
        else:
            pred_keypoints_2d = hamer_out['pred_keypoints_2d'][n, :, :].squeeze()
            multiplier = (2*is_right-1)
            pred_keypoints_2d[:, 0] = pred_keypoints_2d[:, 0] * multiplier
            joints = pred_keypoints_2d.detach().cpu() * box_size[n] + box_center[n, :]

            joints = np.hstack((joints.numpy(), np.ones((21, 1)))).reshape(-1).tolist()
            
            box = boxes[n, :].tolist()
        if is_right:
            ujson.dump(joints, kps_right_f)
            kps_right_f.write('\n')
            ujson.dump(box, bbx_right_f)
            bbx_right_f.write('\n')
        else:
            ujson.dump(joints, kps_left_f)
            kps_left_f.write('\n')
            ujson.dump(box, bbx_left_f)
            bbx_left_f.write('\n')
            
    
def process_all_vitposes_for_hamer(pred_poses, frame_buffer, hand_side='both'):
    all_processed_bbox = []
    is_right = []
    for pred_pose in pred_poses:
        processed_pose = process_all_vitposes(pred_pose)
        left_bbox = processed_pose['left_bbox'] if hand_side in ('left', 'both') else [0.0] * 4
        right_bbox = processed_pose['right_bbox'] if hand_side in ('right', 'both') else [0.0] * 4
        all_processed_bbox.append(left_bbox)
        all_processed_bbox.append(right_bbox)
        is_right.extend([0, 1])

    all_processed_bbox_array = np.array(all_processed_bbox)
    is_right_array = np.array(is_right)
    frame_buffer = np.array(frame_buffer)
    repeated_frame_buffer = frame_buffer[np.repeat(np.arange(len(frame_buffer)), 2)]
    return all_processed_bbox_array, is_right_array, repeated_frame_buffer
    
def process_all_vitposes(pred_poses, kps_left_f=None, bbx_left_f=None, kps_right_f=None, bbx_right_f=None, hand_side='both'):
    if len(pred_poses) == 0:
        left_bbox = [0.0] * 4
        right_bbox = [0.0] * 4
        left_keyp = [0.0] * (21 * 3)
        right_keyp = [0.0] * (21 * 3)
    else:
        keypoints_all = pred_poses

        # Split into left and right hands
        left_hand_keyps = keypoints_all[:, -42:-21, :]
        right_hand_keyps = keypoints_all[:, -21:, :]

        # Process each hand
        left_bboxes, left_keyps, left_valid_counts = process_hand_keypoints_batch(left_hand_keyps, 0.7, 12)
        right_bboxes, right_keyps, right_valid_counts = process_hand_keypoints_batch(right_hand_keyps, 0.7, 12)

        # To identify best left and right hands, consider a criteria, e.g., max valid keypoints
        best_left_index = np.argmax(left_valid_counts)
        best_right_index = np.argmax(right_valid_counts)

        left_bbox = left_bboxes[best_left_index].tolist() if hand_side in ('left', 'both') else [0.0] * 4
        right_bbox = right_bboxes[best_right_index].tolist() if hand_side in ('right', 'both') else [0.0] * 4
        left_keyp = left_keyps[best_left_index].tolist() if hand_side in ('left', 'both') else [0.0] * (21 * 3)
        right_keyp = right_keyps[best_right_index].tolist() if hand_side in ('right', 'both') else [0.0] * (21 * 3)
    
    # Writting
    if kps_left_f:
        ujson.dump(left_keyp, kps_left_f)
        kps_left_f.write('\n')

        ujson.dump(left_bbox, bbx_left_f)
        bbx_left_f.write('\n')

        ujson.dump(right_keyp, kps_right_f)
        kps_right_f.write('\n')

        ujson.dump(right_bbox, bbx_right_f)
        bbx_right_f.write('\n')
    else:
        return {
            'left_keyp': left_keyp,
            'left_bbox': left_bbox,
            'right_keyp': right_keyp,
            'right_bbox': right_bbox,
        }
    

def process_all_yolo_results(results, bboxes_buffer, im_h, im_w, box_score_threshold=0.2, padding=5):

    for result in results:
        if len(result) == 0:
            pred_bboxes_scores = []
        else:
            valid_idx = (result.boxes.cls==0) & (result.boxes.conf > box_score_threshold)
            pred_bboxes=result.boxes.xyxy[valid_idx].cpu().numpy()
            padded_bboxes = np.copy(pred_bboxes)
            padded_bboxes[:, 0:2] -= padding  # x_min - 10
            padded_bboxes[:, 2:] += padding  # x_max + 10
            padded_bboxes[:, 0] = np.clip(padded_bboxes[:, 0], 0, im_h)  # x_min
            padded_bboxes[:, 1] = np.clip(padded_bboxes[:, 1], 0, im_w) # y_min
            padded_bboxes[:, 2] = np.clip(padded_bboxes[:, 2], 0, im_h)  # x_max
            padded_bboxes[:, 3] = np.clip(padded_bboxes[:, 3], 0, im_w) # y_max
            pred_scores=result.boxes.conf[valid_idx].cpu().numpy()
            pred_bboxes_scores = np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)
        bboxes_buffer.append(pred_bboxes_scores)


def main():
    parser = argparse.ArgumentParser(description='2D Keypoint Detection')
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--box_score_threshold', type=float, default=0.2, help='Confidence Threshold for BBX Detection')
    parser.add_argument('--yolo_model', type=str, default='yolov9c.pt', help='YOLO Model for BBX Detection')
    parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
    parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
    parser.add_argument('--use_hamer', type=bool, default=True, help='YOLO -> ViTPose -> Hamer pipeline')
    parser.add_argument('--hand_side', type=str, default='both', choices=['left', 'right', 'both'], help='Which hand(s) to detect; use left or right to suppress false positives from the absent hand')
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, "hand")
    os.system("module load ffmpeg")

    # Setup HaMeR model
    device = torch.device('cuda')
    cpm = ViTPoseModel(device)
    model = YOLO(args.yolo_model)

    if args.use_hamer:
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
        hamer_model, hamer_model_cfg = load_hamer(HAMER_CKPT_PATH)
        hamer_model = hamer_model.to(device)
        hamer_model.eval()

    input_path = os.path.join(args.root_dir, args.seq_path)

    if args.use_optim_params:
        params_txt = "optim_params.txt"
    else:
        params_txt = "params.txt"

    calib_dir = os.path.join(
        args.root_dir, args.seq_path, args.multisequence,
        "calib", f"stage{args.stage}", "sparse", "0"
    )
    params_path = os.path.join(calib_dir, params_txt)
    if args.stage == 1:
        params = param_utils.read_params(params_path, distortion=True, args=args)
        use_parsed = False
    elif args.stage == 2:
        params = param_utils.read_params(params_path, distortion=False, args=args)
        use_parsed = True
    else:
        raise ValueError("Cannot determine whether to assume undistorted.")
    cam_names = list(params[:]["cam_name"])
    cam_names = [c.replace(".", "") for c in cam_names]
    removed_camera_path = os.path.join(calib_dir, 'ignore_camera.txt')
    if os.path.isfile(removed_camera_path):
        with open(removed_camera_path) as file:
            ignored_cameras = [line.rstrip() for line in file]
    else:
        ignored_cameras = None
    cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras)

    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)
    if args.video_dir:
        video_dir = os.path.join(args.video_dir, args.seq_path)
    else:
        video_dir = input_path
    cam_mapper = map_camera_names(video_dir, cam_names)

    if args.ith == -1:
        total_video_idxs = 0
        max_folder_id = 0
        for fid, folder in enumerate(os.listdir(video_dir)):
            if 'cam' in folder and folder not in cams_to_remove:
                length = len([file for file in os.listdir(os.path.join(video_dir, folder)) if file.endswith('.mp4')])
                if length > total_video_idxs:
                    total_video_idxs = length
                    max_folder_id = fid
                    anchor_camera_by_length = os.listdir(video_dir)[fid]
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
        
        output_kps_left_path = f'{args.out_dir}/intermediate/keypoints_2d/left/{selected_vid_idx:03d}'
        output_bbx_left_path = f'{args.out_dir}/intermediate/bboxes/left/{selected_vid_idx:03d}'
        output_kps_right_path = f'{args.out_dir}/intermediate/keypoints_2d/right/{selected_vid_idx:03d}'
        output_bbx_right_path = f'{args.out_dir}/intermediate/bboxes/right/{selected_vid_idx:03d}'
        os.makedirs(output_kps_left_path, exist_ok=True)
        os.makedirs(output_bbx_left_path, exist_ok=True)
        os.makedirs(output_kps_right_path, exist_ok=True)
        os.makedirs(output_bbx_right_path, exist_ok=True)

        # Get files to process
        reader = Reader(args.input_type, video_dir, cam_names=cam_names, cams_to_remove=cams_to_remove, ith=selected_vid_idx, anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera)
        if reader.frame_count <= 0:
            continue
        
        extra_cams_to_remove = reader.to_delete
        cur_cam_names = cam_names.copy()

        for cam in extra_cams_to_remove:
            if cam in cur_cam_names:
                cur_cam_names.remove(cam)
        print("Total Views:", len(cur_cam_names))
        print("Total frames:", reader.frame_count)
        
        intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cur_cam_names, cam_mapper, easymocap_format=True)
        print("Total cams:", len(intrs), len(dist_intrs), len(dists))
        print("Reader Length", len(reader.vids))

        # Detect 2D Keypoints for all valid views
        time_list = []
        for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids)):
            # print(input_video_path)
            # print(intrs[v_idx], dist_intrs[v_idx], dists[v_idx])
            im_names, orig_imgs, im_h, im_w = frame_preprocess(input_video_path, use_parsed, args, intrs[v_idx], dist_intrs[v_idx], dists[v_idx])

            video_name = input_video_path.split('/')[-1].split('.')[0]

            output_kps_left_file_path = f"{output_kps_left_path}/{video_name}.jsonl"
            output_bbx_left_file_path = f"{output_bbx_left_path}/{video_name}.jsonl"
            output_kps_right_file_path = f"{output_kps_right_path}/{video_name}.jsonl"
            output_bbx_right_file_path = f"{output_bbx_right_path}/{video_name}.jsonl"

            if im_h is None:
                # Camera has no parsed image in any timestamp. Write placeholder
                # jsonls matching the noframe pattern (x=0, y=0, conf=1 for kps;
                # zero box) so downstream triangulation alignment stays intact
                # and the invalidity filter excludes this camera.
                print(f"Warning: no parsed frames for {video_name}, writing placeholder jsonl")
                placeholder_kps = [0.0, 0.0, 1.0] * 21
                placeholder_box = [0.0, 0.0, 0.0, 0.0]
                with open(output_kps_left_file_path, 'w') as kps_left_f, \
                     open(output_bbx_left_file_path, 'w') as bbx_left_f, \
                     open(output_kps_right_file_path, 'w') as kps_right_f, \
                     open(output_bbx_right_file_path, 'w') as bbx_right_f:
                    for _ in im_names:
                        for f in (kps_left_f, kps_right_f):
                            ujson.dump(placeholder_kps, f)
                            f.write('\n')
                        for f in (bbx_left_f, bbx_right_f):
                            ujson.dump(placeholder_box, f)
                            f.write('\n')
                continue

            start_time = time.time()
            if use_parsed:
                assert args.use_hamer, "Not implemented"
            with open(output_kps_left_file_path, 'w') as kps_left_f, open(output_bbx_left_file_path, 'w') as bbx_left_f, open(output_kps_right_file_path, 'w') as kps_right_f, open(output_bbx_right_file_path, 'w') as bbx_right_f:
                frame_buffer, bboxes_buffer = [], []
                noframe_buffer = []
                for im_name, frame in zip(im_names, orig_imgs):
                    if frame is None:
                        frame = np.zeros((im_h, im_w, 3), dtype=np.uint8)
                        noframe_buffer.append(1)
                    else:
                        noframe_buffer.append(0)
                    frame_buffer.append(frame)
                    if len(frame_buffer) == args.batch_size:
                        # Detect humans in image
                        with torch.no_grad():
                            results = model(frame_buffer, verbose=False, stream=True)
                        process_all_yolo_results(results, bboxes_buffer, im_h, im_w, args.box_score_threshold, padding=0)
                        
                        # Detect human keypoints for each person
                        with torch.no_grad():        
                            pred_poses = cpm.predict_pose_batch(
                                frame_buffer,
                                bboxes_buffer
                            )

                        if args.use_hamer:
                            boxes, right, repeated_frame_buffer = process_all_vitposes_for_hamer(pred_poses, frame_buffer, args.hand_side)
                            noframe_buffer = np.array(noframe_buffer)
                            repeated_noframe_buffer = noframe_buffer[np.repeat(np.arange(len(noframe_buffer)), 2)]
                            if args.hand_side == 'right':
                                repeated_noframe_buffer[0::2] = 1  # suppress left
                            elif args.hand_side == 'left':
                                repeated_noframe_buffer[1::2] = 1  # suppress right
                            hamer_dataset = ViTDetDataset(hamer_model_cfg, repeated_frame_buffer, boxes, right, rescale_factor=2.0, device=device)
                            hamer_dataloader = torch.utils.data.DataLoader(hamer_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
                            for hamer_batch in hamer_dataloader:
                                with torch.no_grad():
                                    hamer_out = hamer_model(hamer_batch)
                                    recursive_clear(hamer_batch)
                                prorcess_all_hamerposes(repeated_noframe_buffer, hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)                            
                        else:
                            for pred_pose in pred_poses:
                                processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f, args.hand_side)

                        frame_buffer, bboxes_buffer, noframe_buffer = [], [], []

                if len(frame_buffer) > 0:
                    # Detect humans in image
                    results = model(frame_buffer, verbose=False, stream=True)
                    process_all_yolo_results(results, bboxes_buffer, im_h, im_w, args.box_score_threshold, padding=5)
                    
                    # Detect human keypoints for each person               
                    pred_poses = cpm.predict_pose_batch(
                        frame_buffer,
                        bboxes_buffer
                    )

                    if args.use_hamer:
                        boxes, right, repeated_frame_buffer = process_all_vitposes_for_hamer(pred_poses, frame_buffer, args.hand_side)
                        noframe_buffer = np.array(noframe_buffer)
                        repeated_noframe_buffer = noframe_buffer[np.repeat(np.arange(len(noframe_buffer)), 2)]
                        if args.hand_side == 'right':
                            repeated_noframe_buffer[0::2] = 1  # suppress left
                        elif args.hand_side == 'left':
                            repeated_noframe_buffer[1::2] = 1  # suppress right
                        hamer_dataset = ViTDetDataset(hamer_model_cfg, repeated_frame_buffer, boxes, right, rescale_factor=2.0, device=device)
                        hamer_dataloader = torch.utils.data.DataLoader(hamer_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
                        for hamer_batch in hamer_dataloader:
                            start_time = time.time()
                            with torch.no_grad():
                                hamer_out = hamer_model(hamer_batch)
                                recursive_clear(hamer_batch)
                            prorcess_all_hamerposes(repeated_noframe_buffer, hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f) 
                                
                    else:
                        for pred_pose in pred_poses:
                            processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f, args.hand_side)
            time_list.append(time.time() - start_time)

            # Per-camera 2D keypoint visualization
            vis_path = os.path.join(args.out_dir, 'vis', 'keypoints_2d',
                                    f'{selected_vid_idx:03d}', f"{video_name}.mp4")
            os.makedirs(os.path.dirname(vis_path), exist_ok=True)
            vis_writer = create_video_writer(vis_path, (im_w, im_h), fps=30)
            with open(output_kps_left_file_path, 'r') as kl, \
                 open(output_kps_right_file_path, 'r') as kr:
                for frame, left_line, right_line in zip(orig_imgs, kl, kr):
                    if frame is None:
                        frame_vis = np.zeros((im_h, im_w, 3), dtype=np.uint8)
                    else:
                        frame_vis = frame if isinstance(frame, np.ndarray) else np.array(frame)
                    left_kps = np.array(ujson.loads(left_line.strip())).reshape(21, 3)
                    right_kps = np.array(ujson.loads(right_line.strip())).reshape(21, 3)
                    vis_frame = draw_hand_keypoints_on_image(frame_vis, left_kps, right_kps)
                    vis_writer.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            vis_writer.release()
            convert_video_ffmpeg(vis_path)


if __name__ == '__main__':
    main()
