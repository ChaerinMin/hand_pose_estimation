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

import sys
sys.path.append(".")
from src.utils.video_reader import frame_preprocess
from src.utils.cameras import removed_cameras
from src.utils.parser import add_common_args
from src.vitpose_wrapper import ViTPoseModel

# ------------------------------ Alpha Pose Helpers ------------------------------ #


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

def process_all_vitposes(pred_poses, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f):                    
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
        left_bboxes, left_keyps, left_valid_counts = process_hand_keypoints_batch(left_hand_keyps, 0.5, 3)
        right_bboxes, right_keyps, right_valid_counts = process_hand_keypoints_batch(right_hand_keyps, 0.5, 3)

        # To identify best left and right hands, consider a criteria, e.g., max valid keypoints
        best_left_index = np.argmax(left_valid_counts)
        best_right_index = np.argmax(right_valid_counts)

        left_bbox = left_bboxes[best_left_index].tolist()
        right_bbox = right_bboxes[best_right_index].tolist()
        left_keyp = left_keyps[best_left_index].tolist()
        right_keyp = right_keyps[best_right_index].tolist()
    
    # Writting
    ujson.dump(left_keyp, kps_left_f)
    kps_left_f.write('\n')

    ujson.dump(left_bbox, bbx_left_f)
    bbx_left_f.write('\n')

    ujson.dump(right_keyp, kps_right_f)
    kps_right_f.write('\n')

    ujson.dump(right_bbox, bbx_right_f)
    bbx_right_f.write('\n')
    

def process_all_yolo_results(results, bboxes_buffer, box_score_threshold=0.2):

    for result in results:
        if len(result) == 0:
            pred_bboxes_scores = []
        else:
            valid_idx = (result.boxes.cls==0) & (result.boxes.conf > box_score_threshold)
            pred_bboxes=result.boxes.xyxy[valid_idx].cpu().numpy()
            pred_scores=result.boxes.conf[valid_idx].cpu().numpy()
            pred_bboxes_scores = np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)
        bboxes_buffer.append(pred_bboxes_scores)


def main():
    parser = argparse.ArgumentParser(description='2D Keypoint Detection')
    add_common_args(parser)
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--box_score_threshold', type=float, default=0.2, help='Confidence Threshold for BBX Detection')
    parser.add_argument('--yolo_model', type=str, default='yolov9c.pt', help='YOLO Model for BBX Detection')
    parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
    parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
    args = parser.parse_args()

    # Setup HaMeR model
    device = torch.device('cuda')
    cpm = ViTPoseModel(device)
    model = YOLO(args.yolo_model)
    
    input_path = os.path.join(args.root_dir, args.seq_path)
    if args.ith == -1:
        folder0 = os.listdir(input_path)[0]
        folder0_path = os.path.join(input_path, folder0)
        selected_vid_idxs = list(range(len(os.listdir(folder0_path))//2))
    else:    
        selected_vid_idxs = [args.ith]
    
    for selected_vid_idx in selected_vid_idxs:
        
        output_kps_left_path = f'{args.out_dir}/keypoints_2d/left/{selected_vid_idx:03d}'
        output_bbx_left_path = f'{args.out_dir}/bboxes/left/{selected_vid_idx:03d}'
        output_kps_right_path = f'{args.out_dir}/keypoints_2d/right/{selected_vid_idx:03d}'
        output_bbx_right_path = f'{args.out_dir}/bboxes/right/{selected_vid_idx:03d}'
        os.makedirs(output_kps_left_path, exist_ok=True)
        os.makedirs(output_bbx_left_path, exist_ok=True)
        os.makedirs(output_kps_right_path, exist_ok=True)
        os.makedirs(output_bbx_right_path, exist_ok=True)
        
        cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam)
        
        # Detect 2D Keypoints for all valid views
        time_list = []
        cams = sorted([_ for _ in os.listdir(input_path) if _ not in cams_to_remove and 'imu' not in _ and 'mic' not in _])[:]
        for cam_name in cams:
            
            input_video_path = glob(f"{input_path}/{cam_name}/*.mp4")[selected_vid_idx]

            im_names, orig_imgs = frame_preprocess(input_video_path)
            
            video_name = input_video_path.split('/')[-1].split('.')[0]
            
            output_kps_left_file_path = f"{output_kps_left_path}/{video_name}.jsonl"
            output_bbx_left_file_path = f"{output_bbx_left_path}/{video_name}.jsonl"
            output_kps_right_file_path = f"{output_kps_right_path}/{video_name}.jsonl"
            output_bbx_right_file_path = f"{output_bbx_right_path}/{video_name}.jsonl"                
            
            start_time = time.time()
            with open(output_kps_left_file_path, 'w') as kps_left_f, open(output_bbx_left_file_path, 'w') as bbx_left_f, open(output_kps_right_file_path, 'w') as kps_right_f, open(output_bbx_right_file_path, 'w') as bbx_right_f:
                frame_buffer, bboxes_buffer = [], []
                for im_name, frame in tqdm(zip(im_names, orig_imgs), total=len(im_names)):
                    frame_buffer.append(frame)
                    if len(frame_buffer) == args.batch_size:
                    
                        # Detect humans in image
                        results = model(frame_buffer, verbose=False, stream=True)
                        process_all_yolo_results(results, bboxes_buffer, args.box_score_threshold)
                        
                        # Detect human keypoints for each person        
                        pred_poses = cpm.predict_pose_batch(
                            frame_buffer,
                            bboxes_buffer
                        )

                        for pred_pose in pred_poses:
                            processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)
                        
                        frame_buffer, bboxes_buffer = [], []

                if len(frame_buffer) > 0:
                    # Detect humans in image
                    results = model(frame_buffer, verbose=False, stream=True)
                    process_all_yolo_results(results, bboxes_buffer, args.box_score_threshold)
                    
                    # Detect human keypoints for each person               
                    pred_poses = cpm.predict_pose_batch(
                        frame_buffer,
                        bboxes_buffer
                    )
                        
                    for pred_pose in pred_poses:
                        processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)
            time_list.append(time.time() - start_time)
            print(f'Total Runtime:{time_list[-1]} s. Average Time {sum(time_list)/len(time_list)} s.')
            
            # exit(0)
if __name__ == '__main__':
    main()
