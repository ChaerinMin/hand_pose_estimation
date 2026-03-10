import argparse
import os
import numpy as np
from tqdm import tqdm
import ujson
import copy
import mediapipe as mp

import sys
sys.path.append(".")
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
import src.utils.params as param_utils
from src.utils.parser import add_common_args

os.system("module load ffmpeg")


def get_bbox(keypoints, img_size):
    """
    keypoints: (n_points, 2)
    img_size: (w, h)

    Returns
        List (xyxy in pixel space)
    """
    im_w, im_h = img_size
    xmin, ymin = keypoints.min(axis=0)
    xmax, ymax = keypoints.max(axis=0)
    padding_x = int(0.1 * (xmax - xmin))
    padding_y = int(0.1 * (ymax - ymin))
    xmin = max(0, xmin - padding_x)
    ymin = max(0, ymin - padding_y)
    xmax = min(im_w - 1, xmax + padding_x)
    ymax = min(im_h - 1, ymax + padding_y)
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def detect_hand_pose(hands, image):
    """
    hands: Mediapipe model
    image: np.ndarray (H, W, 3)

    Return:
        hand_pose: np.ndarray (42, 2)
        detected: np.ndarray (2,)
    """
    H, W, _ = image.shape
    mp_pose = hands.process(image)
    hand_pose = np.zeros((42, 2))
    detected = np.array([0, 0])
    start_idx = 0
    if mp_pose.multi_hand_landmarks:
        # handedness is flipped assuming the input image is mirrored in MediaPipe
        for hand_landmarks, handedness in zip(
            mp_pose.multi_hand_landmarks, mp_pose.multi_handedness
        ):
            # actually right hand
            if handedness.classification[0].label == "Left":
                start_idx = 0
                detected[0] = 1
            # actually left hand
            elif handedness.classification[0].label == "Right":
                start_idx = 21
                detected[1] = 1
            for i, landmark in enumerate(hand_landmarks.landmark):
                hand_pose[start_idx + i] = [landmark.x * W, landmark.y * H]
    return hand_pose, detected

def main():
    parser = argparse.ArgumentParser(description='2D Keypoint Detection')
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--box_score_threshold', type=float, default=0.2, help='Confidence Threshold for BBX Detection')
    parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
    parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
    args = parser.parse_args()
    os.system("module load ffmpeg")
    input_path = os.path.join(args.root_dir, args.seq_path)

    # load params
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

    # cam names
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

    # which video
    if args.ith == -1:
        total_video_idxs = 0
        for fid, folder in enumerate(os.listdir(input_path)):
            if 'cam' in folder and folder not in cams_to_remove:
                length = len([file for file in os.listdir(os.path.join(input_path, folder)) if file.endswith('.mp4')])
                if length > total_video_idxs:
                    total_video_idxs = length
                    max_folder_id = fid
                    anchor_camera_by_length = os.listdir(input_path)[fid]
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

    # mediapipe
    hands = mp.solutions.hands.Hands(
        model_complexity=0,
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    )

    for selected_vid_idx in selected_vid_idxs:
        print(f'Video ID {selected_vid_idx}...')
        
        # save dirs
        output_kps_left_path = f'{args.out_dir}/keypoints_2d/left/{selected_vid_idx:03d}'
        output_bbx_left_path = f'{args.out_dir}/bboxes/left/{selected_vid_idx:03d}'
        output_kps_right_path = f'{args.out_dir}/keypoints_2d/right/{selected_vid_idx:03d}'
        output_bbx_right_path = f'{args.out_dir}/bboxes/right/{selected_vid_idx:03d}'
        os.makedirs(output_kps_left_path, exist_ok=True)
        os.makedirs(output_bbx_left_path, exist_ok=True)
        os.makedirs(output_kps_right_path, exist_ok=True)
        os.makedirs(output_bbx_right_path, exist_ok=True)

        reader = Reader(args.input_type, video_dir, cam_names=cam_names, cams_to_remove=cams_to_remove, ith=selected_vid_idx, anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera)
        if reader.frame_count <= 0:
            continue
        
        # remove cams
        extra_cams_to_remove = reader.to_delete
        cur_cam_names = cam_names.copy()
        for cam in extra_cams_to_remove:
            if cam in cur_cam_names:
                cur_cam_names.remove(cam)
        print("Total Views:", len(cur_cam_names))
        print("Total frames:", reader.frame_count)
        
        # intrinsics / extrinsics
        intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cur_cam_names, cam_mapper, easymocap_format=True)
        print("Total cams:", len(intrs), len(dist_intrs), len(dists))
        print("Reader Length", len(reader.vids))

        for v_idx, input_video_path in enumerate(reader.vids):
            # read frame
            im_names, orig_imgs, im_h, im_w = frame_preprocess(input_video_path, use_parsed, args, intrs[v_idx], dist_intrs[v_idx], dists[v_idx])
            
            # save paths
            video_name = input_video_path.split('/')[-1].split('.')[0]
            output_kps_left_file_path = f"{output_kps_left_path}/{video_name}.jsonl"
            output_bbx_left_file_path = f"{output_bbx_left_path}/{video_name}.jsonl"
            output_kps_right_file_path = f"{output_kps_right_path}/{video_name}.jsonl"
            output_bbx_right_file_path = f"{output_bbx_right_path}/{video_name}.jsonl"                
            
            with open(output_kps_left_file_path, 'w') as kps_left_f,\
                  open(output_bbx_left_file_path, 'w') as bbx_left_f,\
                      open(output_kps_right_file_path, 'w') as kps_right_f,\
                          open(output_bbx_right_file_path, 'w') as bbx_right_f:
                for im_name, frame in tqdm(zip(im_names, orig_imgs), desc=f"{v_idx} / {len(reader.vids)}"):
                    if frame is None:
                        frame = np.zeros((im_h, im_w, 3), dtype=np.uint8)
                        box_l = [0.0, 0.0, 0.0, 0.0]
                        joints_l = [0.0, 0.0, 1.0] * 21
                        box_r = copy.copy(box_l)
                        joints_r = copy.copy(joints_l)
                    else:
                        # 2D keypoints and bbox
                        hand_pose, detected = detect_hand_pose(hands, frame)
                        if detected[1]:
                            joints_l = hand_pose[21:]
                            box_l = get_bbox(joints_l, (im_w, im_h))
                            joints_l = np.concatenate((joints_l, np.ones((21, 1))), axis=-1)
                            joints_l = joints_l.reshape(-1).tolist()
                        else:
                            joints_l = [0.0, 0.0, 1.0] * 21
                            box_l = [0.0, 0.0, 0.0, 0.0]
                        if detected[0]:
                            joints_r = hand_pose[:21]
                            box_r = get_bbox(joints_r, (im_w, im_h))
                            joints_r = np.concatenate((joints_r, np.ones((21, 1))), axis=-1)
                            joints_r = joints_r.reshape(-1).tolist()
                        else:
                            joints_r = [0.0, 0.0, 1.0] * 21
                            box_r = [0.0, 0.0, 0.0, 0.0]

                    ujson.dump(joints_l, kps_left_f)
                    kps_left_f.write('\n')
                    ujson.dump(box_l, bbx_left_f)
                    bbx_left_f.write('\n')
                    ujson.dump(joints_r, kps_right_f)
                    kps_right_f.write('\n')
                    ujson.dump(box_r, bbx_right_f)
                    bbx_right_f.write('\n')
            print(f"Saved {output_kps_left_file_path} and others")


if __name__ == '__main__':
    main()
