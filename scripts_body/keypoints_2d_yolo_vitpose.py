"""
Full-body 2D keypoint extraction using YOLO (person detection) + ViTPose (COCO-WholeBody 133 keypoints).

Two-stage approach:
  Stage 1: Run ViTPose on person bounding box -> 133 keypoints (body, foot, face, hands)
  Stage 2: Crop face/hand regions from Stage 1 keypoints, re-run ViTPose for refinement,
           and replace the corresponding keypoints if the refined version is better.

COCO-WholeBody 133 keypoint layout:
  Body:      0-16   (17 keypoints)
  Foot:      17-22  (6 keypoints)
  Face:      23-90  (68 keypoints)
  Left hand: 91-111 (21 keypoints)
  Right hand:112-132(21 keypoints)

Output: keypoints_2d/{vid_idx:03d}/{cam_name}.jsonl
  Each line is a JSON array of shape (133*3,) = [x0, y0, conf0, x1, y1, conf1, ...]
"""

import argparse
import os
import sys
import glob
import natsort

import cv2
import numpy as np
import torch
import ujson
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image

sys.path.append(".")  # noqa: E402
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess, create_video_writer, convert_video_ffmpeg
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.vitpose_wrapper import ViTPoseModel

# COCO-WholeBody index ranges
BODY_IDX = slice(0, 17)
FOOT_IDX = slice(17, 23)
FACE_IDX = slice(23, 91)
LHAND_IDX = slice(91, 112)
RHAND_IDX = slice(112, 133)
NUM_KEYPOINTS = 133

# Colors (BGR) for each body part
PART_COLORS = {
    'body': (0, 255, 0),       # green
    'foot': (0, 200, 200),     # yellow-ish
    'face': (255, 200, 100),   # light blue
    'lhand': (0, 0, 255),      # red
    'rhand': (255, 0, 0),      # blue
}

# COCO-WholeBody skeleton connections (subset for visualization)
BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]
FOOT_SKELETON = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
# Hand skeleton: wrist(0)->1->2->3->4, 0->5->6->7->8, etc.
HAND_SKELETON = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def draw_keypoints_on_image(image, keypoints, conf_thresh=0.3):
    """Draw COCO-WholeBody 133 keypoints and skeleton on image.

    Args:
        image: BGR image (will be modified in place)
        keypoints: (133, 3) array [x, y, confidence]
        conf_thresh: minimum confidence to draw a keypoint/limb

    Returns:
        image with keypoints drawn
    """
    vis = image.copy()

    def draw_limbs(skeleton, offset, color, thickness=2):
        for (i, j) in skeleton:
            pi, pj = i + offset, j + offset
            if (keypoints[pi, 2] > conf_thresh and keypoints[pj, 2] > conf_thresh):
                pt1 = (int(keypoints[pi, 0]), int(keypoints[pi, 1]))
                pt2 = (int(keypoints[pj, 0]), int(keypoints[pj, 1]))
                cv2.line(vis, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw skeletons
    draw_limbs(BODY_SKELETON, 0, PART_COLORS['body'], 4)
    draw_limbs(FOOT_SKELETON, 0, PART_COLORS['foot'], 4)
    draw_limbs(HAND_SKELETON, 91, PART_COLORS['lhand'], 2)
    draw_limbs(HAND_SKELETON, 112, PART_COLORS['rhand'], 2)

    # Draw keypoints as circles
    part_ranges = [
        (BODY_IDX, PART_COLORS['body'], 5),
        (FOOT_IDX, PART_COLORS['foot'], 4),
        (FACE_IDX, PART_COLORS['face'], 2),
        (LHAND_IDX, PART_COLORS['lhand'], 3),
        (RHAND_IDX, PART_COLORS['rhand'], 3),
    ]
    for idx_slice, color, radius in part_ranges:
        kps = keypoints[idx_slice]
        for k in range(kps.shape[0]):
            if kps[k, 2] > conf_thresh:
                pt = (int(kps[k, 0]), int(kps[k, 1]))
                cv2.circle(vis, pt, radius, color, -1, cv2.LINE_AA)

    return vis


def compute_crop_bbox(keypoints, indices, im_w, im_h, padding_ratio=0.5, min_valid=3, conf_thresh=0.3):
    """Compute a bounding box around a subset of keypoints for cropping.

    Args:
        keypoints: (133, 3) array [x, y, confidence]
        indices: slice or array of indices to use
        im_w, im_h: image dimensions
        padding_ratio: extra padding around the keypoint bbox
        min_valid: minimum number of valid keypoints required
        conf_thresh: confidence threshold for valid keypoints

    Returns:
        bbox [x1, y1, x2, y2] or None if not enough valid keypoints
    """
    kps = keypoints[indices]
    valid = kps[:, 2] > conf_thresh
    if valid.sum() < min_valid:
        return None

    pts = kps[valid, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    w, h = x2 - x1, y2 - y1
    pad_x = padding_ratio * w
    pad_y = padding_ratio * h

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(im_w, int(x2 + pad_x))
    y2 = min(im_h, int(y2 + pad_y))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return [x1, y1, x2, y2]


def refine_keypoints_batch(cpm, frame_buffer, all_keypoints, im_w, im_h):
    """Refine face and hand keypoints for a batch of frames.

    Collects all crops from all frames and processes them in one batch call to ViTPose.

    Args:
        cpm: ViTPoseModel instance
        frame_buffer: list of BGR images
        all_keypoints: list of (133, 3) coarse keypoints from Stage 1
        im_w, im_h: image dimensions

    Returns:
        list of (133, 3) refined keypoints
    """
    parts = [
        ("face", FACE_IDX, 0.5, 5, 0.2),
        ("lhand", LHAND_IDX, 0.6, 3, 0.2),
        ("rhand", RHAND_IDX, 0.6, 3, 0.2),
    ]

    # Collect all crops from all frames
    all_crops = []
    all_person_bboxes = []
    crop_info = []  # (frame_idx, part_idx, bbox, orig_keypoints_idx)

    for frame_idx, (frame, keypoints) in enumerate(zip(frame_buffer, all_keypoints)):
        for name, idx, pad_ratio, min_valid, conf_thresh in parts:
            bbox = compute_crop_bbox(keypoints, idx, im_w, im_h,
                                     padding_ratio=pad_ratio, min_valid=min_valid,
                                     conf_thresh=conf_thresh)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_h < 10 or crop_w < 10:
                continue

            # Person bbox covering the entire crop (in crop coordinates)
            person_bbox = np.array([[0, 0, crop_w, crop_h, 1.0]], dtype=np.float32)

            all_crops.append(crop)
            all_person_bboxes.append(person_bbox)
            crop_info.append((frame_idx, idx, bbox))

    # If no crops, return original keypoints
    if len(all_crops) == 0:
        return all_keypoints

    # Batch process all crops at once
    pred_poses = cpm.predict_pose_batch(all_crops, all_person_bboxes)
    if pred_poses is None:
        return all_keypoints

    # Apply refined keypoints back to original frames
    refined_keypoints = [kps.copy() for kps in all_keypoints]

    for crop_idx, (frame_idx, part_idx, bbox) in enumerate(crop_info):
        if crop_idx >= len(pred_poses) or pred_poses[crop_idx] is None or len(pred_poses[crop_idx]) == 0:
            continue

        refined_kps = pred_poses[crop_idx]  # (N_persons, 133, 3)
        if len(refined_kps) == 0:
            continue

        # Take the first (and likely only) person
        if refined_kps.ndim == 3:
            refined_part = refined_kps[0][part_idx]
        else:
            refined_part = refined_kps[part_idx]

        # Shift coordinates back to original image space
        x1, y1, x2, y2 = bbox
        refined_part_shifted = refined_part.copy()
        refined_part_shifted[:, 0] += x1
        refined_part_shifted[:, 1] += y1

        # Replace if refined has higher mean confidence
        orig_conf = refined_keypoints[frame_idx][part_idx, 2].mean()
        refined_conf = refined_part_shifted[:, 2].mean()
        if refined_conf > orig_conf:
            refined_keypoints[frame_idx][part_idx] = refined_part_shifted

    return refined_keypoints


def process_all_yolo_results(results, bboxes_buffer, im_h, im_w, box_score_threshold=0.2, padding=5):
    """Extract person bounding boxes from YOLO results."""
    for result in results:
        if len(result) == 0:
            pred_bboxes_scores = np.zeros((0, 5), dtype=np.float32)
        else:
            valid_idx = (result.boxes.cls == 0) & (result.boxes.conf > box_score_threshold)
            pred_bboxes = result.boxes.xyxy[valid_idx].cpu().numpy()
            if len(pred_bboxes) == 0:
                pred_bboxes_scores = np.zeros((0, 5), dtype=np.float32)
            else:
                padded_bboxes = np.copy(pred_bboxes)
                padded_bboxes[:, 0] = np.clip(padded_bboxes[:, 0] - padding, 0, im_w)
                padded_bboxes[:, 1] = np.clip(padded_bboxes[:, 1] - padding, 0, im_h)
                padded_bboxes[:, 2] = np.clip(padded_bboxes[:, 2] + padding, 0, im_w)
                padded_bboxes[:, 3] = np.clip(padded_bboxes[:, 3] + padding, 0, im_h)
                pred_scores = result.boxes.conf[valid_idx].cpu().numpy()
                pred_bboxes_scores = np.concatenate([padded_bboxes, pred_scores[:, None]], axis=1)
        bboxes_buffer.append(pred_bboxes_scores)


def select_best_person(keypoints_all, im_w, im_h):
    """Select the best person detection based on number of valid body keypoints
    and proximity to image center.

    Args:
        keypoints_all: (N_persons, 133, 3) array
        im_w, im_h: image dimensions

    Returns:
        best_keypoints: (133, 3) array, or zeros if no valid detection
    """
    if len(keypoints_all) == 0:
        return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    if keypoints_all.ndim == 2:
        # Single person
        return keypoints_all

    # Score: number of valid body keypoints (conf > 0.3)
    body_kps = keypoints_all[:, BODY_IDX, :]
    valid_counts = (body_kps[:, :, 2] > 0.3).sum(axis=1)

    # Among those with most valid keypoints, prefer the one closest to center
    best_idx = np.argmax(valid_counts)

    if valid_counts[best_idx] < 3:
        return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    return keypoints_all[best_idx]


def process_vitpose_results(pred_poses, frame_buffer, cpm, im_w, im_h, refine_crops=True):
    """Process ViTPose batch results: select best person per frame, optionally refine.

    Args:
        pred_poses: list of (N_persons, 133, 3) arrays, one per frame
        frame_buffer: list of BGR images
        cpm: ViTPoseModel instance (for refinement)
        im_w, im_h: image dimensions
        refine_crops: whether to do two-stage crop refinement

    Returns:
        list of (133, 3) arrays, one per frame
    """
    # Select best person for each frame
    results = []
    for pred_pose in pred_poses:
        kps = select_best_person(pred_pose, im_w, im_h)
        results.append(kps)

    # Batch refine all frames at once
    if refine_crops:
        # Only refine frames with valid keypoints
        valid_frames = [i for i, kps in enumerate(results) if kps[:, 2].sum() > 0]
        if len(valid_frames) > 0:
            valid_frame_buffer = [frame_buffer[i] for i in valid_frames]
            valid_keypoints = [results[i] for i in valid_frames]

            refined = refine_keypoints_batch(cpm, valid_frame_buffer, valid_keypoints, im_w, im_h)

            # Put refined keypoints back
            for i, frame_idx in enumerate(valid_frames):
                results[frame_idx] = refined[i]

    return results


def create_collage(images, grid_cols=None):
    """Create a grid collage from multiple camera images.

    Args:
        images: list of BGR images (all same size)
        grid_cols: number of columns in grid (auto-calculated if None)

    Returns:
        collage image
    """
    if len(images) == 0:
        return None

    n = len(images)
    if grid_cols is None:
        # Auto-determine grid size (prefer wider layouts)
        grid_cols = int(np.ceil(np.sqrt(n * 1.5)))
    grid_rows = int(np.ceil(n / grid_cols))

    h, w = images[0].shape[:2]
    collage = np.zeros((h * grid_rows, w * grid_cols, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        collage[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return collage


def extract_keypoints(args, params, cam_names, cam_mapper,
                      cur_cam_names, reader, selected_vid_idx,
                      cpm, model, refine_crops):
    """Phase 1: Extract VitPose keypoints for all cameras."""
    output_kps_path = f'{args.out_dir}/keypoints_2d/{selected_vid_idx:03d}'
    os.makedirs(output_kps_path, exist_ok=True)

    intrs, projs, dist_intrs, dists, cameras = get_projections(
        args, params, cur_cam_names, cam_mapper, easymocap_format=True
    )

    for v_idx, input_video_path in tqdm(
        enumerate(reader.vids), total=len(reader.vids),
        desc="Extracting keypoints"
    ):
        im_names, orig_imgs, im_h, im_w = frame_preprocess(
            input_video_path,
            args.stage == 2,  # use_parsed
            args,
            intrs[v_idx], dist_intrs[v_idx], dists[v_idx]
        )

        video_name = input_video_path.split('/')[-1].split('.')[0]
        output_kps_file = f"{output_kps_path}/{video_name}.jsonl"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vis_path = os.path.join(args.out_dir, 'vis_keypoints_2d', f'{selected_vid_idx:03d}', f"{video_name}.mp4")
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)
        vis_writer = cv2.VideoWriter(vis_path, fourcc, 30, (im_w, im_h))

        with open(output_kps_file, 'w') as kps_f:
            frame_buffer, bboxes_buffer = [], []
            noframe_buffer = []

            for _, frame in zip(im_names, orig_imgs):
                if frame is None:
                    frame = np.zeros((im_h, im_w, 3), dtype=np.uint8)
                    noframe_buffer.append(True)
                else:
                    noframe_buffer.append(False)
                frame_buffer.append(frame)

                if len(frame_buffer) == args.batch_size:
                    _process_batch(
                        model, cpm, frame_buffer,
                        bboxes_buffer, noframe_buffer,
                        im_h, im_w,
                        args.box_score_threshold,
                        refine_crops, kps_f, vis_writer=vis_writer
                    )
                    frame_buffer = []
                    bboxes_buffer = []
                    noframe_buffer = []

            if len(frame_buffer) > 0:
                _process_batch(
                    model, cpm, frame_buffer,
                    bboxes_buffer, noframe_buffer,
                    im_h, im_w,
                    args.box_score_threshold,
                    refine_crops, kps_f, vis_writer=vis_writer
                )
        vis_writer.release()


def create_collage_video(args, params, cam_names, cam_mapper,
                         cur_cam_names, reader, selected_vid_idx):
    # read keypoints
    output_kps_path = f'{args.out_dir}/keypoints_2d/{selected_vid_idx:03d}'
    cam_keypoints = {}
    cam_video_paths = {}
    for v_idx, input_video_path in enumerate(reader.vids):
        video_name = input_video_path.split('/')[-1].split('.')[0]
        kps_file = f"{output_kps_path}/{video_name}.jsonl"
        if not os.path.exists(kps_file):
            continue
        keypoints_list = []
        with open(kps_file, 'r') as f:
            for line in f:
                kps_flat = np.array(ujson.loads(line.strip()), dtype=np.float32)
                kps = kps_flat.reshape(NUM_KEYPOINTS, 3)
                keypoints_list.append(kps)
        cam_keypoints[video_name] = keypoints_list
        cam_video_paths[video_name] = input_video_path
    if len(cam_keypoints) == 0:
        print("No keypoints found, skipping visualization")
        return

    # cam names and num of frames
    cam_names_sorted = sorted(cam_keypoints.keys())
    num_frames = min(len(kps) for kps in cam_keypoints.values())
    print(f"Collage: {len(cam_names_sorted)} cameras, {num_frames} frames")

    # multiseq, parsed, timestamp dirs, image size
    if args.setting == "brics-mini":
        cam_name_slicer = slice(0, 21)
    elif args.setting == "brics-studio":
        cam_name_slicer = slice(0,18)
    else:
        raise NotImplementedError()
    assert args.stage == 2, "Only implemented for use_parsed"
    cam_name = os.path.basename(cam_video_paths[cam_names_sorted[0]]).split('.')[0][cam_name_slicer]
    # multiseq_chr = args.out_dir.index("multisequence")
    # multiseq = args.out_dir[multiseq_chr:multiseq_chr+19]
    # data_root = args.out_dir[:multiseq_chr]
    parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")
    timestamp_dirs = natsort.natsorted(glob.glob(os.path.join(parsed_dir, "timestamp_*")))
    datalen = len(timestamp_dirs)
    img_path = os.path.join(timestamp_dirs[datalen//2], "images", f"{cam_name}.jpg")
    frame = Image.open(img_path)
    im_w, im_h = frame.width, frame.height

    # rows & columns
    grid_cols = int(np.ceil(np.sqrt(len(cam_names_sorted) * 1.5)))
    grid_rows = int(np.ceil(len(cam_names_sorted) / grid_cols))
    target_collage_height = 1000
    scale_factor = target_collage_height / (grid_rows * im_h)
    scaled_w = int(im_w * scale_factor)
    scaled_h = int(im_h * scale_factor)

    # video writer
    collage_w = scaled_w * grid_cols
    collage_h = scaled_h * grid_rows
    vis_dir = os.path.join(args.out_dir, 'vis_keypoints_2d', f'{selected_vid_idx:03d}')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, f"collage.mp4")
    vis_writer = create_video_writer(vis_path, (collage_w, collage_h), fps=30)

    for frame_idx in tqdm(range(num_frames), desc="Creating collage video"):
        frame_images = []

        for cam_name in cam_names_sorted:
            img_path = os.path.join(timestamp_dirs[frame_idx], "images", f"{cam_name[cam_name_slicer]}.jpg")
            if os.path.exists(img_path):
                frame = np.array(Image.open(img_path))
            else:
                frame = np.ones((im_h, im_w, 3), dtype=np.uint8) * 255

            kps = cam_keypoints[cam_name][frame_idx]
            vis_frame = draw_keypoints_on_image(frame, kps)
            cv2.putText(vis_frame, cam_name, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)
            vis_frame_resized = cv2.resize(vis_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            vis_frame_resized = cv2.cvtColor(vis_frame_resized, cv2.COLOR_BGR2RGB)
            frame_images.append(vis_frame_resized)

        collage = create_collage(frame_images, grid_cols=grid_cols)
        vis_writer.write(collage)

    vis_writer.release()
    convert_video_ffmpeg(vis_path)
    print(f"Collage video saved to: {vis_path}")


def main():
    parser = argparse.ArgumentParser(description='Full-Body 2D Keypoint Detection (COCO-WholeBody 133)')
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for YOLO + ViTPose')
    parser.add_argument('--box_score_threshold', type=float, default=0.2, help='Confidence threshold for person detection')
    parser.add_argument('--yolo_model', type=str, default='yolov9c.pt', help='YOLO model for person detection')
    parser.add_argument('--no_refine', action='store_true', help='Skip two-stage crop refinement for face/hands')
    # parser.add_argument('--vis', action='store_true', default=True, help='Create collage visualization video from all cameras')
    parser.add_argument('--vis_only', action='store_true', help='Only create visualization (skip keypoint extraction)')
    args = parser.parse_args()

    device = torch.device('cuda')
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
    elif args.stage == 2:
        params = param_utils.read_params(params_path, distortion=False, args=args)
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
    cams_to_remove = removed_cameras(
        remove_side=False, remove_bottom=False, ignored_cameras=ignored_cameras
    )
    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)
    if args.video_dir:
        video_dir = os.path.join(args.video_dir, args.seq_path)
    else:
        video_dir = input_path
    cam_mapper = map_camera_names(video_dir, cam_names)

    # Determine which videos to process
    if args.ith == -1:
        total_video_idxs = 0
        anchor_camera_by_length = None
        for fid, folder in enumerate(os.listdir(input_path)):
            if 'cam' in folder and folder not in cams_to_remove:
                length = len([f for f in os.listdir(os.path.join(input_path, folder)) if f.endswith('.mp4')])
                if length > total_video_idxs:
                    total_video_idxs = length
                    anchor_camera_by_length = os.listdir(input_path)[fid]
        if args.start > 0:
            selected_vid_idxs = list(range(args.start, args.end if args.end > 0 else total_video_idxs))
        else:
            selected_vid_idxs = list(range(args.end if args.end > 0 else total_video_idxs))
    else:
        selected_vid_idxs = [args.ith]

    refine_crops = not args.no_refine

    # Initialize models only if we need to extract keypoints
    if not args.vis_only:
        cpm = ViTPoseModel(device)
        model = YOLO(args.yolo_model)
        # Note: Ultralytics YOLO uses device parameter at inference time

    for selected_vid_idx in selected_vid_idxs:
        print(f'Video ID {selected_vid_idx}...')

        reader = Reader(
            args.input_type, video_dir, cam_names=cam_names,
            cams_to_remove=cams_to_remove, ith=selected_vid_idx,
            anchor_camera=anchor_camera_by_length if args.ith == -1 else args.anchor_camera
        )
        if reader.frame_count <= 0:
            continue

        extra_cams_to_remove = reader.to_delete
        cur_cam_names = cam_names.copy()
        for cam in extra_cams_to_remove:
            if cam in cur_cam_names:
                cur_cam_names.remove(cam)
        print("Total Views:", len(cur_cam_names))
        print("Total frames:", reader.frame_count)

        # Phase 1: Extract keypoints
        if not args.vis_only:
            print("Phase 1: Extracting keypoints...")
            extract_keypoints(args, params, cam_names, cam_mapper,
                            cur_cam_names, reader, selected_vid_idx,
                            cpm, model, refine_crops)

        # Phase 2: Create collage visualization
        # if args.vis:
        print("Phase 2: Creating collage visualization...")
        create_collage_video(args, params, cam_names, cam_mapper,
                            cur_cam_names, reader, selected_vid_idx)


def _process_batch(
    yolo_model, cpm, frame_buffer, bboxes_buffer,
    noframe_buffer, im_h, im_w, box_score_threshold,
    refine_crops, kps_f, vis_writer=None
):
    """Run YOLO + ViTPose on a batch of frames and write results."""
    with torch.no_grad():
        # stream=False for true batch processing, device='cuda' for GPU
        results = yolo_model(
            frame_buffer, verbose=False, stream=False, device='cuda'
        )
    process_all_yolo_results(
        results, bboxes_buffer, im_h, im_w,
        box_score_threshold, padding=5
    )

    with torch.no_grad():
        pred_poses = cpm.predict_pose_batch(
            frame_buffer, bboxes_buffer
        )

    all_kps = process_vitpose_results(
        pred_poses, frame_buffer, cpm, im_w, im_h,
        refine_crops=refine_crops
    )

    for i, kps in enumerate(all_kps):
        if noframe_buffer[i]:
            kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            kps[:, 2] = 1.0
        ujson.dump(kps.reshape(-1).tolist(), kps_f)
        kps_f.write('\n')

        if vis_writer is not None and not noframe_buffer[i]:
            vis_frame = draw_keypoints_on_image(
                frame_buffer[i], kps
            )
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            vis_writer.write(vis_frame)


if __name__ == '__main__':
    main()
