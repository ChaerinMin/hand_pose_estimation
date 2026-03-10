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
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tqdm import tqdm
import ujson

from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.parser import add_common_args
import src.utils.params as param_utils
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess, load_first_frame
from easymocap.mytools.vis_base import merge, get_row_col


def compute_hand_bbox(keypoints, valid_mask, im_w, im_h, padding_ratio=0.2):
    """Compute bounding box around hand keypoints with padding

    Args:
        keypoints: (21, 3) array of keypoints [x, y, confidence]
        valid_mask: (21,) boolean array indicating valid keypoints
        im_w: image width
        im_h: image height
        padding_ratio: ratio of bbox size to add as padding (e.g., 0.2 = 20%)

    Returns:
        bbox: [xmin, ymin, xmax, ymax] in absolute coordinates
    """
    valid_points = keypoints[valid_mask][:, :2]  # (n_points, 2)
    if len(valid_points) == 0:
        return None

    xmin, ymin = valid_points.min(axis=0)
    xmax, ymax = valid_points.max(axis=0)

    # Add padding
    width = xmax - xmin
    height = ymax - ymin
    padding_x = int(padding_ratio * width)
    padding_y = int(padding_ratio * height)

    xmin = max(0, int(xmin - padding_x))
    ymin = max(0, int(ymin - padding_y))
    xmax = min(im_w - 1, int(xmax + padding_x))
    ymax = min(im_h - 1, int(ymax + padding_y))

    return [xmin, ymin, xmax, ymax]

def crop_image(image, bbox):
    """Crop image given bounding box

    Args:
        image: numpy array or PIL Image
        bbox: [xmin, ymin, xmax, ymax]

    Returns:
        cropped_image: PIL Image
        bbox: same bbox for reference
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    xmin, ymin, xmax, ymax = bbox
    cropped = image.crop((xmin, ymin, xmax, ymax))
    return cropped, bbox
    
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

# remove cameras
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
    if args.video_dir:
        video_dir = os.path.join(args.video_dir, args.seq_path)
    else:
        video_dir = image_base
    reader = Reader(
        args.input_type, video_dir, cam_names=cam_names,
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
    mask_dir = os.path.join(output_path, "mask_2d", str(selected_vid_idx).zfill(3))
    os.makedirs(mask_dir, exist_ok=True)

    # Store visualization images for collage
    collage_images = []
    collage_camnames = []

    # Store all masks in a dictionary (will save as single npz)
    # Keys: camname, Values: (H, W) array with 0=empty, 1=left hand, 2=right hand
    all_masks = {}
    # Store segmentation success status
    # Keys: camname_left/camname_right, Values: boolean
    seg_status = {}

    # Build SAM3 model once (reuse for all images)
    model = build_sam3_image_model(checkpoint_path=args.sam_path)
    processor = Sam3Processor(model)

    for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids)):
        if args.collage_only:
            break

        # read first frame only
        orig_img, im_h, im_w, _ = load_first_frame(
            input_video_path, use_parsed, args, intrs[v_idx], dist_intrs[v_idx], dists[v_idx]
        )
        orig_image = Image.fromarray(orig_img)

        # read 2D keypoints for the first frame
        camname = input_video_path.split('/')[-2]
        ap_keypoints_path_left = os.path.join(
            keypoints2d_dir_left, f"{cam_mapper[camname]}.jsonl"
        )
        ap_keypoints_path_right = os.path.join(
            keypoints2d_dir_right, f"{cam_mapper[camname]}.jsonl"
        )
        # Read only first line from each file
        with open(ap_keypoints_path_left, "r") as fl:
            first_line_left = fl.readline()
            keypoints2d_left = np.array(ujson.loads(first_line_left)).reshape(-1, 3)  # (21, 3)
        with open(ap_keypoints_path_right, "r") as fr:
            first_line_right = fr.readline()
            keypoints2d_right = np.array(ujson.loads(first_line_right)).reshape(-1, 3)  # (21, 3)

        # keypoint validity (for single frame now)
        valid_left = np.logical_not(
            (keypoints2d_left[:, 0] == 0) &\
                 (keypoints2d_left[:, 1] == 0) &\
                     (keypoints2d_left[:, 2] == 1)
        )
        valid_left = np.logical_and(
            valid_left,
            (keypoints2d_left[:, 0] < im_w) & (keypoints2d_left[:, 0] >= 0) &
            (keypoints2d_left[:, 1] < im_h) & (keypoints2d_left[:, 1] >= 0)
        ) # (21,)
        valid_right = np.logical_not(
            (keypoints2d_right[:, 0] == 0) &\
                 (keypoints2d_right[:, 1] == 0) &\
                     (keypoints2d_right[:, 2] == 1)
        )
        valid_right = np.logical_and(
            valid_right,
            (keypoints2d_right[:, 0] < im_w) & (keypoints2d_right[:, 0] >= 0) &
            (keypoints2d_right[:, 1] < im_h) & (keypoints2d_right[:, 1] >= 0)
        ) # (21,)

        # check if hands are detected
        has_left_hand = valid_left.sum() > 5
        has_right_hand = valid_right.sum() > 5

        # Compute bounding boxes for cropping
        bbox_left = None
        bbox_right = None
        if has_left_hand:
            bbox_left = compute_hand_bbox(keypoints2d_left, valid_left, im_w, im_h, padding_ratio=0.3)
        if has_right_hand:
            bbox_right = compute_hand_bbox(keypoints2d_right, valid_right, im_w, im_h, padding_ratio=0.3)

        # Initialize combined mask (0=empty, 1=left hand, 2=right hand)
        combined_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        # Track segmentation success
        left_segmented = False
        right_segmented = False

        # Process left hand
        if has_left_hand and bbox_left is not None:
            # Crop image around left hand
            cropped_left, bbox_l = crop_image(orig_img, bbox_left)

            # Set image and prompt with text (reuse processor)
            inference_state = processor.set_image(cropped_left)
            output = processor.set_text_prompt(state=inference_state, prompt="hand")

            # Extract mask
            if output and 'masks' in output:
                # Get the mask with highest score
                masks = output['masks'].detach().cpu().numpy()[:, 0, ...]  # (num_preds, obj, H, W)
                scores = output['scores'].detach().cpu().numpy() if 'scores' in output else None

                if scores is not None and len(scores) > 0:
                    best_mask_idx = np.argmax(scores)
                    cropped_mask = masks[best_mask_idx]
                else:
                    cropped_mask = masks[0] if len(masks) > 0 else None

                if cropped_mask is not None:
                    # Paste mask back to original image coordinates (mark as 1 for left hand)
                    combined_mask[bbox_left[1]:bbox_left[3], bbox_left[0]:bbox_left[2]][cropped_mask > 0] = 1
                    left_segmented = True

        # Process right hand
        if has_right_hand and bbox_right is not None:
            # Crop image around right hand
            cropped_right, bbox_r = crop_image(orig_img, bbox_right)

            # Set image and prompt with text (reuse processor)
            inference_state = processor.set_image(cropped_right)
            output = processor.set_text_prompt(state=inference_state, prompt="hand")

            # Extract mask
            if output and 'masks' in output:
                # Get the mask with highest score
                masks = output['masks'].detach().cpu().numpy()[:, 0, ...]
                scores = output['scores'].detach().cpu().numpy() if 'scores' in output else None

                if scores is not None and len(scores) > 0:
                    best_mask_idx = np.argmax(scores)
                    cropped_mask = masks[best_mask_idx]
                else:
                    cropped_mask = masks[0] if len(masks) > 0 else None

                if cropped_mask is not None:
                    # Paste mask back to original image coordinates (mark as 2 for right hand)
                    combined_mask[bbox_right[1]:bbox_right[3], bbox_right[0]:bbox_right[2]][cropped_mask > 0] = 2
                    right_segmented = True

        # Store mask and segmentation status
        all_masks[camname] = combined_mask
        seg_status[f"{camname}_left"] = left_segmented
        seg_status[f"{camname}_right"] = right_segmented

        # Store for collage - create a combined visualization with both hands
        mask_left = (combined_mask == 1)
        mask_right = (combined_mask == 2)
        fig_collage, ax_collage = plt.subplots(1, 1, figsize=(8, 6))
        ax_collage.imshow(orig_img)
        if left_segmented:
            ax_collage.imshow(mask_left, alpha=0.4, cmap='Reds')
        if right_segmented:
            ax_collage.imshow(mask_right, alpha=0.4, cmap='Blues')
        ax_collage.set_title(f"{cam_mapper[camname]}")
        ax_collage.axis('off')

        # Convert figure to numpy array for collage
        fig_collage.tight_layout()
        fig_collage.canvas.draw()
        collage_img = np.frombuffer(fig_collage.canvas.tostring_rgb(), dtype=np.uint8)
        collage_img = collage_img.reshape(fig_collage.canvas.get_width_height()[::-1] + (3,))
        collage_images.append(collage_img)
        collage_camnames.append(cam_mapper[camname])
        plt.close(fig_collage)

    # Save all masks and segmentation status to a single npz file
    mask_save_path = os.path.join(mask_dir, "hand_masks.npz")
    save_dict = {}
    # Add all masks
    for cam_name, mask in all_masks.items():
        save_dict[cam_name] = mask
    # Add segmentation status
    for status_key, status_val in seg_status.items():
        save_dict[status_key] = status_val

    np.savez_compressed(mask_save_path, **save_dict)
    print(f"\nSaved all masks to {mask_save_path}")
    print(f"  - {len(all_masks)} camera views")
    print(f"  - Segmentation status saved for each hand")

    # Clean up SAM3 model after processing all images
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # Create collage of all camera views
    if len(collage_images) > 0:
        print(f"\nCreating collage with {len(collage_images)} camera views...")

        # Use easymocap's get_row_col to determine grid layout
        nrows, ncols = get_row_col(len(collage_images), square=False)

        # Create grid collage
        fig_grid, axes_grid = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4.5))
        if nrows == 1 and ncols == 1:
            axes_grid = np.array([[axes_grid]])
        elif nrows == 1 or ncols == 1:
            axes_grid = axes_grid.reshape(nrows, ncols)

        for idx in range(nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            ax = axes_grid[row, col]

            if idx < len(collage_images):
                ax.imshow(collage_images[idx])
            ax.axis('off')

        plt.tight_layout()
        collage_path = os.path.join(output_path, "mask_2d", f"collage_{str(selected_vid_idx).zfill(3)}.png")
        plt.savefig(collage_path, dpi=150, bbox_inches='tight')
        plt.close(fig_grid)
        print(f"Saved collage: {collage_path}")
