"""
Immediate proceeder: scripts_body/keypoints_2d_yolo_vitpose.py
This file: get full-body person mask (brics-studio), conditioned on 2D keypoints and bbox.
    Reads COCO-WholeBody 133-keypoint format and uses body keypoints to locate the person.
    This will be used to optimize SMPL-X shape parameters at smplx_em.py
Immediate successor: scripts_body/keypoints_3d_fast.py

Output mask: 0=empty, 1=person
"""

import argparse
import gc
import os

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
from src.utils.video_handler import load_first_frame
from easymocap.mytools.vis_base import get_row_col

# COCO-WholeBody: use all body+foot keypoints (0-22) for person bbox
BODY_IDX = slice(0, 23)
NUM_KEYPOINTS = 133


def compute_person_bbox(keypoints, valid_mask, im_w, im_h, padding_ratio=0.05):
    """Compute bounding box around person keypoints with padding.

    Args:
        keypoints: (N, 3) array of keypoints [x, y, confidence]
        valid_mask: (N,) boolean array indicating valid keypoints
        im_w: image width
        im_h: image height
        padding_ratio: ratio of bbox size to add as padding

    Returns:
        bbox: [xmin, ymin, xmax, ymax] or None
    """
    valid_points = keypoints[valid_mask][:, :2]
    if len(valid_points) == 0:
        return None

    xmin, ymin = valid_points.min(axis=0)
    xmax, ymax = valid_points.max(axis=0)

    width = xmax - xmin
    height = ymax - ymin
    padding_x = int(padding_ratio * width)
    padding_y = int(padding_ratio * height)

    xmin = max(0, int(xmin - padding_x))
    ymin = max(0, int(ymin - padding_y))
    xmax = min(im_w - 1, int(xmax + padding_x))
    ymax = min(im_h - 1, int(ymax + padding_y))

    return [xmin, ymin, xmax, ymax]


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument(
        "--sam_path",
        type=str,
        default="/oscar/data/ssrinath/datasets/sam3/sam3.pt",
        help="Path to sam3.pt"
    )
    parser.add_argument(
        "--collage_only", action="store_true",
        help="If true, only create the collage image from existing per-camera masks"
    )
    args = parser.parse_args()

    # paths
    base_path = args.root_dir
    image_base = os.path.join(base_path, args.seq_path)
    output_path = os.path.join(args.out_dir, "body")

    # load cameras
    params_txt = "optim_params.txt" if args.use_optim_params else "params.txt"
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

    # cameras (brics-studio: keep all cameras)
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

    # which video(s) to process
    if args.ith == -1:
        total_video_idxs = 0
        anchor_camera_by_length = None
        for fid, folder in enumerate(os.listdir(image_base)):
            if 'cam' in folder and folder not in cams_to_remove:
                length = len([f for f in os.listdir(os.path.join(image_base, folder)) if f.endswith('.mp4')])
                if length > total_video_idxs:
                    total_video_idxs = length
                    anchor_camera_by_length = os.listdir(image_base)[fid]
        if args.start > 0:
            selected_vid_idxs = list(range(args.start, args.end if args.end > 0 else total_video_idxs))
        else:
            selected_vid_idxs = list(range(args.end if args.end > 0 else total_video_idxs))
    else:
        anchor_camera_by_length = None
        selected_vid_idxs = [args.ith]

    # video dir
    if args.video_dir:
        video_dir = os.path.join(args.video_dir, args.seq_path)
    else:
        video_dir = image_base

    for selected_vid_idx in selected_vid_idxs:
        print(f'Video ID {selected_vid_idx}...')

        keypoints2d_dir = os.path.join(
            output_path, "intermediate", "keypoints_2d", str(selected_vid_idx).zfill(3)
        )
        cam_mapper = map_camera_names(keypoints2d_dir, cam_names)

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

        intrs, projs, dist_intrs, dists, cameras = get_projections(
            args, params, cur_cam_names, cam_mapper, easymocap_format=True
        )
        print("Total cams:", len(intrs))

        mask_dir = os.path.join(output_path, "intermediate", "mask_2d", str(selected_vid_idx).zfill(3))
        os.makedirs(mask_dir, exist_ok=True)

        collage_images = []
        collage_camnames = []
        all_masks = {}       # camname -> (H, W) uint8, 0=empty 1=person
        seg_status = {}      # camname -> bool
        timestamp_used = {}  # camname -> timestamp directory name

        # Build SAM3 model once
        model = build_sam3_image_model(checkpoint_path=args.sam_path)
        processor = Sam3Processor(model)

        for v_idx, input_video_path in tqdm(enumerate(reader.vids), total=len(reader.vids)):
            if args.collage_only:
                break

            # read first frame only
            orig_img, im_h, im_w, ts_name = load_first_frame(
                input_video_path, use_parsed, args, intrs[v_idx], dist_intrs[v_idx], dists[v_idx]
            )

            camname = input_video_path.split('/')[-2]
            if camname not in cam_mapper:
                continue

            # read 2D keypoints (133-point COCO-WholeBody)
            kps_path = os.path.join(keypoints2d_dir, f"{cam_mapper[camname]}.jsonl")
            if not os.path.exists(kps_path):
                continue
            with open(kps_path, "r") as f:
                first_line = f.readline()
            keypoints_all = np.array(ujson.loads(first_line)).reshape(NUM_KEYPOINTS, 3)  # (133, 3)

            # use body+foot keypoints (0-22) to locate person
            body_kps = keypoints_all[BODY_IDX]  # (23, 3)
            conf_thresh = 0.3
            valid_body = (
                (body_kps[:, 2] > conf_thresh) &
                (body_kps[:, 0] >= 0) & (body_kps[:, 0] < im_w) &
                (body_kps[:, 1] >= 0) & (body_kps[:, 1] < im_h)
            )
            has_person = valid_body.sum() > 3

            person_mask = np.zeros((im_h, im_w), dtype=np.uint8)
            segmented = False

            if has_person:
                bbox = compute_person_bbox(body_kps, valid_body, im_w, im_h, padding_ratio=0.05)
                if bbox is not None:
                    orig_pil = Image.fromarray(orig_img)
                    inference_state = processor.set_image(orig_pil)
                    output = processor.set_text_prompt(state=inference_state, prompt="person")

                    if output and 'masks' in output:
                        masks = output['masks'].detach().cpu().numpy()[:, 0, ...]
                        scores = output['scores'].detach().cpu().numpy() if 'scores' in output else None
                        if scores is not None and len(scores) > 0:
                            best_mask = masks[np.argmax(scores)]
                        else:
                            best_mask = masks[0] if len(masks) > 0 else None

                        if best_mask is not None:
                            person_mask[best_mask > 0] = 1
                            segmented = True

            all_masks[camname] = person_mask
            seg_status[camname] = segmented
            timestamp_used[camname] = ts_name

            # collage visualization: dim background, vivid green on mask
            vis = orig_img.astype(np.float32)
            if segmented:
                mask_bool = person_mask > 0
                vis[~mask_bool] *= 0.25  # dim background to 25%
                # blend vivid green onto mask area
                vis[mask_bool] = vis[mask_bool] * 0.45 + np.array([30, 220, 80], dtype=np.float32) * 0.55
            vis = np.clip(vis, 0, 255).astype(np.uint8)

            fig_collage, ax_collage = plt.subplots(1, 1, figsize=(8, 6))
            title = f"{cam_mapper[camname]}"
            if ts_name is not None:
                title += f"\n({ts_name})"
            ax_collage.imshow(vis)
            ax_collage.set_title(title, fontsize=8)
            ax_collage.axis('off')
            fig_collage.tight_layout()
            fig_collage.canvas.draw()
            collage_img = np.frombuffer(fig_collage.canvas.tostring_rgb(), dtype=np.uint8)
            collage_img = collage_img.reshape(fig_collage.canvas.get_width_height()[::-1] + (3,))
            collage_images.append(collage_img)
            collage_camnames.append(cam_mapper[camname])
            plt.close(fig_collage)

        # save all masks
        mask_save_path = os.path.join(mask_dir, "body_masks.npz")
        save_dict = {}
        for cam_name, mask in all_masks.items():
            save_dict[cam_name] = mask
        for cam_name, status_val in seg_status.items():
            save_dict[f"{cam_name}_seg"] = status_val
        for cam_name, ts in timestamp_used.items():
            save_dict[f"{cam_name}_timestamp"] = ts if ts is not None else "N/A"
        np.savez_compressed(mask_save_path, **save_dict)
        print(f"\nSaved all masks to {mask_save_path}")
        print(f"  - {len(all_masks)} camera views")

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

        # create collage image
        if len(collage_images) > 0:
            print(f"\nCreating collage with {len(collage_images)} camera views...")
            nrows, ncols = get_row_col(len(collage_images), square=False)
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
            collage_path = os.path.join(
                output_path, "intermediate", "mask_2d", f"collage_{str(selected_vid_idx).zfill(3)}.png"
            )
            plt.savefig(collage_path, dpi=150, bbox_inches='tight')
            plt.close(fig_grid)
            print(f"Saved collage: {collage_path}")


if __name__ == "__main__":
    main()
