"""
Full-body 3D keypoint triangulation for COCO-WholeBody 133 keypoints.

Triangulates 2D keypoints from multiple camera views to 3D coordinates.
Optionally visualizes reprojected 3D keypoints on each camera view.

COCO-WholeBody 133 keypoint layout:
  Body:      0-16   (17 keypoints)
  Foot:      17-22  (6 keypoints)
  Face:      23-90  (68 keypoints)
  Left hand: 91-111 (21 keypoints)
  Right hand:112-132(21 keypoints)

Input: keypoints_2d/{vid_idx:03d}/{cam_name}.jsonl
  Each line is a JSON array of shape (133*3,) = [x0, y0, conf0, x1, y1, conf1, ...]

Output: keypoints_3d/{vid_idx:03d}/wholebody.jsonl
  Each line is a JSON array of shape (133, 4) = [[x, y, z, conf], ...]
"""

import argparse
import os
import sys
import glob
import ujson
import shutil

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")
from src.utils.reader_v2 import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.triangulate import triangulate_joints, ransac_processor
from src.utils.filter import apply_one_euro_filter_3d
from src.utils.video_handler import create_video_writer, convert_video_ffmpeg

sys.path.append("./EasyMocap")
from myeasymocap.operations.triangulate import SimpleTriangulate

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

# COCO-WholeBody skeleton connections
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


def projectN3(kpts3d, Pall):
    """Project 3D keypoints to multiple camera views.

    Args:
        kpts3d: (N_kpts, 4) array [x, y, z, conf]
        Pall: list of (3, 4) projection matrices

    Returns:
        kpts2d: (N_views, N_kpts, 3) array [x, y, conf]
    """
    N_views = len(Pall)
    N_kpts = kpts3d.shape[0]
    kpts2d = np.zeros((N_views, N_kpts, 3))

    for nv in range(N_views):
        kp_homo = np.hstack([kpts3d[:, :3], np.ones((N_kpts, 1))])  # (N_kpts, 4)
        kp_proj = (Pall[nv] @ kp_homo.T).T  # (N_kpts, 3)
        kpts2d[nv, :, :2] = kp_proj[:, :2] / kp_proj[:, 2:3]
        kpts2d[nv, :, 2] = kpts3d[:, 3]  # copy confidence

    return kpts2d


def draw_skeleton_on_image(image, keypoints_2d, conf_thresh=0.3):
    """Draw COCO-WholeBody skeleton on image.

    Args:
        image: BGR image
        keypoints_2d: (133, 3) array [x, y, conf]
        conf_thresh: minimum confidence to draw

    Returns:
        image with skeleton drawn
    """
    vis = image.copy()

    def draw_limbs(skeleton, offset, color, thickness=2):
        for (i, j) in skeleton:
            pi, pj = i + offset, j + offset
            if (keypoints_2d[pi, 2] > conf_thresh and keypoints_2d[pj, 2] > conf_thresh):
                pt1 = (int(keypoints_2d[pi, 0]), int(keypoints_2d[pi, 1]))
                pt2 = (int(keypoints_2d[pj, 0]), int(keypoints_2d[pj, 1]))
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
        kps = keypoints_2d[idx_slice]
        for k in range(kps.shape[0]):
            if kps[k, 2] > conf_thresh:
                pt = (int(kps[k, 0]), int(kps[k, 1]))
                cv2.circle(vis, pt, radius, color, -1, cv2.LINE_AA)

    return vis


def create_visualization_grid(images, grid_cols=None):
    """Create a grid collage from camera images.

    Args:
        images: list of BGR images (all same size)
        grid_cols: number of columns (auto if None)

    Returns:
        collage image
    """
    if len(images) == 0:
        return None

    n = len(images)
    if grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(n * 1.5)))
    grid_rows = int(np.ceil(n / grid_cols))

    h, w = images[0].shape[:2]
    collage = np.zeros((h * grid_rows, w * grid_cols, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        collage[row*h:(row+1)*h, col*w:(col+1)*w] = img

    return collage


def main():
    parser = argparse.ArgumentParser(description='Full-Body 3D Keypoint Triangulation')
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument("--to_smooth", action="store_true", help="Temporal smoothing of 3D keypoints")
    parser.add_argument("--all_frames", default=False, action="store_true")
    parser.add_argument("--easymocap", default=False, action="store_true", help='Use EasyMocap triangulation')
    parser.add_argument("--confidence_thresh", type=float, default=None, help="Camera confidence threshold")
    parser.add_argument("--optimize_bad_views", action="store_true", help="Optimize extrinsics of bad views")
    # parser.add_argument("--vis_repro", action="store_true", help="Visualize reprojected 3D keypoints")
    args = parser.parse_args()

    base_path = os.path.join(args.root_dir)
    image_base = os.path.join(base_path, args.seq_path)
    output_path = os.path.join(args.out_dir, "body")

    # Load camera parameters
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

    cams_to_remove = removed_cameras(
        remove_side=False, remove_bottom=False, ignored_cameras=ignored_cameras
    )
    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)

    # Select videos
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
        selected_vid_idxs = [args.ith]

    # Camera confidence
    if args.confidence_thresh is not None:
        # conf_dir = args.out_dir[:args.out_dir.index("/stage")]
        conf_path = os.path.join(
            args.root_dir, args.seq_path, args.multisequence, "calib", "image_confidence.json"
        )
        with open(conf_path, "r") as f:
            image_confidence = ujson.load(f)

    for selected_vid_idx in selected_vid_idxs:
        print(f'Video ID {selected_vid_idx}...')

        keypoints2d_dir = os.path.join(output_path, "intermediate", "keypoints_2d", str(selected_vid_idx).zfill(3))
        if not os.path.exists(keypoints2d_dir):
            print(f"Keypoints directory not found: {keypoints2d_dir}")
            continue

        # Get camera mapper
        if args.video_dir:
            video_dir = os.path.join(args.video_dir, args.seq_path)
        else:
            video_dir = image_base
        cam_mapper = map_camera_names(keypoints2d_dir, cam_names)

        # Get reader (used only to determine which cameras are valid)
        reader = Reader(
            args.input_type, video_dir, cam_names=cam_names,
            cams_to_remove=cams_to_remove, ith=selected_vid_idx,
            anchor_camera=anchor_camera_by_length if args.ith == -1 else args.anchor_camera,
            match_by_timestamp=(args.setting != "brics-mobile")
        )

        extra_cams_to_remove = reader.to_delete
        cur_cam_names = cam_names.copy()
        for cam in extra_cams_to_remove:
            if cam in cur_cam_names:
                cur_cam_names.remove(cam)

        # Derive frame count from JSONL files (not video clips)
        jsonl_frame_count = 0
        for cam in cur_cam_names:
            if cam in cam_mapper:
                kp_path = os.path.join(keypoints2d_dir, f"{cam_mapper[cam]}.jsonl")
                if os.path.exists(kp_path):
                    with open(kp_path, "r") as _f:
                        jsonl_frame_count = sum(1 for _ in _f)
                    break
        if jsonl_frame_count <= 0:
            print("No JSONL frames found")
            continue

        print("Total Views:", len(cur_cam_names))
        print("Total frames:", jsonl_frame_count)

        intrs, projs, dist_intrs, dists, cameras = get_projections(
            args, params, cur_cam_names, cam_mapper, easymocap_format=True
        )

        keypoints3d_dir = os.path.join(output_path, "intermediate", "keypoints_3d", str(selected_vid_idx).zfill(3))
        try:
            shutil.rmtree(keypoints3d_dir)
        except FileNotFoundError:
            pass
        os.makedirs(keypoints3d_dir)

        if args.all_frames:
            chosen_frames = range(0, jsonl_frame_count, 1)
        else:
            chosen_frames = range(args.start, args.end, args.stride)

        # Load 2D keypoints from all cameras
        all_keypoints2d = []  # (N_cams, N_frames, 133, 3)
        cam_confidence = []

        for cam in cur_cam_names:
            if cam in cam_mapper:
                keypoints2d = []
                kp_path = os.path.join(keypoints2d_dir, f"{cam_mapper[cam]}.jsonl")

                if not os.path.exists(kp_path):
                    print(f"Warning: Keypoint file not found: {kp_path}")
                    continue

                with open(kp_path, "r") as f:
                    for l_idx, line in enumerate(f):
                        if l_idx in chosen_frames:
                            kp_flat = np.array(ujson.loads(line))
                            kp = kp_flat.reshape(NUM_KEYPOINTS, 3)
                            keypoints2d.append(kp)

                keypoints2d = np.asarray(keypoints2d)  # (N_frames, 133, 3)

                # Check for invalid frames (all zeros with conf=1)
                valid = np.logical_not(
                    np.logical_and(
                        np.logical_and(
                            (keypoints2d[:, :, 0] == 0).all(axis=1),
                            (keypoints2d[:, :, 1] == 0).all(axis=1)
                        ),
                        (keypoints2d[:, :, 2] == 1).all(axis=1)
                    )
                )

                # Temporal smoothing
                if args.to_smooth:
                    for i in range(len(keypoints2d)):
                        if valid[i]:
                            # Smooth each keypoint separately
                            # Note: apply_one_euro_filter_3d expects (N_frames, N_points, 3)
                            pass  # TODO: implement frame-by-frame smoothing if needed

                all_keypoints2d.append(keypoints2d)

                if args.confidence_thresh is not None:
                    cam_conf = image_confidence[cam + ".jpg"]["num_visible_3D_points"]
                    cam_confidence.append(cam_conf)

        all_keypoints2d = np.asarray(all_keypoints2d)  # (N_cams, N_frames, 133, 3)
        cam_confidence = np.asarray(cam_confidence)

        # Triangulate keypoints
        keypt_file = os.path.join(keypoints3d_dir, "wholebody.jsonl")
        all_keypoints3d = []

        print(f"Writing 3D keypoints to {keypt_file}")

        chosen_frames_record = []
        with open(keypt_file, "w") as f3d:
            for l_idx in tqdm(range(jsonl_frame_count), total=all_keypoints2d.shape[1]):
                if l_idx >= all_keypoints2d.shape[1]:
                    break
                if args.end > 0 and l_idx > args.end:
                    break

                if l_idx in chosen_frames:
                    frame_idx = list(chosen_frames).index(l_idx)
                    keypoints2d = all_keypoints2d[:, frame_idx, :, :]  # (N_cams, 133, 3)

                    # Check valid cameras (not all zeros)
                    valid = np.logical_not(
                        np.logical_and(
                            np.logical_and(
                                (keypoints2d[:, :, 0] == 0).all(axis=1),
                                (keypoints2d[:, :, 1] == 0).all(axis=1)
                            ),
                            (keypoints2d[:, :, 2] == 1).all(axis=1)
                        )
                    )

                    if args.confidence_thresh is not None:
                        valid = np.logical_and(valid, cam_confidence >= float(args.confidence_thresh))

                    if not valid.any():
                        keypoints3d = np.zeros((NUM_KEYPOINTS, 4))
                        keypoints3d[:, 3] = 1
                    elif not args.easymocap:
                        keypoints3d, residuals = triangulate_joints(np.asarray(keypoints2d)[valid], np.asarray(projs)[valid], processor=ransac_processor, residual_threshold=10, min_samples=2)
                        print(f"Error: {residuals.mean()}")
                    else:
                        triangulation = SimpleTriangulate("ransac")
                        valid_cameras = {}
                        for k_cam in cameras:
                            if k_cam == "names":
                                continue
                            valid_cameras[k_cam] = cameras[k_cam][valid]
                        kp_in = np.asarray(keypoints2d)[valid]
                        keypoints3d = triangulation(kp_in, valid_cameras)['keypoints3d']
                        print(f"DEBUG iterative result: nonzero={((keypoints3d[:, 3]>0).sum())}, sample={keypoints3d[0]}", flush=True)
                    ujson.dump(keypoints3d.tolist(), f3d)
                    f3d.write('\n')
                    all_keypoints3d.append(keypoints3d)
                else:
                    ujson.dump(np.zeros((NUM_KEYPOINTS, 4)).tolist(), f3d)
                    f3d.write('\n')
                    all_keypoints3d.append(np.zeros((NUM_KEYPOINTS, 4)))
                
                chosen_frames_record.append(l_idx)

        all_keypoints3d = np.asarray(all_keypoints3d)  # (N_frames, 133, 4)

        # Temporal smoothing of 3D keypoints
        if args.to_smooth:
            print('Smoothing 3D keypoints...')
            valid_frames = all_keypoints3d[:, :, 3].sum(axis=1) > 0
            if valid_frames.sum() > 2:
                all_keypoints3d[valid_frames] = apply_one_euro_filter_3d(
                    all_keypoints3d[valid_frames],
                    mincutoff=0.5, beta=0.0, dcutoff=1.0
                )
                # Rewrite smoothed keypoints
                with open(keypt_file, "w") as f3d:
                    for kp3d in all_keypoints3d:
                        ujson.dump(kp3d.tolist(), f3d)
                        f3d.write('\n')

        # Optimize camera extrinsics
        if args.optimize_bad_views:
            # Reshape for optimization
            all_kp2d = all_keypoints2d.reshape(all_keypoints2d.shape[0], -1, 3)  # (N_cams, N_frames*133, 3)
            all_kp3d = all_keypoints3d.reshape(-1, 4)  # (N_frames*133, 4)

            new_rot, new_tr = param_utils.optimize_extrinsics(
                cameras, all_kp2d, all_kp3d, inspect_only=False
            )
            new_params_path = os.path.join(calib_dir, "new_params.txt")
            param_utils.update_extrinsics(new_params_path, params, new_rot, new_tr)
        else:
            all_kp2d = all_keypoints2d.reshape(all_keypoints2d.shape[0], -1, 3)
            all_kp3d = all_keypoints3d.reshape(-1, 4)
            param_utils.optimize_extrinsics(cameras, all_kp2d, all_kp3d, inspect_only=True)

        chosen_path = os.path.join(keypoints3d_dir, "chosen_frames.json")
        with open(chosen_path, "w") as f:
            ujson.dump(chosen_frames_record, f, indent=2)

        # Visualization
        # if args.vis_repro:
        print("Creating reprojection visualization...")
        vis_dir = os.path.join(output_path, "vis", "repro_3d", str(selected_vid_idx).zfill(3))
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, "repro.mp4")

        # Load images
        if use_parsed:
            # multiseq_dir = args.out_dir[:args.out_dir.index("calib")]
            parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")

        # Determine grid size
        grid_cols = int(np.ceil(np.sqrt(len(cur_cam_names) * 1.5)))

        # Get image dimensions from first frame
        im_h, im_w = int(params["height"][0]), int(params["width"][0])

        # Scale for collage
        grid_rows = int(np.ceil(len(cur_cam_names) / grid_cols))
        target_height = 1000
        scale_factor = target_height / (grid_rows * im_h)
        scaled_w = int(im_w * scale_factor)
        scaled_h = int(im_h * scale_factor)

        collage_w = scaled_w * grid_cols
        collage_h = scaled_h * grid_rows

        vis_writer = create_video_writer(vis_path, (collage_w, collage_h), fps=30)

        for frame_idx in tqdm(range(len(chosen_frames)), desc="Creating visualization"):
            if frame_idx >= all_keypoints3d.shape[0]:
                break
            chosen_f = chosen_frames[frame_idx]

            # Load images
            if use_parsed:
                timestamp_dir = os.path.join(parsed_dir, f"timestamp_{chosen_f}", "images")
                frames = {}
                for cam in cur_cam_names:
                    if cam in cam_mapper:
                        frame_path = os.path.join(timestamp_dir, f"{cam}.jpg")
                        if os.path.exists(frame_path):
                            frames[cam] = cv2.imread(frame_path)
                        else:
                            frames[cam] = np.ones((im_h, im_w, 3), dtype=np.uint8) * 255
            else:
                # Load from video (not implemented here)
                frames = {cam: np.ones((im_h, im_w, 3), dtype=np.uint8) * 255 for cam in cur_cam_names}

            # Project 3D keypoints to 2D
            kp3d = all_keypoints3d[frame_idx]
            kp2d_repro = projectN3(kp3d, projs)  # (N_cams, 133, 3)
            kp2d_repro[:, :, 2] = 0.5

            # Draw on each camera
            vis_images = []
            for cam_idx, cam in enumerate(cur_cam_names):
                if cam not in frames:
                    continue

                img = frames[cam].copy()

                # Undistort if needed
                if args.undistort and not (dists[cam_idx] == 0).all():
                    img = param_utils.undistort_image(
                        intrs[cam_idx], dist_intrs[cam_idx], dists[cam_idx], img
                    )

                # Draw skeleton
                img_vis = draw_skeleton_on_image(img, kp2d_repro[cam_idx], conf_thresh=0.3)

                # Add camera name
                cv2.putText(img_vis, cam, (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

                # Resize for collage
                img_vis_resized = cv2.resize(img_vis, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                vis_images.append(img_vis_resized)

            # Create collage
            collage = create_visualization_grid(vis_images, grid_cols=grid_cols)
            vis_writer.write(collage)

        vis_writer.release()
        convert_video_ffmpeg(vis_path)
        print(f"Visualization saved to: {vis_path}")


if __name__ == '__main__':
    main()
