"""
Temporary script: re-render SMPL-X fit into a 3x3 grid from a user-specified
set of 9 cameras, without the red border and without hand keypoints.

Reads saved SMPL-X parameters + keypoints from the body/ output directory
(same layout produced by scripts_body/smplx_em.py) and renders only the
requested 9 views per frame, merged into a 3x3 collage. Writes per-frame
jpgs and an mp4.

Example:
    python scripts_body/vis_selected_views.py \
        --setting brics-studio \
        --root_dir /oscar/data/ssrinath/public \
        -s 2025-04-05 \
        -m multisequence000001 \
        --ith 0 \
        --out_dir /oscar/data/ssrinath/public/brics-studio/2025-04-05/multisequence000001/outputs \
        --undistort \
        --vis_smpl \
        --cams bric-rev1-002_cam0.jpg bric-rev1-005_cam0.jpg ... (9 cams total)
"""

import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np
import ujson
from tqdm import tqdm

import src.utils.params as param_utils
from src.utils.cameras import (get_projections, map_camera_names,
                               removed_cameras)
from src.utils.easymocap_utils import load_model, projectN3
from src.utils.parser import add_common_args
from src.utils.reader_v2 import Reader
from src.utils.video_handler import convert_video_ffmpeg, create_video_writer
from easymocap.mytools import Timer
from easymocap.mytools.file_utils import get_bbox_from_pose
from easymocap.mytools.vis_base import plot_keypoints, merge
from easymocap.smplmodel import select_nf
from easymocap.smplmodel.body_model import SMPLlayer
from easymocap.visualize.renderer import Renderer

os.environ['PYOPENGL_PLATFORM'] = 'egl'
sys.path.append(".")
sys.path.append("./third-party/EasyMocap")

# Same mapping used in scripts_body/smplx_em.py
COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def coco_wholebody_to_body25_hands(keypoints):
    """Same as scripts_body/smplx_em.py (67-joint Body25+Hands format)."""
    n_frames = keypoints.shape[0]
    n_coords = keypoints.shape[-1]
    out = np.zeros((n_frames, 67, n_coords))
    coco_body = keypoints[:, :17, :]
    for ci in range(17):
        out[:, COCO17_IN_BODY25[ci], :] = coco_body[:, ci, :]
    out[:, 1, :n_coords - 1] = (out[:, 2, :n_coords - 1] + out[:, 5, :n_coords - 1]) / 2
    out[:, 1, n_coords - 1] = np.minimum(out[:, 2, n_coords - 1], out[:, 5, n_coords - 1])
    out[:, 8, :n_coords - 1] = (out[:, 9, :n_coords - 1] + out[:, 12, :n_coords - 1]) / 2
    out[:, 8, n_coords - 1] = np.minimum(out[:, 9, n_coords - 1], out[:, 12, n_coords - 1])
    if keypoints.shape[1] >= 23:
        out[:, 19:22, :] = keypoints[:, 17:20, :n_coords]
        out[:, 22:25, :] = keypoints[:, 20:23, :n_coords]
    out[:, 26:46, :] = keypoints[:, 92:112, :n_coords]
    out[:, 47:67, :] = keypoints[:, 113:133, :n_coords]
    return out


def get_body25_config():
    from easymocap.dataset import CONFIG
    return copy.deepcopy(CONFIG['body25'])


def get_bodyhand_config():
    from easymocap.dataset import CONFIG
    return copy.deepcopy(CONFIG['bodyhand'])


def estimate_scale(body_model, kp3ds, kintree):
    """Same scale estimator as scripts_body/smplx_em.py."""
    kintree = np.array(kintree, dtype=int)
    src, dst = kintree[:, 0], kintree[:, 1]
    vecs_obs = kp3ds[:, dst, :3] - kp3ds[:, src, :3]
    L_obs = np.linalg.norm(vecs_obs, axis=2)
    conf_obs = np.minimum(kp3ds[:, src, 3], kp3ds[:, dst, 3])
    params0 = body_model.init_params(nFrames=1)
    kpts_model = body_model(return_verts=False, return_tensor=False, only_shape=True, **params0)[0]
    vecs_model = kpts_model[dst, :3] - kpts_model[src, :3]
    L_model = np.linalg.norm(vecs_model, axis=1)
    ratios = []
    for ts in range(kp3ds.shape[0]):
        for li in range(L_model.shape[0]):
            if L_obs[ts, li] > 1e-8 and L_model[li] > 1e-8 and conf_obs[ts, li] > 0.1:
                ratios.append(L_model[li] / (L_obs[ts, li] + 1e-8))
    return float(np.median(ratios))


def vis_smpl_no_border(vertices, faces, images, cameras, renderer, add_back=True):
    """Render mesh over images -- no confident / red border."""
    render_data = {0: {'vertices': vertices, 'faces': faces, 'vid': 0, 'name': 'human_0'}}
    return renderer.render(render_data, cameras, images, add_back=add_back, confident=None)


def vis_repro_no_border(images, kpts_repro, config):
    """Draw reprojected keypoints -- no red border, no camera name text."""
    out = []
    for nv, image in enumerate(images):
        img = image.copy()
        kps = kpts_repro[nv]
        plot_keypoints(img, kps, pid=0, config=config, use_limb_color=True, lw=4)
        out.append(img)
    return out


def parse_args():
    parser = argparse.ArgumentParser("Selected-view SMPL-X visualization (3x3 grid)")
    add_common_args(parser)
    parser.add_argument('--model', type=str, default='smplx', choices=['smpl', 'smplh', 'smplx'])
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--remove_side_cam', type=bool, default=True)
    parser.add_argument('--remove_bottom_cam', type=bool, default=True)
    parser.add_argument('--use_optim_params', action='store_true')
    parser.add_argument('--optimize_bad_views', action='store_true')
    parser.add_argument('--save_origin', action='store_true')
    parser.add_argument('--use_filtered', action='store_true')
    parser.add_argument('--vis_smpl', action='store_true', help='Overlay SMPL-X mesh')
    parser.add_argument('--vis_3d_repro', action='store_true', help='Overlay reprojected 3D keypoints')
    parser.add_argument('--vis_2d_repro', action='store_true', help='Overlay 2D detections')
    parser.add_argument('--cams', nargs='+', required=True,
                        help='9 camera names to visualize (e.g. bric-rev1-002_cam0.jpg ...)')
    parser.add_argument('--out_name', type=str, default='grid_3x3',
                        help='Subdirectory name under vis/ to write results to')
    parser.add_argument('--fix_hands', action=argparse.BooleanOptionalAction, default=True,
                        help='Zero out wrist + finger poses so hands follow the arm rigidly '
                             '(SMPL-X template hand). --no-fix_hands keeps the fitted hand pose.')
    parser.add_argument('--hide_hand_mesh', action=argparse.BooleanOptionalAction, default=False,
                        help='Drop faces skinned to wrist/finger joints so the rendered '
                             'mesh ends at the forearm (no hand geometry).')
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.cams) == 9, f"Expected 9 cameras, got {len(args.cams)}"

    # Normalize user-provided cam names to the form used in params.txt (drop .jpg ext)
    req_cams = [c.replace('.jpg', '').replace('.', '') for c in args.cams]

    # Paths
    base_path = args.root_dir
    image_dir = os.path.join(base_path, args.seq_path)
    output_path = os.path.join(args.out_dir, "body")
    calib_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence,
                             "calib", f"stage{args.stage}", "sparse", "0")
    params_txt = "new_params.txt" if args.optimize_bad_views else (
        "optim_params.txt" if args.use_optim_params else "params.txt")
    params_path = os.path.join(calib_dir, params_txt)
    assert os.path.exists(params_path), f"Params file not found: {params_path}"

    # Cameras
    if args.stage == 1:
        params = param_utils.read_params(params_path, distortion=True, args=args)
        use_parsed = False
    else:
        params = param_utils.read_params(params_path, distortion=False, args=args)
        use_parsed = True

    cam_names_all = list(params[:]["cam_name"])
    removed_camera_path = os.path.join(calib_dir, 'ignore_camera.txt')
    ignored = [l.rstrip() for l in open(removed_camera_path)] if os.path.isfile(removed_camera_path) else None
    cams_to_remove = removed_cameras(remove_side=args.remove_side_cam,
                                     remove_bottom=args.remove_bottom_cam,
                                     ignored_cameras=ignored)
    cam_names_full = [c for c in cam_names_all if c not in cams_to_remove]

    # Keep order identical to the user's --cams argument so it maps to the 3x3 grid slots
    missing = [c for c in req_cams if c not in cam_names_full]
    if missing:
        raise ValueError(f"Requested cameras not found in calibration / filtered list: {missing}")

    # Video index selection
    selected_vid_idx = args.ith

    # Reader is only needed in stage-1 mode (use_parsed=False) to read frames from mp4s.
    # In stage-2 (use_parsed=True) we read pre-extracted jpgs from parsed/timestamp_*/images/
    # so we can skip Reader entirely and avoid needing --video_dir.
    reader = None
    if not use_parsed:
        video_dir = os.path.join(args.video_dir, args.seq_path) if args.video_dir else image_dir
        reader = Reader("video", video_dir, cam_names=cam_names_full,
                        cams_to_remove=cams_to_remove, ith=selected_vid_idx,
                        anchor_camera=args.anchor_camera)
        cur_cam_names = [c for c in cam_names_full if c not in reader.to_delete]
    else:
        cur_cam_names = cam_names_full

    still_missing = [c for c in req_cams if c not in cur_cam_names]
    if still_missing:
        raise ValueError(f"Requested cams not available: {still_missing}")

    # Keypoint / param paths
    keypoints3d_dir = os.path.join(output_path, "intermediate", "keypoints_3d",
                                    str(selected_vid_idx).zfill(3))
    keypoints2d_dir = os.path.join(output_path, "intermediate", "keypoints_2d",
                                    str(selected_vid_idx).zfill(3))
    keypt3d_file = os.path.join(keypoints3d_dir, "wholebody.jsonl")
    params_file = os.path.join(output_path, "smplx_params",
                                f"{str(selected_vid_idx).zfill(3)}.json")
    assert os.path.exists(keypt3d_file), f"Not found: {keypt3d_file}"
    assert os.path.exists(params_file), f"Not found: {params_file}"

    cam_mapper = map_camera_names(keypoints2d_dir, cur_cam_names)

    # Chosen frames
    if args.use_filtered:
        chosen_path = os.path.join(keypoints3d_dir, "chosen_frames.json")
        if os.path.exists(chosen_path):
            with open(chosen_path) as f:
                chosen_frames = sorted(set(json.load(f)))
        else:
            chosen_frames = list(range(args.start, args.end, args.stride))
    else:
        chosen_frames = list(range(args.start, args.end, args.stride))
    chosen_frames = sorted(chosen_frames)
    print(f"Total frames to render: {len(chosen_frames)}")

    # Camera intrinsics / projections
    intrs, projs, dist_intrs, dists, cameras = get_projections(
        args, params, cur_cam_names, cam_mapper, easymocap_format=True
    )
    cam_mapper_list = [c for c in cur_cam_names if c in cam_mapper]
    # Index selection = user order
    vis_idx = [cam_mapper_list.index(c) for c in req_cams]
    vis_projs = [projs[i] for i in vis_idx]
    vis_cameras = {}
    for k, v in cameras.items():
        if k == 'names':
            vis_cameras[k] = [v[i] for i in vis_idx]
        elif isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(cam_mapper_list):
            vis_cameras[k] = v[vis_idx]
        else:
            vis_cameras[k] = v

    # Load SMPL-X model + params
    print(f"Loading {args.model} ({args.gender})...")
    with Timer(f"Loading {args.model}", not False):
        body_model: SMPLlayer = load_model(
            gender=args.gender, model_type=args.model,
            model_path="data/smplx",
            use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False,
        )
    with open(params_file) as f:
        raw = ujson.load(f)
    params_body = {k: np.asarray(v) for k, v in raw.items()}

    # Fix wrist + finger pose to SMPL-X template so hands rigidly follow the arm.
    # For SMPL-X poses layout (use_pca=True, use_flat_mean=True, NUM_POSES=87):
    #   [0:66]  body (22 joints * 3)  <- joints 20,21 are L/R wrists at [60:66]
    #   [66:72] left-hand PCA coeffs  -> 0 = flat/straight fingers (flat mean hand)
    #   [72:78] right-hand PCA coeffs -> 0 = flat/straight fingers
    #   [78:87] jaw + L/R eye
    if args.fix_hands:
        assert params_body['poses'].shape[-1] == 87, \
            f"Unexpected poses dim {params_body['poses'].shape[-1]} (expected 87 for smplx)"
        params_body['poses'][:, 60:78] = 0.0
        print("Fixed wrists (poses[60:66]) and hand PCA (poses[66:78]) to template.")

    # 3D keypoints (for root + optional 3D repro)
    kp3d_raw = []
    with open(keypt3d_file) as f:
        for idx, line in enumerate(f):
            if idx in chosen_frames:
                kp3d_raw.append(np.array(ujson.loads(line)).reshape(-1, 4))
    kp3d_raw = np.asarray(kp3d_raw)
    # Match smplx_em.py: convert to bodyhand (67 joints) and estimate scale from that kintree
    kp3d_bodyhand = coco_wholebody_to_body25_hands(kp3d_raw)
    bodyhand_cfg = get_bodyhand_config()
    scale = estimate_scale(body_model, kp3d_bodyhand, kintree=bodyhand_cfg['kintree'])
    final_scale = 1.0 / scale
    # root = first joint of each frame in observed (pre-scale) space (smplx_em.apply_scale_to_keypoints)
    root = kp3d_bodyhand[:, 0:1, :3] * 1.0

    # Vis config (body25 only - no hand)
    body25_cfg = get_body25_config()

    # Optional: drop faces belonging to the hand so only the arm mesh is rendered.
    # SMPL-X kintree -> joints 20 (L wrist), 21 (R wrist) and 25..54 (finger bones)
    # are the wrist + hand chain. A vertex whose dominant skinning weight falls on
    # any of these is considered a hand vertex, and any face containing a hand
    # vertex is dropped.
    render_faces = body_model.faces
    if args.hide_hand_mesh:
        W = body_model.weights.detach().cpu().numpy()  # (V, 55)
        dominant = W.argmax(axis=1)
        hand_joint_ids = set([20, 21]) | set(range(25, 55))
        hand_vert_mask = np.isin(dominant, list(hand_joint_ids))
        faces_np = np.asarray(body_model.faces)
        face_touches_hand = hand_vert_mask[faces_np].any(axis=1)
        render_faces = faces_np[~face_touches_hand].astype(body_model.faces.dtype)
        print(f"Hiding hand mesh: kept {render_faces.shape[0]}/{faces_np.shape[0]} faces "
              f"({hand_vert_mask.sum()}/{len(hand_vert_mask)} vertices classified as hand)")

    # Output directory (add suffix when hiding the hand mesh so we don't overwrite
    # the version with hands).
    out_name = args.out_name + ('_nohand' if args.hide_hand_mesh else '')
    out_dir = os.path.join(output_path, 'vis', out_name,
                           str(selected_vid_idx).zfill(3))
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = out_dir + ".mp4"

    # Renderer
    renderer = Renderer(height=1024, width=1024, faces=None) if args.vis_smpl else None

    # Frame generator for non-parsed mode
    if not use_parsed:
        generator = reader(chosen_frames)

    writer = None
    n_params_frames = params_body['Rh'].shape[0]

    for abs_idx, chosen_f in tqdm(list(enumerate(chosen_frames))):
        if use_parsed:
            parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")
            ts_dir = os.path.join(parsed_dir, f"timestamp_{chosen_f}", "images")
            frames = {}
            for cam in cur_cam_names:
                if cam in cam_mapper:
                    fp = os.path.join(ts_dir, f"{cam}.jpg")
                    if os.path.exists(fp):
                        frames[cam_mapper[cam]] = cv2.imread(fp)
                    else:
                        frames[cam_mapper[cam]] = np.ones(
                            (params["height"][0], params["width"][0], 3), dtype=np.uint8) * 255
        else:
            frames, _ = next(generator)

        # Undistort + collect the 9 images in user order
        full_images = []
        c_idx = 0
        for cam in cur_cam_names:
            if cam in cam_mapper:
                img = frames[cam_mapper[cam]]
                if args.undistort and not (dists[c_idx] == 0).all():
                    img = param_utils.undistort_image(intrs[c_idx], dist_intrs[c_idx], dists[c_idx], img)
                c_idx += 1
                full_images.append(img)
        vis_images = [full_images[i] for i in vis_idx]

        # Per-frame params (clamp if smoothing produced fewer frames)
        nf = min(abs_idx, n_params_frames - 1)
        param_frame = select_nf(params_body, nf)

        # Mesh overlay
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **param_frame)
            vertices = (vertices - root[abs_idx:abs_idx + 1]) * final_scale + root[abs_idx:abs_idx + 1]
            vertices = vertices.squeeze(0)
            rendered = vis_smpl_no_border(vertices, render_faces, vis_images,
                                           vis_cameras, renderer, add_back=True)
            base_for_joints = rendered
        else:
            base_for_joints = vis_images

        # Joint reprojection overlay (body25 only -> no hand)
        joints = body_model(return_verts=False, return_tensor=False, **param_frame)
        joints = (joints - root[abs_idx:abs_idx + 1]) * final_scale + root[abs_idx:abs_idx + 1]
        joints = joints.squeeze(0)
        joints_repro = projectN3(joints, vis_projs)
        joints_repro[:, :, 2] = 0.5
        overlaid = vis_repro_no_border(base_for_joints, joints_repro, config=body25_cfg)

        # 3x3 grid
        grid = merge(overlaid, row=3, col=3, resize=not args.save_origin)

        out_jpg = os.path.join(out_dir, f"{chosen_f:06d}.jpg")
        cv2.imwrite(out_jpg, grid)

        if writer is None:
            writer = create_video_writer(out_mp4, (grid.shape[1], grid.shape[0]), fps=30)
        writer.write(grid)

    if writer is not None:
        writer.release()
        convert_video_ffmpeg(out_mp4)
        print(f"Saved video to {out_mp4}")


if __name__ == '__main__':
    main()
