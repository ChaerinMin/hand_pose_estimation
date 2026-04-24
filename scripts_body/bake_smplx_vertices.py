"""
Evaluate the fitted SMPL-X model on every frame and save the resulting vertex
sequence (+ faces, + optional camera extrinsics) as a single .npz that can be
loaded in aitviewer via `aitviewer.renderables.meshes.Meshes`.

Why bake: the params were fit in a scaled space and EasyMocap's (Rh, Th) layout
doesn't match aitviewer's (global_orient, transl) convention. Evaluating the
model here sidesteps all of that — aitviewer just has to render plain meshes.

Output layout (smplx_verts.npz):
  vertices: (N, 10475, 3)  float32  world-space, real scale
  faces:    (F, 3)         uint32
  frame_ids: (N,)          int32    original frame indices (chosen_frames)
  camera_names: (C,)       str      (optional, if --save_cameras)
  camera_K: (C, 3, 3)      float32
  camera_R: (C, 3, 3)      float32
  camera_T: (C, 3)         float32
"""

import argparse
import os
import sys

import numpy as np
import ujson
from tqdm import tqdm

import src.utils.params as param_utils
from src.utils.cameras import (get_projections, map_camera_names,
                               removed_cameras)
from src.utils.easymocap_utils import load_model
from src.utils.parser import add_common_args
from easymocap.mytools import Timer
from easymocap.smplmodel import select_nf
from easymocap.smplmodel.body_model import SMPLlayer

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.append(".")
sys.path.append("./third-party/EasyMocap")


# Same joint indexing as scripts_body/smplx_em.py / vis_selected_views.py
COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def coco_wholebody_to_body25_hands(keypoints):
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


def estimate_scale(body_model, kp3ds, kintree):
    kintree = np.array(kintree, dtype=int)
    src, dst = kintree[:, 0], kintree[:, 1]
    vecs_obs = kp3ds[:, dst, :3] - kp3ds[:, src, :3]
    L_obs = np.linalg.norm(vecs_obs, axis=2)
    conf_obs = np.minimum(kp3ds[:, src, 3], kp3ds[:, dst, 3])
    params0 = body_model.init_params(nFrames=1)
    kpts_model = body_model(return_verts=False, return_tensor=False,
                            only_shape=True, **params0)[0]
    vecs_model = kpts_model[dst, :3] - kpts_model[src, :3]
    L_model = np.linalg.norm(vecs_model, axis=1)
    ratios = []
    for ts in range(kp3ds.shape[0]):
        for li in range(L_model.shape[0]):
            if L_obs[ts, li] > 1e-8 and L_model[li] > 1e-8 and conf_obs[ts, li] > 0.1:
                ratios.append(L_model[li] / (L_obs[ts, li] + 1e-8))
    return float(np.median(ratios))


def parse_args():
    p = argparse.ArgumentParser("Bake SMPL-X vertices for aitviewer")
    add_common_args(p)
    p.add_argument('--model', type=str, default='smplx', choices=['smpl', 'smplh', 'smplx'])
    p.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    p.add_argument('--remove_side_cam', type=bool, default=True)
    p.add_argument('--remove_bottom_cam', type=bool, default=True)
    p.add_argument('--use_optim_params', action='store_true')
    p.add_argument('--optimize_bad_views', action='store_true')
    p.add_argument('--use_filtered', action='store_true')
    p.add_argument('--fix_hands', action=argparse.BooleanOptionalAction, default=False,
                   help='Zero out wrist + hand PCA so hands are rigid w.r.t. forearm.')
    p.add_argument('--save_cameras', action='store_true',
                   help='Also save camera K/R/T so you can place them in aitviewer.')
    p.add_argument('--out_npz', type=str, default=None,
                   help='Output npz path. Defaults to body/aitviewer/{ith}.npz')
    return p.parse_args()


def main():
    args = parse_args()

    # Paths
    output_path = os.path.join(args.out_dir, "body")
    calib_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence,
                             "calib", f"stage{args.stage}", "sparse", "0")
    params_txt = "new_params.txt" if args.optimize_bad_views else (
        "optim_params.txt" if args.use_optim_params else "params.txt")
    params_path = os.path.join(calib_dir, params_txt)
    assert os.path.exists(params_path), f"Params file not found: {params_path}"

    if args.stage == 1:
        cam_params = param_utils.read_params(params_path, distortion=True, args=args)
    else:
        cam_params = param_utils.read_params(params_path, distortion=False, args=args)

    cam_names_all = list(cam_params[:]["cam_name"])
    removed_camera_path = os.path.join(calib_dir, 'ignore_camera.txt')
    ignored = [l.rstrip() for l in open(removed_camera_path)] if os.path.isfile(removed_camera_path) else None
    cams_to_remove = removed_cameras(remove_side=args.remove_side_cam,
                                     remove_bottom=args.remove_bottom_cam,
                                     ignored_cameras=ignored)
    cam_names = [c for c in cam_names_all if c not in cams_to_remove]

    selected_vid_idx = args.ith

    keypoints3d_dir = os.path.join(output_path, "intermediate", "keypoints_3d",
                                    str(selected_vid_idx).zfill(3))
    keypoints2d_dir = os.path.join(output_path, "intermediate", "keypoints_2d",
                                    str(selected_vid_idx).zfill(3))
    keypt3d_file = os.path.join(keypoints3d_dir, "wholebody.jsonl")
    params_file = os.path.join(output_path, "smplx_params",
                                f"{str(selected_vid_idx).zfill(3)}.json")
    assert os.path.exists(keypt3d_file), f"Not found: {keypt3d_file}"
    assert os.path.exists(params_file), f"Not found: {params_file}"

    cam_mapper = map_camera_names(keypoints2d_dir, cam_names)

    # chosen_frames
    import json as _json
    if args.use_filtered:
        chosen_path = os.path.join(keypoints3d_dir, "chosen_frames.json")
        if os.path.exists(chosen_path):
            with open(chosen_path) as f:
                chosen_frames = sorted(set(_json.load(f)))
        else:
            chosen_frames = list(range(args.start, args.end, args.stride))
    else:
        chosen_frames = list(range(args.start, args.end, args.stride))
    chosen_frames = sorted(chosen_frames)
    print(f"Baking {len(chosen_frames)} frames...")

    # Load model + saved params
    print(f"Loading {args.model} ({args.gender})...")
    with Timer(f"Loading {args.model}", not False):
        body_model: SMPLlayer = load_model(
            gender=args.gender, model_type=args.model, model_path="data/smplx",
            use_pose_blending=True, use_shape_blending=True,
            use_pca=False, use_flat_mean=False,
        )
    with open(params_file) as f:
        raw = ujson.load(f)
    params_body = {k: np.asarray(v) for k, v in raw.items()}

    if args.fix_hands:
        assert params_body['poses'].shape[-1] == 87
        params_body['poses'][:, 60:78] = 0.0
        print("fix_hands: zeroed poses[60:78] (wrists + hand PCA).")

    # Scale (mirror smplx_em.py fitting pipeline)
    kp3d_raw = []
    with open(keypt3d_file) as f:
        for idx, line in enumerate(f):
            if idx in chosen_frames:
                kp3d_raw.append(np.array(ujson.loads(line)).reshape(-1, 4))
    kp3d_raw = np.asarray(kp3d_raw)
    kp3d_bh = coco_wholebody_to_body25_hands(kp3d_raw)
    from easymocap.dataset import CONFIG
    scale = estimate_scale(body_model, kp3d_bh, kintree=CONFIG['bodyhand']['kintree'])
    final_scale = 1.0 / scale
    root = kp3d_bh[:, 0:1, :3] * 1.0
    print(f"Estimated scale={scale:.4f} -> final_scale={final_scale:.4f}")

    # Bake
    n_params_frames = params_body['Rh'].shape[0]
    n_use = min(len(chosen_frames), n_params_frames)
    verts_all = np.empty((n_use, 10475, 3), dtype=np.float32)
    for i in tqdm(range(n_use)):
        pf = select_nf(params_body, i)
        v = body_model(return_verts=True, return_tensor=False, **pf).squeeze(0)
        v = (v - root[i]) * final_scale + root[i]
        verts_all[i] = v.astype(np.float32)

    out_payload = {
        'vertices': verts_all,
        'faces': np.asarray(body_model.faces).astype(np.uint32),
        'frame_ids': np.asarray(chosen_frames[:n_use], dtype=np.int32),
    }

    if args.save_cameras:
        _, _, _, _, cameras = get_projections(
            args, cam_params, cam_names, cam_mapper, easymocap_format=True
        )
        out_payload['camera_names'] = np.array(cameras['names'])
        out_payload['camera_K'] = cameras['K'].astype(np.float32)
        out_payload['camera_R'] = cameras['R'].astype(np.float32)
        out_payload['camera_T'] = cameras['T'].astype(np.float32)
        print(f"Saved {len(cameras['names'])} cameras.")

    if args.out_npz is None:
        out_dir = os.path.join(output_path, "aitviewer")
        os.makedirs(out_dir, exist_ok=True)
        args.out_npz = os.path.join(out_dir, f"{str(selected_vid_idx).zfill(3)}.npz")
    else:
        os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)

    np.savez_compressed(args.out_npz, **out_payload)
    size_mb = os.path.getsize(args.out_npz) / 1e6
    print(f"Wrote {args.out_npz} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
