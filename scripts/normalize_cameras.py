import sys

sys.path.append(".")
sys.path.append("./instant-ngp/build")
import os
import numpy as np
import json
from copy import deepcopy
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from src.utils.parser import add_common_args
import src.utils.params as param_utils


def save_extrinsics(params, extrs, params_path):
    optim_params = deepcopy(params)
    fields = params.dtype.fields
    for idx in range(extrs.shape[0]):
        # cx, cy = params["cx"][idx], params["cy"][idx]
        # width, height = params["width"][idx], params["height"][idx]
        # cx = cx * width
        # cy = cy * height
        # fx, fy = params["fx"][idx], params["fy"][idx]
        
        c2w = extrs[idx]
        # c2w = np.vstack((c2w, np.asarray([[0, 0, 0, 1]])))
        w2c = np.linalg.inv(c2w)
        qvec = Rotation.from_matrix(w2c[:3, :3]).as_quat()
        tvec = w2c[:3, 3]
        optim_params[idx]["qvecx"] = qvec[0]
        optim_params[idx]["qvecy"] = qvec[1]
        optim_params[idx]["qvecz"] = qvec[2]
        optim_params[idx]["qvecw"] = qvec[3]
        optim_params[idx]["tvecx"] = tvec[0]
        optim_params[idx]["tvecy"] = tvec[1]
        optim_params[idx]["tvecz"] = tvec[2]
        # optim_params[idx]["cx"] = cx
        # optim_params[idx]["cy"] = cy
        # optim_params[idx]["fx"] = fx
        # optim_params[idx]["fy"] = fy
    
    np.savetxt(
        os.path.join(os.path.dirname(params_path), "optim_params.txt"),
        optim_params,
        fmt="%s",
        header=" ".join(fields),
    )
       
def get_parser():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--network", type=str, required=False, default='./data/nerf_base.json')
    parser.add_argument("--action", type=str, default="")
    parser.add_argument("--cam_traj_path", type=str, default="")
    parser.add_argument("--separate_calib", action="store_true")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=227840)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--overwrite_segmentation", action="store_true")
    parser.add_argument("--aabb_scale", type=int, default=1)
    parser.add_argument("--num_objects", type=int, default=1)
    parser.add_argument("--camera_scale", type=float, default=1.0)
    parser.add_argument("--pad", nargs="+", default=[0.05, 0.05, 0.05], type=float)

    parser.add_argument(
        "--mesh_path",
        default="mesh.ply",
        help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.",
    )
    parser.add_argument(
        "--marching_cubes_res",
        default=256,
        type=int,
        help="Sets the resolution for the marching cubes grid.",
    )

    parser.add_argument("--optimize_extrinsics", action="store_true")
    parser.add_argument("--optimize_focal_length", action="store_true")
    parser.add_argument("--optimize_distortion", action="store_true")
    parser.add_argument("--save_segmented_images", action="store_true")
    parser.add_argument("--save_raw_density", action="store_true")
    parser.add_argument("--align_bounding_box", action="store_true")
    parser.add_argument(
        "--face_to_cam_path", type=str, default="./metadata/faceToCam.json"
    )
    parser.add_argument("--downscale_factor", type=float, default=1.0)
    parser.add_argument("--face_to_cam", action="store_true")
    parser.add_argument("--cam_faces_path", type=str, default="./data/faces_v2.json")
    parser.add_argument("--save_dir_name", type=str, default="camera_check")

    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    params_path = os.path.join(args.out_dir, "params.txt")

    with open(args.cam_faces_path, "r") as f:
        faces = json.load(f)

    if "stage2" in args.out_dir:
        params_orig = param_utils.read_params(params_path, distortion=False)
    else:
        params_orig = param_utils.read_params(params_path, distortion=True)
    params = deepcopy(params_orig)
    cam2idx = {}
    pos = []
    rot = []
    intrs = []
    dists = []
    c2ws = []
    for idx, param in enumerate(params):
        w2c = param_utils.get_extr(param)
        intr, dist = param_utils.get_intr(param)
        w2c = np.vstack((w2c, np.asarray([[0, 0, 0, 1]])))
        c2w = np.linalg.inv(w2c)
        cam2idx[param["cam_name"]] = idx
        intrs.append(intr)
        dists.append(dist)
        pos.append(c2w[:3, 3])
        rot.append(c2w[:3, :3])
        c2ws.append(c2w)

    if args.align_bounding_box:
        pos = np.stack(pos)
        rot = np.stack(rot)
        center = pos.mean(axis=0)
        max_dist = cdist(pos, pos).max()

        # Move center of scene to [0, 0, 0]
        pos -= center

        axs = np.zeros((3, 3))

        # Rotate to align bounding box
        find_axes = True
        for idx, dir_ in enumerate(
            [
                ["1 0 0", "-1 0 0"],
                ["0 1 0", "0 -1 0"],
                ["0 0 1", "0 0 -1"],
            ]
        ):
            avg1 = []
            for camera in faces[dir_[0]]["cameras"]:
                try:
                    avg1.append(pos[cam2idx[camera]])
                except:
                    pass

            avg2 = []
            for camera in faces[dir_[1]]["cameras"]:
                try:
                    avg2.append(pos[cam2idx[camera]])
                except:
                    pass
            
            if not avg1 or not avg2:
                find_axes = False
                break

            axs[idx] = np.asarray(avg1).mean(axis=0) - np.asarray(avg2).mean(axis=0)
            axs[idx] /= np.linalg.norm(axs[idx])

        # Get closest orthormal basis
        if find_axes:
            u, _, v = np.linalg.svd(axs)
            orth_axs = u @ v
            new_pos = (orth_axs @ pos.T).T
            new_rot = orth_axs @ rot
        else:
            new_pos = pos
            new_rot = rot

        # Scale to fit diagonal in unity cube
        scale_factor = np.sqrt(2) / max_dist * args.camera_scale
        new_pos *= scale_factor

        # Move center of scene to [0.5, 0.5, 0.5]
        new_pos += 0.5

        extrs = np.zeros((new_pos.shape[0], 4, 4))
        extrs[:, :3, :3] = new_rot
        extrs[:, :3, 3] = new_pos
        extrs[:, 3, 3] = 1
    else:
        extrs = np.array(c2ws)

    save_extrinsics(params, extrs, params_path)
    
            
if __name__ == "__main__":
    main()
