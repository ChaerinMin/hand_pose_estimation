import sys

sys.path.append(".")
sys.path.append("./instant-ngp/build")
import os
import numpy as np
import pyngp as ngp
import matplotlib.pyplot as plt
import cv2
import time
import ipdb
import json
from tqdm import tqdm
from copy import deepcopy
from natsort import natsorted
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import glob
import shutil
from src.utils.parser import add_common_args
from src.utils.instantngp_helper import *
import src.utils.params as param_utils

def train(args, params, image_dir, intrs, extrs, dists, params_path, frame, kps):
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.reload_network_from_file(args.network)
    skip_images = [ ]

    ## Verify if the images are present
    for idx, param in enumerate(params):
        img_path = glob.glob(f"{image_dir}/{param['cam_name']}*/{frame}.png")
            
        if len(img_path) == 0:
            skip_images.append(param['cam_name'])
    
    ## Remove camera from params
    params = [param for param in params if param['cam_name'] not in skip_images]
    params = np.asarray(params)
    
    testbed.create_empty_nerf_dataset(
        n_images=len(params), aabb_scale=args.aabb_scale
    )
    print(f"Training on {len(params)} views.")
    id_ = 0
    
    imgs = []
    img_names = []

    for idx, param in enumerate(params):
        img_path = glob.glob(f"{image_dir}/{param['cam_name']}*/{frame}.png")[0]
        img_names.append(img_path.split('/')[-1].split('.')[0])
        img = cv2.cvtColor( cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float32)
        depth_img = np.zeros((img.shape[0], img.shape[1]))
        img /= 255
        imgs.append(img)
        img = srgb_to_linear(img)
        # # premultiply
        img[..., :3] *= img[..., 3:4]

        extr = extrs[idx]
        intr = intrs[idx]
        dist = dists[idx]

        testbed.nerf.training.set_image(id_, img, depth_img)
        testbed.nerf.training.set_camera_extrinsics(id_, extr[:3], convert_to_ngp=False)
        testbed.nerf.training.set_camera_intrinsics(
            id_,
            fx=param["fx"],
            fy=param["fy"],
            cx=param["cx"],
            cy=param["cy"],
            k1=param["k1"],
            k2=param["k2"],
            p1=param["p1"],
            p2=param["p2"],
        )
        id_ += 1
        
    # Taken from i-ngp:scripts/run.py
    # testbed.color_space = ngp.ColorSpace.SRGB
    testbed.nerf.visualize_cameras = True
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    testbed.nerf.training.random_bg_color = True
    testbed.training_batch_size = args.batch_size

    testbed.nerf.training.n_images_for_training = len(params) - len(skip_images)

    testbed.shall_train = True
    # testbed.nerf.training.optimize_extrinsics = args.optimize_extrinsics
    testbed.nerf.training.optimize_focal_length = args.optimize_focal_length
    testbed.nerf.training.optimize_distortion = args.optimize_distortion
    testbed.nerf.cone_angle_constant = 0.000

    n_steps = args.n_steps
    old_training_step = 0
    tqdm_last_update = 0

    start = time.time()
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break
                
                if testbed.training_step == n_steps // 2 :
                    testbed.nerf.training.optimize_extrinsics = args.optimize_extrinsics

                # Update progress bar
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    testbed.shall_train = False
    testbed.nerf.cone_angle_constant = 0.0
    end = time.time()

    if args.gui:
        testbed.init_window(1920, 1080)
        while testbed.frame():
            if testbed.want_repl():
                ipdb.set_trace()

    
    if kps is not None:
        kp_path = os.path.join("/".join(kps[0].split('/')[:-1]), f'{frame}.json')
        with open(kp_path, "r") as f:
            kp3d = np.asarray(json.load(f))[..., :3]
            pc = trimesh.PointCloud(kp3d)
            bounds = trimesh.bounds.corners(pc.bounding_box_oriented.bounds)
            mn = bounds.min(axis=0) - np.array([0.07, 0.07, 0.025])
            mx = bounds.max(axis=0) + np.array([0.07, 0.07, 0.07])
    else:
        mn = None
        mx = None
            
    mesh_dir = os.path.join(args.out_dir, "mesh", "ngp_mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, img_names[0] + '.ply')
    save_mesh(
                args.marching_cubes_res, 
                mesh_path, testbed, 
                args.downscale_factor, 
                pad=args.pad, 
                num_objects = args.num_objects, 
                aabb_mn = mn,
                aabb_mx = mx,
                refine = True
            )
    
    if args.cam_traj_path != "":
        print("Generating test video")
        output_dir = os.path.join(args.out_dir, "gt_contacts")
        os.makedirs(output_dir, exist_ok=True)
        generate_test_video(testbed, args, output_dir , args.cam_traj_path, testbed.render_aabb.min, testbed.render_aabb.max)
        
    # if args.save_raw_density:
    #     save_raw_density(testbed, 256, args.base_path, img_names[0])
    
    if args.save_segmented_images:        
        seg_dir = os.path.join(args.out_dir, "images", args.save_dir_name)
        os.makedirs(seg_dir, exist_ok=True)
        save_mask(testbed, imgs, img_names, params, seg_dir, args.num_objects)
        
    if args.optimize_extrinsics or args.optimize_focal_length:
        save_extrinsics(testbed, params, params_path)

def save_extrinsics(testbed, params, params_path):
    optim_params = deepcopy(params)
    fields = params.dtype.fields
    for idx, meta in enumerate(testbed.nerf.training.dataset.metadata):
        cx, cy = meta.principal_point
        width, height = meta.resolution
        cx = cx * width
        cy = cy * height
        fx, fy = meta.focal_length
        
        c2w = testbed.nerf.training.get_camera_extrinsics(idx, convert_to_ngp=False)
        c2w = np.vstack((c2w, np.asarray([[0, 0, 0, 1]])))
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
        optim_params[idx]["cx"] = cx
        optim_params[idx]["cy"] = cy
        optim_params[idx]["fx"] = fx
        optim_params[idx]["fy"] = fy
    
    np.savetxt(
        os.path.join(os.path.dirname(params_path), "optim_params.txt"),
        optim_params,
        fmt="%s",
        header=" ".join(fields),
    )
    
def connected_components(th):
    num_labels, labels_im = cv2.connectedComponents(th)
    s = 0
    ind = 1
    if num_labels > 2:
        for nl in range(1, num_labels):
            temp = np.sum((labels_im == nl).astype(np.uint8))
            if temp > s:
                s = temp
                ind = nl
    mask = (labels_im == ind) * 255
    return mask
    
def process_mask(
    alpha,
    opening_kernel,
    thresh,
    con_comp=False,
    use_alpha=False,
    dilate_mask=False,
    contours=False,
    contour_thresh=100,
):
    mask = alpha.copy()
    mask = (mask > thresh) * 255
    mask = mask.astype(np.uint8)

    # mask = cv2.morphologyEx(
    #     (mask * 255).astype(np.uint8), cv2.MORPH_OPEN, opening_kernel
    # )

    if con_comp:
        mask = connected_components((mask * 255).astype(np.uint8))

    if dilate_mask:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), opening_kernel, iterations=1)

    if contours:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_TC89_L1,
        )

        # Choose contours with are greater than the given threshold
        chosen = []
        for contour in contours:
            if cv2.contourArea(contour) > contour_thresh:
                chosen.append(contour)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, chosen, -1, 255, -1)
        mask = new_mask

    if use_alpha:
        mask = mask * alpha

    return mask
    
def save_mask(testbed, imgs, img_names, params, output_path, num_objects = 1):
    testbed.nerf.render_with_lens_distortion = False
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    width= 1280
    height = 720
    
    for idx in tqdm(range(len(imgs))):
        testbed.set_camera_to_training_view(idx)
        frame = testbed.render(width, height, 1, True)
        alpha = frame[:, :, 3]

        con_comp = True if num_objects == 1 else False 
        mask = process_mask( alpha, opening_kernel, 0.8, con_comp, False, False, False, None)
        mask = mask.astype(np.float32) 
        img = deepcopy(imgs[idx])
        img = img * 255
        img[..., 3] = mask

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        output_dir = os.path.join(output_path, params[idx]["cam_name"])
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(
                output_dir, f"{img_names[idx]}.png"
            ),
            img,
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

    image_dir = os.path.join(args.out_dir, "images", "segmented_sam")
    params_path = os.path.join(args.out_dir, "params.txt")

    with open(args.cam_faces_path, "r") as f:
        faces = json.load(f)

    params_orig = param_utils.read_params(params_path)
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

            axs[idx] = np.asarray(avg1).mean(axis=0) - np.asarray(avg2).mean(axis=0)
            axs[idx] /= np.linalg.norm(axs[idx])

        # Get closest orthormal basis
        u, _, v = np.linalg.svd(axs)
        orth_axs = u @ v

        new_pos = (orth_axs @ pos.T).T
        new_rot = orth_axs @ rot

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

    img_paths = natsorted(glob.glob(f"{image_dir}/{params[0]['cam_name']}*/*.[jp][pn]g"))
    
    usable_frames_path = os.path.join(args.out_dir, "optmize_frame.json")
    usable_img_paths = img_paths
    if os.path.exists(usable_frames_path):
        with open(usable_frames_path, 'r') as f:
            usable_frames = json.load(f)
    else:
        usable_frames = [args.ith]
        
    usable_img_paths = []
    for idx, path in enumerate(img_paths):
        frame = int(path.split('/')[-1].split('.')[0])
        if frame in usable_frames:
            usable_img_paths.append(path)
    
    usable_kps = None
    if ("actions" in args.out_dir) or ("evals" in args.out_dir) :
        pose_dir = os.path.join(args.out_dir, "pose_dumps/keypoints_3d")
        if os.path.exists(pose_dir):
            kps = natsorted(glob.glob(os.path.join(pose_dir, "*.json")))
            
            usable_kps = []
            for idx, path in enumerate(kps):
                frame = int(path.split('/')[-1].split('.')[0])
                if frame in usable_frames:
                    usable_kps.append(path)
                    
            assert len(usable_kps) == len(usable_img_paths)

    for _, path in enumerate(usable_img_paths):
        frame = path.split('/')[-1].split('.')[0]
        
        if not args.overwrite_segmentation: 
            ## Check if segmentation is already done
            seg_dir = os.path.join(args.out_dir, "images", args.save_dir_name)
            
            if os.path.exists(seg_dir):
                seg_path = os.path.join(seg_dir, params[0]["cam_name"], str(frame) + ".png") 
                
                if os.path.exists(seg_path):
                    print("Segmentation exists at ", seg_path, " Skipping...")
                    continue
                    
        train(args, params, image_dir, intrs, extrs, dists, params_path, frame, usable_kps)
    
            
if __name__ == "__main__":
    main()
