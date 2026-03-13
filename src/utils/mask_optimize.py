"""
Mask-based shape refinement for MANO models
"""
import cv2
import numpy as np
import torch
from torch.optim import LBFGS

from easymocap.pyfitting.optimize import grad_require, FittingMonitor
from src.utils.mask_utils import render_hand_silhouette_differentiable


def refine_shape_with_mask(
    body_model, body_params, hand_masks, cameras, cam_names,
    cam_mapper, intrs, scale, kp3d, hand_side="right",
    weight_loss=None, max_iter=20, verbose=False,
    cam_frame_indices=None, kp3d_all=None,
    render_scale=0.25,
    overflow_weight=1.0, underflow_weight=1.0
):
    """
    Refine MANO/SMPL-X shape parameters using mask silhouettes.

    Args:
        body_model: MANO/SMPL-X model (SMPLlayer)
        body_params: dict with keys ['shapes', 'poses', 'Rh', 'Th']
                     shapes: (1, nBetas), poses: (nFrames, nPose), Rh: (nFrames, 3), Th: (nFrames, 3)
        hand_masks: dict mapping camera name to ground truth mask (H, W) with values 1 or 2 (or 1 for body)
        cameras: dict with 'R', 'T', 'names' (easymocap format)
        cam_names: list of camera names
        cam_mapper: dict mapping camera names
        intrs: list of camera intrinsic matrices (3, 3)
        scale: global metric scale factor (1/s from estimate_scale_from_keypoints)
        kp3d: (n_joints, 4) keypoints for root joint extraction (frame 0 fallback)
        hand_side: "left" (mask value 1), "right" (mask value 2), "body" (mask value 1)
        weight_loss: dict with keys ['mask', 'reg_shapes', 'init_shape']
        max_iter: maximum LBFGS iterations
        verbose: whether to print loss values
        cam_frame_indices: optional dict mapping camname -> index into params arrays.
                           If provided, each camera uses its own frame's pose for rendering
                           (important when masks come from different timestamps per camera).
                           If None, frame 0 is used for all cameras.
        kp3d_all: optional (nFrames, n_joints, 4) keypoints for all frames.
                  If provided together with cam_frame_indices, each camera uses its own
                  frame's root joint for metric scaling.

    Returns:
        body_params: refined parameters with updated shapes
    """
    device = body_model.device

    # Default weights
    if weight_loss is None:
        weight_loss = {
            'mask': 1e3,           # Silhouette matching loss
            'reg_shapes': 1e2,     # Shape regularization
            'init_shape': 5e2      # Stay close to initial shape
        }

    # Convert to torch tensors
    body_params = {key: torch.Tensor(val).to(device) for key, val in body_params.items()}
    body_params_init = {key: val.clone() for key, val in body_params.items()}

    # Only optimize shape parameters
    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)

    # Prepare ground truth masks for valid cameras
    # hand_side: "left" -> mask value 1, "right" -> mask value 2, "body" -> mask value 1
    hand_value = 2 if hand_side == "right" else 1
    valid_cam_indices = []
    gt_masks_list = [] 

    for cam_idx, cam in enumerate(cam_names):
        if cam in hand_masks and cam in cam_mapper:
            mask = hand_masks[cam]
            hand_mask = (mask == hand_value).astype(np.uint8) * 255

            # Check if mask has sufficient pixels
            if hand_mask.sum() > 2000 * 255: 
                valid_cam_indices.append(cam_idx)
                gt_masks_list.append(hand_mask)

    if len(valid_cam_indices) == 0:
        print(f"Warning: No valid masks found for {hand_side} hand. Skipping mask refinement.")
        body_params = {key: val.detach().cpu().numpy() for key, val in body_params.items()}
        return body_params

    print(f"  - Refining {hand_side} hand shape with {len(valid_cam_indices)} mask views")

    # Get image dimensions from first valid camera
    img_height, img_width = gt_masks_list[0].shape
    render_h = int(img_height * render_scale)
    render_w = int(img_width * render_scale)

    # Convert GT masks to torch tensors (downsampled for memory efficiency)
    gt_masks_tensors = [
        torch.tensor(
            cv2.resize((m > 0).astype(np.float32), (render_w, render_h), interpolation=cv2.INTER_LINEAR),
            dtype=torch.float32, device=device
        )
        for m in gt_masks_list
    ]

    # Pre-convert faces to tensor (constant, no gradient needed)
    faces_tensor = torch.tensor(body_model.faces.astype(np.int64), dtype=torch.int64, device=device)

    # SMPL-X shapedirs may expect more betas than were optimized (e.g. 20 vs 10).
    # Record how many extra zeros to pad when calling the model.
    n_betas_model = body_model.shapedirs.shape[-1]
    n_betas_opt = body_params['shapes'].shape[-1]
    n_betas_pad = max(0, n_betas_model - n_betas_opt)

    # Setup optimizer
    optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter)

    def closure(debug=False):
        optimizer.zero_grad()

        # Pad shapes to match model's expected number of betas (gradient flows through opt betas)
        if n_betas_pad > 0:
            shapes_full = torch.cat([
                body_params['shapes'],
                torch.zeros(body_params['shapes'].shape[0], n_betas_pad, device=device)
            ], dim=-1)
        else:
            shapes_full = body_params['shapes']

        # Compute mask loss across all valid views (differentiable)
        # Each camera uses its own frame's pose if cam_frame_indices is provided,
        # so vertices are computed per-camera inside the loop.
        mask_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        for i, cam_idx in enumerate(valid_cam_indices):
            # Determine which frame to use for this camera's pose
            cam = cam_names[cam_idx]
            frame_idx = 0
            if cam_frame_indices is not None:
                frame_idx = cam_frame_indices.get(cam, 0)
                frame_idx = min(frame_idx, body_params['poses'].shape[0] - 1)

            params_this_frame = {
                'shapes': shapes_full,
                'poses': body_params['poses'][frame_idx:frame_idx+1],
                'Rh': body_params['Rh'][frame_idx:frame_idx+1],
                'Th': body_params['Th'][frame_idx:frame_idx+1]
            }
            vertices = body_model(return_verts=True, return_tensor=True, **params_this_frame)[0]

            # Metric scaling around the root joint
            if kp3d_all is not None:
                root_np = kp3d_all[frame_idx][0, :3]
            else:
                root_np = kp3d[0, :3]
            root = torch.tensor(root_np, device=device, dtype=torch.float32)
            vertices = (vertices - root) * scale + root
            # Render differentiable silhouette (at reduced resolution)
            extrs = np.concatenate([cameras["R"][cam_idx], cameras["T"][cam_idx][:, None]], axis=-1)
            extrs = np.concatenate([extrs, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
            extrs = torch.tensor(extrs, device=device)
            scaled_intr = intrs[cam_idx].copy()
            scaled_intr[0, 0] *= render_scale  # fx
            scaled_intr[1, 1] *= render_scale  # fy
            scaled_intr[0, 2] *= render_scale  # cx
            scaled_intr[1, 2] *= render_scale  # cy
            pred_silhouette = render_hand_silhouette_differentiable(
                vertices, faces_tensor,
                torch.tensor(scaled_intr, device=device, dtype=torch.float32), extrs,
                render_h, render_w, device=device
            )

            # Compute asymmetric mask loss to handle:
            # 1. GT mask has inconsistent wrist cutoff (may include forearm)
            # 2. GT mask may have occlusions (missing hand parts)
            #
            # Strategy:
            # - Predicted outside GT (overflow): HIGH penalty - this is definitely wrong
            # - GT has pixels but predicted doesn't (underflow): LOW penalty - could be occlusion
            gt_mask_tensor = gt_masks_tensors[i]

            # Overflow: predicted=1, gt=0 -> MANO projects outside observed hand region
            # This is almost always wrong (shape too big or wrong pose)
            overflow = torch.clamp(pred_silhouette - gt_mask_tensor, min=0)

            # Underflow: gt=1, predicted=0 -> observed hand not covered by MANO
            # This could be: (1) occlusion, (2) wrist/forearm in GT, (3) shape too small
            # Use lower weight since occlusion is common
            underflow = torch.clamp(gt_mask_tensor - pred_silhouette, min=0)

            mask_loss_sum = mask_loss_sum + (
                overflow_weight * torch.mean(overflow ** 2) +
                underflow_weight * torch.mean(underflow ** 2)
            )

        # Average across views
        mask_loss = mask_loss_sum / len(valid_cam_indices)

        loss_dict = {
            'mask': mask_loss,
            'reg_shapes': torch.sum(body_params['shapes'] ** 2),
            'init_shape': torch.sum((body_params['shapes'] - body_params_init['shapes']) ** 2)
        }

        if verbose:
            print(' '.join([key + ' %.3f' % (loss_dict[key].item() * weight_loss[key])
                           for key in loss_dict.keys() if weight_loss[key] > 0]))

        loss = sum([loss_dict[key] * weight_loss[key] for key in loss_dict.keys()])

        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict

    # Run optimization
    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()

    grad_require(opt_params, False)

    # Get final loss for reporting
    loss_dict = closure(debug=True)
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].item()

    if verbose:
        print(f"  - Final loss: {final_loss:.4f}")
        print(f"  - Loss breakdown: {loss_dict}")

    # Convert back to numpy
    body_params = {key: val.detach().cpu().numpy() for key, val in body_params.items()}

    return body_params
