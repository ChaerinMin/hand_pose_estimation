"""
Mask-based shape refinement for MANO models
"""
import numpy as np
import torch
from torch.optim import LBFGS
from PIL import Image

from easymocap.pyfitting.optimize import grad_require, FittingMonitor
from src.utils.mask_utils import render_hand_silhouette_differentiable


def refine_shape_with_mask(
    body_model, body_params, hand_masks, cameras, cam_names,
    cam_mapper, intrs, scale, kp3d, hand_side="right",
    weight_loss=None, max_iter=20, verbose=False
):
    """
    Refine MANO shape parameters using hand mask silhouettes

    This function takes already optimized MANO parameters (especially pose)
    and refines only the shape parameters to better match the observed hand masks.

    Args:
        body_model: MANO model (SMPLlayer)
        body_params: dict with keys ['shapes', 'poses', 'Rh', 'Th']
                     shapes: (1, 10), poses: (nFrames, 48), Rh: (nFrames, 3), Th: (nFrames, 3)
        hand_masks: dict mapping camera name to ground truth mask (H, W) with values 1 or 2
        cameras: list of camera extrinsic matrices (4, 4)
        cam_names: list of camera names
        cam_mapper: dict mapping camera names
        intrs: list of camera intrinsic matrices (3, 3)
        hand_side: "left" or "right"
        weight_loss: dict with keys ['mask', 'reg_shapes', 'init_shape']
        max_iter: maximum iterations for optimization
        verbose: whether to print loss values

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
    hand_value = 1 if hand_side == "left" else 2
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

    # Convert GT masks to torch tensors
    gt_masks_tensors = [
        torch.tensor((m > 0).astype(np.float32), dtype=torch.float32, device=device)
        for m in gt_masks_list
    ]

    # Pre-convert faces to tensor (constant, no gradient needed)
    faces_tensor = torch.tensor(body_model.faces.astype(np.int64), dtype=torch.int64, device=device)

    # Setup optimizer
    optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe', max_iter=max_iter)

    def closure(debug=False):
        optimizer.zero_grad()

        # Get vertices for first frame only (keep as tensor for gradient flow)
        params_frame0 = {
            'shapes': body_params['shapes'],
            'poses': body_params['poses'][:1],
            'Rh': body_params['Rh'][:1],
            'Th': body_params['Th'][:1]
        }

        # Get vertices as tensor (differentiable)
        vertices = body_model(return_verts=True, return_tensor=True, **params_frame0)[0]

        # scaling
        root = torch.tensor(kp3d[0, :3], device=device, dtype=torch.float32)
        vertices = (vertices - root) * scale + root

        # Compute mask loss across all valid views (differentiable)
        mask_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        for i, cam_idx in enumerate(valid_cam_indices):
            # Render differentiable silhouette
            extrs = np.concatenate([cameras["R"][cam_idx], cameras["T"][cam_idx][:, None]], axis=-1)
            extrs = np.concatenate([extrs, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
            extrs = torch.tensor(extrs, device=device)
            pred_silhouette = render_hand_silhouette_differentiable(
                vertices, faces_tensor,
                torch.tensor(intrs[cam_idx], device=device, dtype=torch.float32), extrs,
                img_height, img_width, device=device
            )

            # Compute asymmetric mask loss to handle:
            # 1. GT mask has inconsistent wrist cutoff (may include forearm)
            # 2. GT mask may have occlusions (missing hand parts)
            #
            # Strategy:
            # - Predicted outside GT (overflow): HIGH penalty - this is definitely wrong
            # - GT has pixels but predicted doesn't (underflow): LOW penalty - could be occlusion
            gt_mask_tensor = gt_masks_tensors[i]

            a = np.zeros((720, 1280, 3), dtype=np.uint8)
            a[pred_silhouette.detach().cpu().numpy()>0.5, :] = np.array([255,0,0], dtype=np.uint8)
            a[gt_mask_tensor.detach().cpu().numpy()>0.5, :] = np.array([0,0,255], dtype=np.uint8)
            Image.fromarray(a).save(f"debug_gt_pred_mask_r_{cameras['names'][cam_idx]}.png")

            # Overflow: predicted=1, gt=0 -> MANO projects outside observed hand region
            # This is almost always wrong (shape too big or wrong pose)
            overflow = torch.clamp(pred_silhouette - gt_mask_tensor, min=0)

            # Underflow: gt=1, predicted=0 -> observed hand not covered by MANO
            # This could be: (1) occlusion, (2) wrist/forearm in GT, (3) shape too small
            # Use lower weight since occlusion is common
            underflow = torch.clamp(gt_mask_tensor - pred_silhouette, min=0)

            # Asymmetric weights: penalize overflow more than underflow
            overflow_weight = 1.0  # Strong penalty for projecting outside GT
            underflow_weight = 1.0  # Weak penalty (could be occlusion or wrist)

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
