"""
Utility functions for mask-based MANO shape refinement
"""
import numpy as np
import pyrender
import trimesh
import cv2
import torch

# PyTorch3D imports for differentiable rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    MeshRenderer,
    BlendParams,
)


def render_hand_silhouette_differentiable(
    vertices, faces, camera_intr, camera_extr, img_height, img_width, device=None
):
    """
    Render hand mesh as a differentiable silhouette mask using PyTorch3D

    Args:
        vertices: (N, 3) torch tensor of mesh vertices in world coordinates
        faces: (F, 3) array or tensor of face indices
        camera_intr: (3, 3) intrinsic matrix
        camera_extr: (4, 4) extrinsic matrix (world to camera)
        img_height: image height
        img_width: image width
        device: torch device

    Returns:
        silhouette: (H, W) differentiable silhouette (values 0-1)
    """
    if device is None:
        device = vertices.device

    # Ensure vertices is a torch tensor
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32, device=device)

    # Ensure faces is a torch tensor
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.int64, device=device)

    # Add batch dimension if needed
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)  # (1, N, 3)
    if faces.dim() == 2:
        faces = faces.unsqueeze(0)  # (1, F, 3)

    # Create mesh
    meshes = Meshes(verts=vertices, faces=faces)

    # Extract camera parameters
    fx, fy = camera_intr[0, 0], camera_intr[1, 1]
    cx, cy = camera_intr[0, 2], camera_intr[1, 2]

    # Convert extrinsic to PyTorch3D convention
    # PyTorch3D uses row-major and different axis conventions
    R = camera_extr[:3, :3]
    T = camera_extr[:3, 3]

    # PyTorch3D expects R and T in a specific format
    # R: (1, 3, 3), T: (1, 3)
    R_pt3d = torch.tensor(R, dtype=torch.float32, device=device)#.unsqueeze(0)
    T_pt3d = torch.tensor(T, dtype=torch.float32, device=device)#.unsqueeze(0)

    # Convert OpenCV camera convention to PyTorch3D
    flip_mat = torch.tensor([[-1., 0., 0.],
                             [ 0., -1., 0.],
                             [ 0., 0., 1.]], device=device, dtype=torch.float32)
    R_pt3d = R_pt3d.clone()
    # R_pt3d[:, :, 1] *= -1  # flip y
    # R_pt3d[:, :, 2] *= -1  # flip z
    R_pt3d = flip_mat @ R_pt3d
    R_pt3d = R_pt3d.t().unsqueeze(0)
    T_pt3d = T_pt3d.clone()
    # T_pt3d[:, 1] *= -1
    # T_pt3d[:, 2] *= -1
    T_pt3d = flip_mat @ T_pt3d
    T_pt3d = T_pt3d.unsqueeze(0)

    # Create perspective camera with intrinsics
    # focal_length in PyTorch3D is in NDC space, need to convert
    focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device)
    principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device)

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R_pt3d,
        T=T_pt3d,
        image_size=torch.tensor([[img_height, img_width]], device=device),
        in_ndc=False,
        device=device,
    )

    # Rasterization settings for soft silhouette
    raster_settings = RasterizationSettings(
        image_size=(img_height, img_width),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-5,  # soft edges for gradient flow
        faces_per_pixel=50,
    )

    # Create silhouette renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4)),
    )

    # Render silhouette
    silhouette = renderer(meshes)  # (1, H, W, 4)

    # Extract alpha channel as silhouette
    silhouette = silhouette[0, :, :, 3]  # (H, W)

    return silhouette


def render_hand_silhouette(vertices, faces, camera_intr, camera_extr, img_height, img_width):
    """
    Render hand mesh as a binary silhouette mask

    Args:
        vertices: (N, 3) array of mesh vertices in world coordinates
        faces: (F, 3) array of face indices
        camera_intr: (3, 3) intrinsic matrix
        camera_extr: (4, 4) extrinsic matrix (world to camera)
        img_height: image height
        img_width: image width

    Returns:
        silhouette: (H, W) binary mask (0 or 255)
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create scene
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0, 0])
    scene.add(mesh)

    # Create camera
    fx, fy = camera_intr[0, 0], camera_intr[1, 1]
    cx, cy = camera_intr[0, 2], camera_intr[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    # Add camera to scene
    scene.add(camera, pose=np.linalg.inv(camera_extr))

    # Render
    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)
    renderer.delete()

    # Create binary silhouette
    silhouette = (depth > 0).astype(np.uint8) * 255

    return silhouette


def compute_mask_iou(mask1, mask2):
    """
    Compute IoU between two binary masks

    Args:
        mask1: (H, W) binary mask
        mask2: (H, W) binary mask

    Returns:
        iou: intersection over union
    """
    mask1_bin = (mask1 > 0).astype(np.uint8)
    mask2_bin = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1_bin, mask2_bin).sum()
    union = np.logical_or(mask1_bin, mask2_bin).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


def compute_wrist_aware_mask_loss(pred_mask, gt_mask, wrist_region_weight=0.3):
    """
    Compute mask loss with reduced weight on wrist region

    The wrist region is less reliable in hand masks, so we give it less weight.
    We identify the wrist region as the bottom part of the mask.

    Args:
        pred_mask: (H, W) predicted silhouette
        gt_mask: (H, W) ground truth mask
        wrist_region_weight: weight for wrist region (default 0.3)

    Returns:
        weighted_loss: scalar loss value
    """
    pred_bin = (pred_mask > 0).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    # Compute pixel-wise difference
    diff = np.abs(pred_bin - gt_bin)

    # Create weight map: lower weight for bottom region (wrist)
    H, W = gt_mask.shape
    weight_map = np.ones((H, W), dtype=np.float32)

    # Find bounding box of ground truth mask
    if gt_bin.sum() > 0:
        y_coords, x_coords = np.where(gt_bin > 0)
        ymin, ymax = y_coords.min(), y_coords.max()

        # Bottom 20% of the hand region gets reduced weight
        wrist_threshold = ymax - 0.2 * (ymax - ymin)
        weight_map[int(wrist_threshold):, :] = wrist_region_weight

    # Compute weighted loss
    weighted_diff = diff * weight_map
    loss = weighted_diff.sum() / (weight_map.sum() + 1e-6)

    return loss


def filter_valid_views(seg_status, hand_masks, cam_names, hand_side):
    """
    Filter cameras that have valid segmentation for the given hand

    Args:
        seg_status: dict mapping "{cam}_{hand_side}" to boolean
        hand_masks: dict mapping cam to mask array
        cam_names: list of camera names
        hand_side: "left" or "right"

    Returns:
        valid_cams: list of camera names with valid masks
        valid_masks: dict mapping camera name to mask for that hand
    """
    valid_cams = []
    valid_masks = {}

    for cam in cam_names:
        status_key = f"{cam}_{hand_side}"
        if status_key in seg_status and seg_status[status_key]:
            if cam in hand_masks:
                mask = hand_masks[cam]
                # Extract mask for the specific hand (1=left, 2=right)
                hand_value = 1 if hand_side == "left" else 2
                hand_mask = (mask == hand_value).astype(np.uint8) * 255

                # Check if mask has sufficient pixels
                if hand_mask.sum() > 100:  # At least 100 foreground pixels
                    valid_cams.append(cam)
                    valid_masks[cam] = hand_mask

    return valid_cams, valid_masks
