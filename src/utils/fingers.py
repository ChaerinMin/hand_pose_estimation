import numpy as np

# Indices of fingers
# If none of the keypoints are present for any finger, skip the frame
FINGER_IDX = [list(range(2, 5))] + [ list(range(i, i+4)) for i in range(5, 18, 4) ]

# Indices of finger tips
# If any of them are missing, skip the frame
TIP_IDX = [4, 8, 12, 16, 20]


def compute_scale_factor(keypoints3d, mano_joints):
    # consider confidence > 0.1
    valid_mask = keypoints3d[:, 3] > 0.1
    if valid_mask.sum() < 3:
        return 1.0
    
    # scale based on root
    root_idx = 0
    kpts_centered = keypoints3d[valid_mask, :3] - keypoints3d[root_idx:root_idx+1, :3]
    mano_centered = mano_joints[valid_mask] - mano_joints[root_idx:root_idx+1]
    kpts_scale = np.linalg.norm(kpts_centered, axis=1).mean()
    mano_scale = np.linalg.norm(mano_centered, axis=1).mean()
    
    # scale factor
    scale_factor = kpts_scale / (mano_scale + 1e-8)
    return scale_factor