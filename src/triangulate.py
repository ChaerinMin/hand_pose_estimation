import numpy as np
from skimage.measure import ransac

class Triangulate:
    def __init__(self):
        self.keypoint3d = np.zeros(3)
        
    def estimate(self, *data):
        keypoints2d, proj_mats = data
        
        # Build linear equation
        A = np.zeros((keypoints2d.shape[0]*2, 4))
        for i in range(keypoints2d.shape[0]):
            u, v = keypoints2d[i]
            A[i] = u*proj_mats[i][2]-proj_mats[i][0]
            A[i+1] = v*proj_mats[i][2]-proj_mats[i][1]

        # Solve the system
        U, S, V = np.linalg.svd(A)
        x = V[-1]
        x /= x[-1]
        self.keypoint3d = x[:-1]
        
        return True
    
    def residuals(self, *data):
        keypoints2d, proj_mats = data
        projected = np.matmul(proj_mats, np.hstack((self.keypoint3d, 1)))
        projected = (projected / projected[:,-1:])[:,:-1]

        error = np.linalg.norm(np.abs(projected - keypoints2d), axis=-1) # V
        return error

def triangulate_joints(keypoints, proj_mats, processor, conf_thresh_start=0.75, **kwargs):
    """
    Triangulate joint locations using DL.
    
    Inputs -
        keypoints2d: Dictionary of predicted 2D keypoints with confidence score
        proj_mats: Projection matrices of all cameras
        processor: A function which should processs the selected camera views to return a `Triangulate` model
        conf_thresh_start: The confidence threshold to start with to select views
        min_joints: Minimum joints required to perform triangulation
        kwargs: Any extra parameters to be passed to processor other than keypoints and projected matrix
    Outputs -
        keypoints3d (N): 3D keypoint locations
        residuals (N): Average reprojection error for each joint accross selected views
    """
    keypoints3d = []
    residuals = []
    min_cams = kwargs['min_samples'] + 1
    # min_cams = 10
    num_joints = keypoints.shape[1]
    v = (keypoints[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v >= min_cams)[0]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    for joint in range(num_joints):
        conf_thresh = conf_thresh_start
        computed = False
        while conf_thresh > 0 and not computed:
            # Select best points
            chosen = keypoints[:,joint,2] > conf_thresh
            selected_keypoints = keypoints[chosen,joint,:2]
            selected_projs = proj_mats[chosen]
            if selected_keypoints.shape[0] < min_cams:
                conf_thresh -= 0.1
                # print(f"Changing confidence threshold for keypoint number {joint}")
                continue

            # Triangulate
            computed = True
            # print(f'conf {conf_thresh} of joint {joint}, points shape {selected_keypoints.shape}')
            model = processor(selected_keypoints, selected_projs, **kwargs)
            if model is not None:
                keypoints3d.append(np.append(model.keypoint3d,conf3d[joint]))
                residuals.append(model.residuals(selected_keypoints, selected_projs).mean())
        if not computed or not model:
            #else:
            keypoints3d.append(np.array([0.,0.,0., 0.]))
            residuals.append(np.array(9999999.))
            print(f'missing one point at joint {joint}')
            
    keypoints3d = np.asarray(keypoints3d)
    residuals = np.asarray(residuals)
    
    # print(f'ransac output {keypoints3d.shape} keypoints')   
    return keypoints3d, residuals


def simple_processor(keypoints, proj_mats, **kwargs):
    """
    Simple processor which considers all keypoints for triangulating.
    """
    model = Triangulate()
    model.estimate(keypoints, proj_mats)
    return model


def ransac_processor(keypoints, proj_mats, **kwargs):
    """
    RANSAC processor for triangulating joints.
    """
    model, _ = ransac((keypoints, proj_mats), Triangulate, **kwargs)
    return model
