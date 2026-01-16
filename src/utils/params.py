import numpy as np
import cv2

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def get_intr(param, undistort=False):
    intr = np.eye(3)
    intr[0, 0] = param["fx_undist" if undistort else "fx"]
    intr[1, 1] = param["fy_undist" if undistort else "fy"]
    intr[0, 2] = param["cx_undist" if undistort else "cx"]
    intr[1, 2] = param["cy_undist" if undistort else "cy"]

    # TODO: Make work for arbitrary dist params in opencv
    if "k1" in param:
        dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])
    else:
        dist = np.zeros((4,))

    return intr, dist


def get_rot_trans(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    r = qvec2rotmat(-qvec)
    return r, tvec


def get_extr(param):
    r, tvec = get_rot_trans(param)
    extr = np.vstack([np.hstack([r, tvec[:, None]]), np.zeros((1, 4))])
    extr[3, 3] = 1
    extr = extr[:3]

    return extr


def read_params(params_path, distortion):
    if distortion:
        params = np.loadtxt(
            params_path,
            dtype=[
                ("cam_id", int),
                ("width", int),
                ("height", int),
                ("fx", float),
                ("fy", float),
                ("cx", float),
                ("cy", float),
                ("k1", float),
                ("k2", float),
                ("p1", float),
                ("p2", float),
                ("cam_name", "<U22"),
                ("qvecw", float),
                ("qvecx", float),
                ("qvecy", float),
                ("qvecz", float),
                ("tvecx", float),
                ("tvecy", float),
                ("tvecz", float),
            ]
        )
    else:
        params = np.loadtxt(
            params_path,
            dtype=[
                ("cam_id", int),
                ("width", int),
                ("height", int),
                ("fx", float),
                ("fy", float),
                ("cx", float),
                ("cy", float),
                ("cam_name", "<U22"),
                ("qvecw", float),
                ("qvecx", float),
                ("qvecy", float),
                ("qvecz", float),
                ("tvecx", float),
                ("tvecy", float),
                ("tvecz", float),
            ]
        )  
    params = np.sort(params, order="cam_name")

    return params

def get_undistort_params(intr, dist, img_size):
    new_intr, _ = cv2.getOptimalNewCameraMatrix(intr, dist, img_size, 0, img_size)
    return new_intr

def undistort_image(intr, dist_intr, dist, img):
    result = cv2.undistort(img, intr, dist, None, dist_intr)
    # result = cv2.undistort(img, intr, dist, None)
    return result

def undistort_points(points, intrs, dists, dist_intrs):
    nViews = len(points)
    pelvis_undis = []
    for nv in range(nViews):
        # camera = {key:cameras[key][nv] for key in ['K', 'dist']}
        camera = {
            "K": np.asarray(intrs)[nv],
            "dist": np.asarray(dists)[nv],
        }
        if points[nv].shape[0] > 0:
            keypoints = points[nv]
            K = camera['K']
            dist = camera['dist']
            assert len(keypoints.shape) == 2, keypoints.shape
            kpts = keypoints[:, None, :2]
            kpts = np.ascontiguousarray(kpts)
            if not (dist == 0).all():
                kpts = cv2.undistortPoints(kpts, K, dist, P=dist_intrs[nv])
            pelvis = np.hstack([kpts[:, 0], keypoints[:, 2:]])
        else:
            pelvis = points[nv].copy()
        pelvis_undis.append(pelvis)
    return pelvis_undis

def optimize_extrinsics(cameras, all_kp2d, all_kp3d, inspect_only=False):
    """
    cameras: Dict with 'K', 'R', 'T', 'dist', 'P', 'names'
    all_kp2d: np.ndarray (views, N, 3)
    all_kp3d: np.ndarray (N, 4)

    Returns:
        new_rot: np.ndarray (views, 3, 3)
        new_tr: np.ndarray (views, 3)
    """
    assert cameras['R'].shape[0] == all_kp2d.shape[0]
    all_kp2d = all_kp2d[..., :2].astype(np.float32)
    all_kp3d = all_kp3d[:, :3].astype(np.float32)
    new_rot = []
    new_tr = []
    init_errors = 0
    final_errors = 0
    num_points = 0
    for v in range(all_kp2d.shape[0]):
        cname = cameras['names'][v]
        intrinsic = cameras['K'][v].astype(np.float32)
        dist = cameras['dist'][v].astype(np.float32)
        R_init = cameras['R'][v]
        T_init = cameras['T'][v]
        rvec_init = cv2.Rodrigues(R_init)[0].astype(np.float32)
        tvec_init = T_init.reshape(3, 1).astype(np.float32)
        kp2d = all_kp2d[v]
        kp3d = all_kp3d.copy()
        valid = np.logical_not((kp2d == 0).all(axis=-1))  # (N,)
        if valid.sum() == 0:
            new_rot.append(R_init)
            new_tr.append(T_init)
            print(f"No valid 2D keypoints for {cname}")
            continue
        init_projected, _ = cv2.projectPoints(
            kp3d[valid], rvec_init, tvec_init, intrinsic, dist
        )
        init_error = np.mean(np.linalg.norm(kp2d[valid] - init_projected.squeeze(), axis=-1))
        print(f"Initial Error for {cname}: {init_error:.4f}")
        init_errors += init_error * valid.sum()
        num_points += valid.sum()
        if inspect_only:
            continue
        success, rvec_opt, t_opt, _ = cv2.solvePnPRansac(
            imagePoints=kp2d[valid],
            objectPoints=kp3d[valid],
            cameraMatrix=intrinsic,
            distCoeffs=dist,
            rvec=rvec_init,
            tvec=tvec_init,
        )
        if success:
            opt_projected, _ = cv2.projectPoints(
                kp3d[valid], rvec_opt, t_opt, intrinsic, dist
            )
            opt_error = np.mean(
                np.linalg.norm(kp2d[valid] - opt_projected.squeeze(), axis=-1)
            )
            print(f"Optimized Error for {cname}: {opt_error:.4f}")
            if np.abs(opt_error - init_error) / init_error < 0.1:
                new_rot.append(R_init)
                new_tr.append(T_init)
                final_errors += init_error * valid.sum()
            else:
                R_opt, _ = cv2.Rodrigues(rvec_opt)
                new_rot.append(R_opt)
                new_tr.append(t_opt.flatten())
                final_errors += opt_error * valid.sum()
        else:
            new_rot.append(R_init)
            new_tr.append(T_init)
            print(f"SolvePNPRansac failed for {cname}")
            final_errors += init_error * valid.sum()
    new_rot = np.array(new_rot)
    new_tr = np.array(new_tr)
    print(f"Average Initial Reprojection Error: {init_errors / num_points:.4f}")
    print(f"Average Final Reprojection Error: {final_errors / num_points:.4f}")
    return new_rot, new_tr

def update_extrinsics(pth, params, new_rot, new_tr) -> None:
    """
    new_rot: np.ndarray (views, 3, 3)
    new_tr: np.ndarray (views, 3)
    """
    qvec = []
    for new_r in new_rot:
        quaternion = -rotmat2qvec(new_r)
        qvec.append(quaternion)
    qvec = np.array(qvec)
    new_params = params.copy()
    new_params["qvecw"] = qvec[:, 0]
    new_params["qvecx"] = qvec[:, 1]
    new_params["qvecy"] = qvec[:, 2]
    new_params["qvecz"] = qvec[:, 3]
    new_params["tvecx"] = new_tr[:, 0]
    new_params["tvecy"] = new_tr[:, 1]
    new_params["tvecz"] = new_tr[:, 2]
    np.savetxt(pth, new_params, fmt="%s", header=" ".join(new_params.dtype.fields))
    print(f"Updated extrinsics saved to {pth}")
    return