"""
Given params.txt from bin_to_params.py, 
    or optim_params.txt from normalize_cameras.py at brics_mini_calibration branch,
convert it back to Colmap's .bin and .txt file.
"""

import argparse
import os
import pandas as pd

import pycolmap


def main(args):
    # paths
    calib_dir = os.path.join(
        args.brics_dir,
        args.day,
        args.multisequence,
        "calib",
        f"stage{args.stage}",
        "sparse",
        "0",
    )
    params_path = os.path.join(calib_dir, f"{args.params_fname}.txt")
    save_dir = os.path.join(calib_dir, f"bin_from_{args.params_fname}")
    os.makedirs(save_dir, exist_ok=True)

    # read params.txt
    full_cols = [
        'cam_id', 'width', 'height', 
        'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 
        'cam_name', 
        'qvecw', 'qvecx', 'qvecy', 'qvecz', 
        'tvecx', 'tvecy', 'tvecz'
    ]
    df = pd.read_csv(params_path, sep='\s+', comment='#', names=full_cols)

    # cameras.txt
    cam_cols = ['cam_id', 'width', 'height', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2']
    cameras_df = df[cam_cols].drop_duplicates(subset=['cam_id'])
    cameras_path = os.path.join(save_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_df)}\n")
        for _, row in cameras_df.iterrows():
            params = [row['fx'], row['fy'], row['cx'], row['cy'], 
                      row['k1'], row['k2'], row['p1'], row['p2']]
            params_str = " ".join(map(str, params))
            line = f"{int(row['cam_id'])} OPENCV {int(row['width'])} {int(row['height'])} {params_str}\n"
            f.write(line)
    print(f"Saved {cameras_path}")

    # images.txt
    images_path = os.path.join(save_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(df)}\n")
        
        for idx, row in df.iterrows():
            qvec = [row['qvecw'], row['qvecx'], row['qvecy'], row['qvecz']]
            tvec = [row['tvecx'], row['tvecy'], row['tvecz']]
            qvec_str = " ".join(map(str, qvec))
            tvec_str = " ".join(map(str, tvec))
            cam_id = int(row['cam_id'])
            name = row['cam_name'] + ".jpg"
            line1 = f"{idx+1} {qvec_str} {tvec_str} {cam_id} {name}\n"
            f.write(line1)
            f.write("\n")  # for points
    print(f"Saved {images_path}")

    # convert to .bin
    points_path = os.path.join(save_dir, "points3D.txt")  # pycolmap requires this file to exist
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point.\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
    try:
        reconstruction = pycolmap.Reconstruction(save_dir)
        reconstruction.write_binary(save_dir) 
        print(f"Converted .txt to .bin in {save_dir}")
    except Exception as e:
        print(f"Failed to convert .txt to .bin using pycolmap: {e}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--brics_dir",
        type=str,
        required=True,
        help="Path to the video file or folder",
    )
    parser.add_argument("-d", "--day", type=str, required=True, help="yyyy-mm-dd")
    parser.add_argument(
        "-m", "--multisequence", type=str, required=True, help="multisequence0000001"
    )
    parser.add_argument(
        "--params_fname", type=str, 
        choices=["params", "optim_params"],
        default="optim_params",
        help="Filename of the parameters to convert"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="1 is before undistort. 2 assumes already undistorted"
    )
    cli_args = parser.parse_args()

    main(cli_args)