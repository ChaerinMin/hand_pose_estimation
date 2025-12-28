"""
Given colmap's .bin files from Plaster,
convert them to params.txt files, required by Rao's hand pose estimation.
"""

import argparse
import os
import pandas as pd

import numpy as np
from numpy.lib import recfunctions as rf
import pycolmap


def main(args):
    # bin -> txt
    calib_dir = os.path.join(
        args.brics_dir,
        args.day,
        args.multisequence,
        "calib",
        "stage1",  # stage 1: before undistort. stage 2: assume already undistorted
        "sparse",
        "0"
    )
    reconstruction = pycolmap.Reconstruction(calib_dir)
    reconstruction.write_text(calib_dir)
    print(f"Converted .bin to .txt in {calib_dir}")

    # colmap txt -> params.txt
    image_params = []
    with open(os.path.join(calib_dir, "images.txt")) as f:
        skip_next = False
        for line in f.readlines():
            if skip_next:
                skip_next = False
                continue
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[8]))
            param.append(data[9].split("/")[0])
            param += [float(datum) for datum in data[1:8]]
            image_params.append(tuple(param))
            skip_next = True

    images = np.array(image_params, dtype=[
        ('cam_id', int), ('cam_name', '<U21'),
        ('qvecw', float), ('qvecx', float), ('qvecy', float), ('qvecz', float),
        ('tvecx', float), ('tvecy', float), ('tvecz', float)
    ])
    cam_params = []
    with open(os.path.join(calib_dir, "cameras.txt")) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            data = line.split()
            param = []
            param.append(int(data[0]))
            param.append(int(data[2]))
            param.append(int(data[3]))
            param += [float(datum) for datum in data[4:]]
            cam_params.append(tuple(param))
    cameras = np.array(cam_params, dtype=[
        ('cam_id', int),
        ('width', int), ('height', int),
        ('fx', float), ('fy', float),
        ('cx', float), ('cy', float),
        ('k1', float), ('k2', float),
        ('p1', float), ('p2', float),
    ])

    # img_cams = rf.join_by('cam_id', cameras, images)
    df_cameras = pd.DataFrame(cameras)
    df_images = pd.DataFrame(images)
    df_merged = pd.merge(df_cameras, df_images, on='cam_id')
    img_cams = df_merged.to_records(index=False)
    out_path = os.path.join(calib_dir, 'params.txt')
    np.savetxt(out_path, img_cams, fmt="%s", header=" ".join(img_cams.dtype.fields))
    print(f"Saved {out_path}")
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
    cli_args = parser.parse_args()

    main(cli_args)
