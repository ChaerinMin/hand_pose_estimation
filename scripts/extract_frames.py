import sys
sys.path.append(".")

import os
import numpy as np
import cv2
import json
from PIL import Image
from tqdm import tqdm, trange
from src.utils.reader_v2 import Reader
from src.utils.parser import add_common_args
from src.utils.general import create_dir
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--out_path", required=True, type=str)
    args =  parser.parse_args()
    
    base_path = args.root_dir
    
    reader = Reader("video", base_path, ith=args.ith)

    chosen_frames_path = os.path.join(args.root_dir, "optmize_frame.json")
    if os.path.exists(chosen_frames_path):
        with open(os.path.join(args.root_dir, "optmize_frame.json")) as f:
            chosen_frames = sorted(json.load(f))
    else:
        chosen_frames = [reader.frame_count - 5 ]

    args =  parser.parse_args()
    
    out_path = os.path.join(args.out_path, "images", 'image')
    
    for frames, frame_num in tqdm(reader(chosen_frames)):
        if frame_num not in chosen_frames:
            continue

        for cam_name, frame in frames.items():
            create_dir(os.path.join(out_path, cam_name))
            cv2.imwrite(os.path.join(out_path, cam_name, f"{frame_num:08}.png"), frame)
    

if __name__ == '__main__':
    main()
