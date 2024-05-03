import sys
sys.path.append(".")
sys.path.append("./lang-segment-anything")
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from lang_sam import LangSAM
from src.utils.parser import add_common_args
from src.utils.seg_utils import sam_pred
from argparse import ArgumentParser
import ipdb

def main():
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--text", required=True, type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_snapshot", action="store_true")
    args =  parser.parse_args()
    
    
    base_path = args.root_dir
    usable_frames = [args.ith]
    
    model = LangSAM()
    
    frames = {}
    if args.use_snapshot:
        all_views = os.listdir(os.path.join(base_path, args.seq_path))
    else:
        all_views = os.listdir(os.path.join(args.out_dir, "images", "image"))
    
    seg_dir = os.path.join(args.out_dir, "images", "segmented_sam")
            
    for view in all_views:
        frames[view] = {}
        if args.use_snapshot:
            view_paths = sorted(os.listdir(os.path.join(base_path, args.seq_path, view)))
        else:
            view_paths = os.listdir(os.path.join(args.seq_path, "images", "image", view))
        
        for f_idx, frame in enumerate(view_paths):
            if args.use_snapshot:
                frame_no = f_idx
            else:
                frame_no = int(frame.split('.')[0])
            
            if len(usable_frames) != 0:
                if frame_no not in usable_frames:
                    continue
            
            if not args.overwrite:
                if os.path.exists(seg_dir):
                    if args.use_snapshot:
                        seg_path = os.path.join(seg_dir, view, f"{frame_no:08}.png") 
                    else:
                        seg_path = os.path.join(seg_dir, view, str(frame.split('.')[0]) + ".png") 
                    if os.path.exists(seg_path):
                        print("Segmentation exists at ", seg_path, " Skipping...")
                        continue
                    
            if args.use_snapshot:
                image = cv2.imread(os.path.join(base_path, args.seq_path, view, frame))
                frame_name = f"{frame_no:08}"
            else:
                image = cv2.imread(os.path.join(base_path, "images", "image", view, frame))
                frame_name = frame.split('.')[0]
            frames[view][frame_name] = image
    
    sam_pred(model, frames, args.text, args.out_dir)

if __name__ == "__main__":
    main()
