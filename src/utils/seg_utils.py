import sys
sys.path.append(".")

import os
import json
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import logging
import cv2
from src.utils.general import create_dir

def sam_pred(model, frames, text_prompt, base_path):
    out_path = os.path.join(base_path, "images", "segmented_sam")
    failed = []
    
    for view in tqdm(frames.keys()):
        os.makedirs(os.path.join(out_path, view), exist_ok=True)
        for frame in frames[view].keys():
            image_pil = Image.fromarray(frames[view][frame]).convert("RGB")
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            
            if masks.shape[0] == 0:
                logging.warn("Segmentation failed")
                failed.append([view, frame])
                # continue
                masks = torch.ones((frames[view][frame].shape[:3])).permute(2,0,1)
            
            masks = np.any(masks.permute(1, 2, 0).numpy(), axis=-1)
            result = np.concatenate([frames[view][frame].astype(np.uint8), (masks[...,None]*255).astype(np.uint8)], axis=-1)
            cv2.imwrite(os.path.join(out_path, view, f"{frame}.png"), result)
            
    with open(os.path.join(base_path, "sam_failed.json"), "w") as f:
        json.dump(failed, f)
