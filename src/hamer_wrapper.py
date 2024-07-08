from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from hamer.datasets.utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])
HAMER_CKPT_PATH = '/gpfs/data/ssrinath/projects/brics-pose/hamer_ckpts/checkpoints/hamer.ckpt'

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2_list: list,
                 boxes_list: np.array,
                 right_list: np.array,
                 rescale_factor=2.5,
                 train = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2_list = img_cv2_list

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        self.boxes_list = boxes_list.astype(np.float32)
        self.center_list = (boxes_list[:, 2:4] + boxes_list[:, 0:2]) / 2.0
        self.scale_list = rescale_factor * (boxes_list[:, 2:4] - boxes_list[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes_list), dtype=np.int32)
        self.right_list = right_list.astype(np.float32)

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        
        boxes = self.boxes_list[idx].copy()
        
        center = self.center_list[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale_list[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        right = self.right_list[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2_list[idx].copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            # print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2_list[idx].shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        item = {
            'img': img_patch,
            'boxes': boxes,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = center
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = right
        return item
