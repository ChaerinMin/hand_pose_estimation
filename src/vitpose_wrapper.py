from __future__ import annotations

import os
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import numpy as np
import torch
import torch.nn as nn
import logging
from mmpose.datasets.pipelines import Compose
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.apis import init_pose_model, process_mmdet_results, inference_top_down_pose_model, vis_pose_result
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook
from mmpose.core.post_processing import flip_back
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
import time
            
os.environ["PYOPENGL_PLATFORM"] = "egl"

# project root directory
ROOT_DIR = "./"
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")

def _box2cs(cfg, box, dtype=np.float32):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=dtype)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=dtype)
    scale = scale * 1.25

    return center, scale

def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

    return bbox_xywh

class ViTPoseModel(object):
    MODEL_DICT = {
        'ViTPose+-G (multi-task train, COCO)':{
            'config': f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py',
            'model_ccv': f'/gpfs/data/ssrinath/projects/brics-pose/wholebody.pth',
            'model_local': f'_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth',
    }}

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model_name = 'ViTPose+-G (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)
        
    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        dic = self.MODEL_DICT[name]
        config = mmcv.Config.fromfile(dic['config'])
        config.model.pretrained = None
        config.model['type'] = 'TopDownClean'
        config.model['test_cfg']['flip_test'] = False
        config.test_pipeline[2] = {'type': 'ToTensorAndHalf'}
        if os.path.isfile(dic['model_ccv']):
            ckpt_path = dic['model_ccv']
        elif os.path.isfile(dic['model_local']):
            ckpt_path = dic['model_local']
        else:
            os.makedirs(os.path.dirname(dic['model_local']), exist_ok=True)
            os.system("wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz")
            print('Extracting checkpoints, might take minutes...')
            os.system(f"tar -xzvf hamer_demo_data.tar.gz _DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth")
            os.system(f"rm hamer_demo_data.tar.gz")
            ckpt_path = dic['model_local']
            
        model = build_posenet(config.model)
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        load_checkpoint(model, ckpt_path, map_location='cpu', logger=logger)
        
        model.to(self.device)
        model.eval()
        model.half()
        model.cfg = config
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        # image BGR
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def predict_pose_batch(
            self,
            image_list, # image BGR
            bboxes_list) -> list[dict[str, np.ndarray]]:
        return self.predict_pose_batch_flip(image_list, bboxes_list)
        


    def predict_pose_batch_flip(
            self,
            image_list, # image BGR
            bboxes_list) -> list[dict[str, np.ndarray]]:
                         

        cfg = self.model.cfg
        dataset_info = DatasetInfo(cfg.dataset_info)

        # build the data pipeline
        test_pipeline = Compose(cfg.test_pipeline)

        start_time_data = time.time()
        
        batch_data = []
        sizes = []
        for image, bboxes in zip(image_list, bboxes_list):
            sizes.append(len(bboxes))
            if len(bboxes) == 0:
                continue
        
            for bbox in bboxes:
                bboxes = _xyxy2xywh(bboxes)
                center, scale = _box2cs(cfg, bbox, dtype=np.float16)

                # prepare data
                data = {
                    'center': center,
                    'scale': scale,
                    'bbox_score': bbox[4],
                    'bbox_id': 0,  # need to be assigned if batch_size > 1
                    'dataset': dataset_info.dataset_name,
                    'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                    'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                    'rotation': 0,
                    'ann_info': {
                        'image_size': np.array(cfg.data_cfg['image_size']),
                        'num_joints': cfg.data_cfg['num_joints'],
                        'flip_pairs': dataset_info.flip_pairs
                    }
                }
                data['img'] = image
                data = test_pipeline(data)
                batch_data.append(data)
                
                    
        if len(batch_data) == 0:
            return [[]] * len(image_list)
        else:
            batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
            batch_data = scatter(batch_data, [self.device])[0]
            # print(batch_data)
            start_time_inference = time.time()
            # forward the model
            with torch.no_grad():
                result = self.model(
                    img=batch_data['img'],
                    img_metas=batch_data['img_metas'],
                    return_loss=False,
                    return_heatmap=False)
            
            batch_size, _, img_height, img_width = batch_data['img'].shape
            result_orig, result_flip = result[:batch_size], result[batch_size:]
            output_heatmap = result_orig.detach().cpu().numpy()
            output_flipped_heatmap = flip_back(result_flip.detach().cpu().numpy(), dataset_info.flip_pairs, target_type=self.model.keypoint_head.target_type)
            output_flipped_heatmap[:, :, :, 1:] = output_flipped_heatmap[:, :, :, :-1]
            output_heatmap = (output_heatmap + output_flipped_heatmap) * 0.5
            
            bbox_ids = []

            c, s, score = np.zeros((batch_size, 2), dtype=np.float32), np.zeros((batch_size, 2), dtype=np.float32), np.ones(batch_size)
            for i in range(batch_size):
                c[i, :] = batch_data['img_metas'][i]['center']
                s[i, :] = batch_data['img_metas'][i]['scale']
                score[i] = np.array(batch_data['img_metas'][i]['bbox_score']).reshape(-1)


            preds, maxvals = keypoints_from_heatmaps(output_heatmap, c, s,
                unbiased=False,
                post_process='default',
                kernel=11,
                valid_radius_factor=0.0546875,
                use_udp=False,
                target_type='GaussianHeatmap'
            )

            keypoint_result = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
            keypoint_result[:, :, 0:2] = preds[:, :, 0:2]
            keypoint_result[:, :, 2:3] = maxvals

            split_indices = np.cumsum(sizes)[:-1]
            splited_results = np.split(keypoint_result, split_indices)
            return splited_results