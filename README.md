# BRICS-MINI Hand Pose Processing
This repo includes 2D hand keypoints, 3D hand keypoints, and Manos parameter extraction for brics-mini-odroid. Test data (testv0) is also included for environment setup.

This repo does not include camera calibration process. For calibration, please refers to [brics_mini_calibration](https://github.com/brown-ivl/hand_pose_estimation/tree/brics_mini_calibration) branch.

## Environments.

On CCV, we need to load `cuda/11.8` as the base environemnt. Install the following modules:


```bash
# Can adapt cuda that fits your own system.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .[all]
pip install -v -e third-party/ViTPose
git clone https://github.com/zju3dv/EasyMocap.git third-party/EasyMocap
pip install -e third-party/EasyMocap
pip install -e third-party/hamer

# (apex is recommended for faster inference with 10% performance boost. Installation can take ~1hr.)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
...
```
Note: We use apex for its compatibility with mmcv(1.5.0). However, apex.amp is deprecated and should be replaced with torch.amp in future development.

### Note
There's a bug in the original EasyMocap `EasyMocap/easymocap/pyfitting/optimize_simple.py`. Change the interp_func to this:
```python
    def interp_func(params):
        for start, end in ranges:
            # 对每个需要插值的区间: 这里直接使用最近帧进行插值了
            left = start - 1
            right = end if end == total_frame_num - 1  else end + 1
            for nf in range(start, end+1):
                weight = (nf - left)/(right - left)
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
        return params
...
```

## Input data structure
Before 2D/3D/Mano feature extraction, you should have the multi-view video data and camera parameters prepared. 

```
root_directory
    |_yyyy-mm-dd (all you need for calibration)
        |_ brics-odroid-001_cam0 
            |_ brics-odroid-001_cam0_timestamp.mp4
            |_ ...
        |_ brics-odroid-002_cam0 
        |_ ... 
output_directory
    |_ optim_params.txt
...
```

Also, put MANO files `MANO_LEFT.pkl` and `MANO_LEFT.pkl` under `data/smplx/smplh`. These files are pre-downloaded as this is a private repo. If the repo turns public, please delete these files and download them from [http://mano.is.tue.mpg.de](http://mano.is.tue.mpg.de).


All checkpoints are pre-downloaded to CCV shared folder, or will be automatically downloaded. 


## Pipeline: 2D->3D->Manos
For end-to-end pipeline, please run:
```bash
bash process_bash.sh
```

For step-by-step breakdown, please follow instruction below. 

### (step 1) 2D Keypoint Extraction
For CCV users, checkpoints are stored under the shared folder `/gpfs/data/ssrinath/projects/brics-pose` and will be loaded automatically. For non-CCV users, checkpoints will be downloaded automatically to your local machine.
```bash
python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO
# -r ROOT directory that stores brics-mini non-pii data
# -s SESSION to proceed, i.e., yyyy-mm-dd
# -o OUTPUT directory to stores the output data. As brics-mini non-pii root folder is read-only.
# --ith the IDX of the video to be proceed. We assume the ith snapshots are time-consistent. 
# --anchor_camera For some reason, there might be random camera missing during filming. The anchor camera helps timestamp check. For time_diff|other_camera - anchor_camera| > 1s will not be considered during triangulation. 
```
Notes: Inference time for the test data should be ~4 sec/video. If your inference time is unreasonable, you might get a slow GPU(re-apply a new one)/older torch/older cuda. 

TODO:
- Large-scale dataset processing.

### (step 2) 3D Triangulation
Run:
```bash
python scripts/keypoints_3d_fast.py -r $OUT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --undistort --all_frames --easymocap --use_optim_params
```

### (step 3) Manos Parameter Fitting
Run:
```bash
python scripts/mano_em.py -r $OUT_DIR -s $SESSION -o $OUT_DIR \
        --model manor --body handr --undistort --ith $IDX_VIDEO \
        --use_filtered --use_optim_params \
        # --vis_smpl --vis_2d_repro --vis_3d_repro 
```
Note: Visualization is slow.

TODO:
- Finetune Manos constraint parameter settings after you have data collected.


## Output data structure
You will get the following structure after processing.

```
root_directory
    |_yyyy-mm-dd (all you need for calibration)
        |_ brics-odroid-001_cam0 
            |_ brics-odroid-001_cam0_timestamp.mp4
            |_ ...
        |_ brics-odroid-002_cam0 
        |_ ... 
output_directory
    |_ optim_params.txt
    |_ bboxes (step 1)
        |_left
            |_000
                |_brics-odroid-001_cam0_timestamp.jsonl
                |_...
            |_...
        |_right
            |_000
            |_...
    |_ keypoints_2d (step 1)
        |_left
            |_000
            |_...
        |_right
            |_000
            |_...
    |_ keypoints_3d (step 2)
        |_000
            |_chosen_frames_left.jsonl
            |_chosen_frames_right.jsonl
            |_left.jsonl
            |right.jsonl
        |_...
    |_ mano (step 3, if visualization)
    |_ repro_2d (step 3, if visualization)
    |_ repro_3d (step 3, if visualization)
...
```
