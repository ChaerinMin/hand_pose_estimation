# BRICS-MINI Calibration
This codebase includes camera calibration codes and test data (testv0) for brics-mini-odriod. 

## Environments.

On CCV, we need to load `cuda/11.8` as the base environemnt. The base environment should following alphamano_env.yml and alphapose_env.txt.

TODO:
- Clean up for alphapose_env.txt.


Then, install the following git repos:

1. `colmap`(offical version). https://github.com/colmap/colmap.git
2. `langSAM`(not offical version). https://github.com/luca-medeiros/lang-segment-anything
3. `instant-ngp`(offical version). https://github.com/coreqode/instant-ngp

You might need to manually change the setup.py file for your own platform. For me, I changed the setup file of `lang-segment-anything` making it compatible with Python3.7.

For instant-ngp, built the version without GUI.

Run the full pipeline with:

```
process_bash.sh
```
If you want to test the pipeline step-by-step, check the following instructions. 

## COLMAP Calibration

### Input data structure
We need two types of data for calibration; 1. checkerboard snapshots for `colmap`; 2. scene snapshots for `instant-ngp` reconstruction. Data Structure befre processing:

```
root_directory
    |_yyyy-mm-dd_session_snapshot (all you need for calibration)
        |_ brics-odroid-001_cam0 
            |_ snap_checkerboard
            |_ snap_scene
            |_ ...
        |_ brics-odroid-002_cam0 
        |_ ... 
...
```

### Calibration command
```bash
python scripts/colmap_calib.py -r $ROOT_DIR -o $OUT_DIR -s $SESSION --ith $IDX_SNAP_CHECKER
# -r ROOT directory that stores brics-mini non-pii data
# -s SESSION to proceed, i.e., yyyy-mm-dd_session_snapshot
# -o OUTPUT directory to stores calibration result. As brics-mini non-pii root folder is read-only.
# --ith the IDX of the checker data in the folder. We assume the ith snapshots are time-consistent. 
```

TODO:
- Add timestamp check for the data reader in case there any view point missing.

Check the calibration accuracy from the bash output, such as:
```bash
Bundle adjustment report
------------------------
 Initial cost : 0.596945 [px]
   Final cost : 0.57893 [px]
  Termination : Convergence
```

### Output data structure

```
output_directory
    |_params.txt
```


## Instant-NGP Calibration Refinement

### Calibration command
First, segment foreground objects with command:
```bash
python scripts/sam_segment.py -r $ROOT_DIR -o $OUT_DIR -s $SESSION --ith $IDX_SNAP_SCENE --text "${TEXT_PROMPT}" --use_snapshot --overwrite
# -r ROOT directory that stores brics-mini non-pii data
# -s SESSION to proceed, i.e., yyyy-mm-dd_session_snapshot
# -o OUTPUT directory to stores calibration result. As brics-mini non-pii root folder is read-only.
# --ith the IDX of the scene data in the folder. We assume the ith snapshots are time-consistent. 
# --text the text prompt to seperate foreground object, such as "teapot". 
```

Then, reconstruct foreground objects with command:
```bash
python scripts/object_reconstruct.py \
    -o $OUT_DIR \
    --ith $IDX_SNAP_SCENE \
    --batch_size 32512 \
    --n_steps 15000 \
    --align_bounding_box \
    --downscale_factor 0.45 \
    --optimize_extrinsics \
    --save_segmented_images \
    --overwrite_segmentation \
```

Check the calibration result from the loss or the reconstructed mesh. Note the instant-ngp camera coordinate frames is different from the colmap camera coordinate frames. 


### Output data structure

```
output_directory
    |_params.txt
    |_optim_params.txt
    |_images
        |- segmented_sam (sam segmentation results)
            |_ brics-odroid-001_cam0 
                |_ %08d.png
            |_ brics-odroid-001_cam0 
                |_ %08d.png
        |- camera_check (foreground object from refined camera)
            |_ brics-odroid-001_cam0 
                |_ %08d.png
            |_ brics-odroid-001_cam0 
                |_ %08d.png
```