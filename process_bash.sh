ROOT_DIR="testv0"
OUT_DIR="testv0" # "../data/processed"
INGP_PATH="./instant-ngp/"
SESSION="2024-04-16_session_snapshot"
IDX_SNAP_CHECKER="1"
IDX_SNAP_SCENE="2"
TEXT_PROMPT="teapot"
PARAMS_PATH="${OUT_DIR}/params.txt" 
export TOKENIZERS_PARALLELISM=true

echo "############################## CALIBRATE with CHECKERBOARD ###########################"
python scripts/colmap_calib.py -r $ROOT_DIR -o $OUT_DIR -s $SESSION --ith $IDX_SNAP_CHECKER

echo "############################### SEGMENT SCENE FOREGROUND ############################"
python scripts/sam_segment.py -r $ROOT_DIR -o $OUT_DIR -s $SESSION --ith $IDX_SNAP_SCENE --text "${TEXT_PROMPT}" --use_snapshot --overwrite

echo "###############################  RECONSTRUCT FOREGROUND  ############################"
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
