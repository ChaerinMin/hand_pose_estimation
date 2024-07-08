ROOT_DIR="/users/rfu7/ssrinath/brics/non-pii/brics-mini"
OUT_DIR="/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-06-18" # "../data/processed"
SESSION="2024-06-18"
IDX_VIDEO="5"
IDX_START="-1"
IDX_END="-1"
ANCHOR_CAMERA="brics-odroid-002_cam0"

echo "########################## EXTRACT 2D KEYPOINTS ################################"
python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END \
        --use_optim_params --anchor_camera $ANCHOR_CAMERA --use_hamer True

# echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
# python scripts/keypoints_3d_fast.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
#         --undistort --all_frames --easymocap --use_optim_params --to_smooth  --anchor_camera $ANCHOR_CAMERA

# echo "################################ MANO FIT ##################################"
# python scripts/mano_em.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR \
#         --model manor --body handr --undistort  --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
#         --use_filtered --use_optim_params --stride 1 --vis_smpl --to_smooth --vis_2d_repro --vis_3d_repro --anchor_camera $ANCHOR_CAMERA # --save_mesh --vis_2d_repro --vis_3d_repro --save_frame