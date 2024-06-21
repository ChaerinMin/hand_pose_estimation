ROOT_DIR="testv0"
OUT_DIR="testv0" # "../data/processed"
SESSION="2024-06-17"
IDX_VIDEO="0"
ANCHOR_CAMERA="brics-odroid-002_cam0"

echo "########################## EXTRACT 2D KEYPOINTS ################################"
python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --anchor_camera $ANCHOR_CAMERA

# echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
# python scripts/keypoints_3d_fast.py -r $OUT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --undistort --all_frames --easymocap --use_optim_params --anchor_camera $ANCHOR_CAMERA

# echo "################################ MANO FIT ##################################"
# python scripts/mano_em.py -r $OUT_DIR -s $SESSION -o $OUT_DIR \
#         --model manor --body handr --undistort --ith $IDX_VIDEO --anchor_camera $ANCHOR_CAMERA\
#         --use_filtered --use_optim_params --vis_smpl --vis_2d_repro --vis_3d_repro 