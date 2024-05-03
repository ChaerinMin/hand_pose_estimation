ROOT_DIR="testv0"
OUT_DIR="testv0" # "../data/processed"
SESSION="2024-04-16"
IDX_VIDEO="0"

echo "########################## EXTRACT 2D KEYPOINTS ################################"
python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO

# echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
# python scripts/keypoints_3d_fast.py -r $OUT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --undistort --all_frames --easymocap --use_optim_params

# echo "################################ MANO FIT ##################################"
# python scripts/mano_em.py -r $OUT_DIR -s $SESSION -o $OUT_DIR \
#         --model manor --body handr --undistort --ith $IDX_VIDEO \
#         --use_filtered --use_optim_params --vis_smpl --vis_2d_repro --vis_3d_repro 