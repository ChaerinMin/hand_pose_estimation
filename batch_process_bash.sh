# source ~/.bashrc
# conda activate /users/rfu7/data/anaconda/pose_env
# cd pose_estimation

ROOT_DIR="/users/rfu7/ssrinath/brics/non-pii/brics-mini"
OUT_DIR="/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-05-09" # "../data/processed"
SESSION="2024-05-09"
IDX_VIDEO="-1"
# echo "########################## EXTRACT 2D KEYPOINTS ################################"
# python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO

echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
python scripts/keypoints_3d_fast.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --undistort --all_frames --easymocap --use_optim_params

echo "################################ MANO FIT ##################################"
python scripts/mano_em.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR \
        --model manor --body handr --undistort --ith $IDX_VIDEO \
        --use_filtered --use_optim_params --vis_smpl # --save_frame # --vis_2d_repro --vis_3d_repro 