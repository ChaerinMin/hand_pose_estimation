source ~/.bashrc
conda activate /users/rfu7/data/anaconda/pose_env
cd pose_estimation

ROOT_DIR="/users/rfu7/ssrinath/brics/non-pii/brics-mini"
OUT_DIR="/users/rfu7/ssrinath/datasets/Action/brics-mini/$1" # "../data/processed"
SESSION=$1
IDX_VIDEO="-1"
IDX_START=$2
IDX_END=$3
ANCHOR_CAMERA="brics-odroid-002_cam0"
# ANCHOR_CAMERA="brics-odroid-003_cam1"

# echo "########################## EXTRACT 2D KEYPOINTS ################################"
# python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END \
#         --use_optim_params --anchor_camera $ANCHOR_CAMERA --use_hamer True

# echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
# python scripts/keypoints_3d_fast.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
#         --undistort --all_frames --easymocap --use_optim_params --to_smooth  --anchor_camera $ANCHOR_CAMERA

echo "################################ MANO FIT ##################################"
python scripts/mano_em.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR \
        --model manor --body handr --undistort  --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
        --use_filtered --use_optim_params --stride 1 --vis_smpl --vis_2d_repro --vis_3d_repro --to_smooth  --anchor_camera $ANCHOR_CAMERA # --save_mesh --save_frame