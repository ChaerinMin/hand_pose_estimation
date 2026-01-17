#!/bin/bash

#SBATCH -J hand_pose_estimation
#SBATCH -p 3090-gcondo --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /oscar/data/ssrinath/users/cmin5/hand_pose_estimation/sbatch_out/sbatch_%a.out
#SBATCH -e /oscar/data/ssrinath/users/cmin5/hand_pose_estimation/sbatch_err/sbatch_%a.err
#SBATCH --array=0-239%30

module load miniforge3
source /oscar/runtime/software/x86_64_v3/miniforge3-25.3.0-3-a6hhdjzejtacz63sugjqnvgosfqz63ul/etc/profile.d/conda.sh
conda activate zeronvs
cd /oscar/data/ssrinath/users/cmin5/hand_pose_estimation
echo Directory is `pwd`
module load ffmpeg

# configs
ROOT_DIR="/oscar/data/ssrinath/brics/non-pii/brics-mini"
SEQ="2024-06-20"
OUT_DIR="/oscar/data/ssrinath/brics/non-pii/brics-mini/2024-06-20/multisequence000001/calib/stage2/sparse/0"

# run 2D mask extraction
python -m scripts.mask_2d -r "$ROOT_DIR" -s "$SEQ" -o "$OUT_DIR" \
    --ith 0 --start 0 --end -1 \
    --anchor_camera "brics-odroid-002_cam0" --use_optim_params \
    --v_idx $SLURM_ARRAY_TASK_ID