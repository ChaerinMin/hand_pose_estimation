import os

input_folder = '/users/rfu7/ssrinath/datasets/Action/brics-mini'
output_folder = '/users/rfu7/data/code/24Text2Action/ABATCH'
start_folder = '2024-10-01-action-clarajin-present'

content = """#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -N 1
#SBATCH -n 4 --mem=64g -p "3090-gcondo" --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH -o /users/rfu7/data/code/24Text2Action/ABATCH/out/{sequence_name}.out

CUDA_VISIBLE_DEVICES=0 cd pose_estimation && bash batch_process_bash.sh {sequence_name} 0 -1"""

started = False
for folder in os.listdir(input_folder):
    if folder == start_folder:
        started = True
    if started:
        output_path = os.path.join(output_folder, f'{folder}.sh')
        with open(output_path, 'w') as f:
            f.write(content.format(sequence_name=folder))

