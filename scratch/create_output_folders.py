import os


input_folder = '/users/rfu7/ssrinath/brics/non-pii/brics-mini'
output_folder = '/users/rfu7/ssrinath/datasets/Action/brics-mini'
start_folder = '2024-10-01-action-clarajin-present'

started = False
for folder in os.listdir(input_folder):
    if folder == start_folder:
        started = True
    src_folder = os.path.join(input_folder, folder)
    dst_folder = os.path.join(output_folder, folder)
    if started and not os.path.exists(dst_folder):
        os.makedirs(dst_folder)