#!/bin/bash

# Define the target directory
TARGET_DIR="/users/rfu7/ssrinath/datasets/Action/brics-mini/2024-06-17/mano"

# Change to the target directory
cd "$TARGET_DIR" || exit

# Loop through all .mp4 files in the directory
for file in *.mp4; 
do 
  # Extract the base name without the extension
  base_name="${file%.mp4}"
  
  # Construct the new file name
  new_file="${base_name}_vit.mp4"
  
  # Rename the file
  mv "$file" "$new_file"
done

echo "Renaming complete."
