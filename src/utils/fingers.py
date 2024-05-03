# Indices of fingers
# If none of the keypoints are present for any finger, skip the frame
FINGER_IDX = [list(range(2, 5))] + [ list(range(i, i+4)) for i in range(5, 18, 4) ]

# Indices of finger tips
# If any of them are missing, skip the frame
TIP_IDX = [4, 8, 12, 16, 20]