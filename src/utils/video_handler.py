import numpy as np
import cv2
import os
import glob
import natsort
from PIL import Image

def frame_preprocess(path, use_parsed, args, intr=None, dist_intr=None, dist=None):
    if args.setting == "brics-mini":
        cam_name_slicer = slice(0, 21)
    elif args.setting == "brics-studio":
        cam_name_slicer = slice(0,18)
    else:
        raise NotImplementedError()
    if use_parsed:
        cam_name = os.path.basename(path).split('.')[0][cam_name_slicer]
        # multiseq_chr = args.out_dir.index("multisequence")
        # multiseq = args.out_dir[multiseq_chr:multiseq_chr+19]
        # data_root = args.out_dir[:multiseq_chr]
        parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")
        timestamp_dirs = natsort.natsorted(glob.glob(os.path.join(parsed_dir, "timestamp_*")))
        if args.len_timestep > 0:
            datalen = args.len_timestep
        else:
            datalen = len(timestamp_dirs)
        # datalen = min(args.len_timestep, len(timestamp_dirs))
    else:
        stream = cv2.VideoCapture(path)
        assert stream.isOpened(), 'Cannot capture source'
        datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    orig_imgs = []
    im_names = []
    frame_num = 0
    for k in range(datalen):
        if k % 3 == 0 or k % 3 == 1 or k % 3 == 2:
            if use_parsed:
                img_path = os.path.join(timestamp_dirs[k], "images", f"{cam_name}.jpg")
                if os.path.exists(img_path):
                    frame = np.array(Image.open(img_path))
                else:
                    frame = None
            else:
                (grabbed, frame) = stream.read()
                if not grabbed:
                    stream.release()
                    break
                if args.undistort and not (dist == 0).all():
                    frame = cv2.undistort(frame, intr, dist, None, dist_intr)
            orig_imgs.append(frame)
            im_names.append(f'{frame_num:08d}' + '.jpg')
            frame_num += 1
    for frame in orig_imgs:
        if frame is not None:
            H, W, _ = frame.shape
            break
    if not use_parsed:
        stream.release()

    return im_names, orig_imgs, H, W

def load_first_frame(path, use_parsed, args, intr=None, dist_intr=None, dist=None):
    """Load only the first frame from a video or parsed directory

    Returns:
        orig_img: numpy array of the first frame
        im_h: image height
        im_w: image width
        timestamp_name: name of the timestamp directory used (e.g. "timestamp_3"), or None for video input
    """
    if args.setting == "brics-mini":
        cam_name_slicer = slice(0, 21)
    elif args.setting == "brics-studio":
        cam_name_slicer = slice(0,18)
    else:
        raise NotImplementedError()
    if use_parsed:
        cam_name = os.path.basename(path).split('.')[0][cam_name_slicer]
        # multiseq_chr = args.out_dir.index("multisequence")
        # multiseq = args.out_dir[multiseq_chr:multiseq_chr+19]
        # data_root = args.out_dir[:multiseq_chr]
        # parsed_dir = os.path.join(data_root, multiseq, "parsed")
        parsed_dir = os.path.join(args.root_dir, args.seq_path, args.multisequence, "parsed")
        timestamp_dirs = natsort.natsorted(glob.glob(os.path.join(parsed_dir, "timestamp_*")))

        # Load first valid frame
        # brics-studio: images may be missing from some timestamps, scan until found
        # brics-mini: timestamp_0 always has all cameras, use it directly
        for k in range(len(timestamp_dirs)):
            img_path = os.path.join(timestamp_dirs[k], "images", f"{cam_name}.jpg")
            if args.setting == "brics-mini":
                frame = np.array(Image.open(img_path))
                H, W, _ = frame.shape
                return frame, H, W, os.path.basename(timestamp_dirs[k])
            if not os.path.exists(img_path):
                continue
            frame = np.array(Image.open(img_path))
            H, W, _ = frame.shape
            return frame, H, W, os.path.basename(timestamp_dirs[k])
        raise ValueError(f"No valid frames found in {parsed_dir}")
    else:
        stream = cv2.VideoCapture(path)
        assert stream.isOpened(), 'Cannot capture source'

        grabbed, frame = stream.read()
        if not grabbed:
            stream.release()
            raise ValueError(f"Cannot read first frame from {path}")

        if args.undistort and not (dist == 0).all():
            frame = cv2.undistort(frame, intr, dist, None, dist_intr)

        H, W, _ = frame.shape
        stream.release()

        return frame, H, W, None

def create_video_writer(filename, frame_size, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def convert_video_ffmpeg(input_path):
    temp_path = input_path + '.temp'
    output_path = input_path
    
    # Rename the original file
    os.rename(input_path, temp_path)
    
    # Construct the ffmpeg command with the -y option
    ffmpeg_command = f'ffmpeg -i {temp_path} -vcodec libx264 -y {output_path}'
    
    # Execute the command
    os.system(ffmpeg_command)
    
    # Remove the temporary file
    os.remove(temp_path)