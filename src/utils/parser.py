from argparse import ArgumentParser

def add_common_args(parser: ArgumentParser):
    parser.add_argument("--root_dir", "-r", required=False, type=str)
    parser.add_argument("--video_dir", "-v", type=str, required=False, help="folder where video directories are stored. only for cam name reading purpose.")
    parser.add_argument("--out_dir", "-o", required=False, type=str, help="Output directory")
    parser.add_argument("--input_type", "-t", default="video", choices=["video", "image"], help="Whether the input is a video or set of images")
    parser.add_argument("--seq_path", "-s", type=str, required=False)
    parser.add_argument("--ith", default=0, type=int, help="The #ith video/image to process.")
    parser.add_argument("--anchor_camera", default=None, type=str, help="The anchor camera for timestamp alignment.")
    parser.add_argument("--handedness", choices=["left", "right"], default="right", type=str)
    parser.add_argument("--undistort", action="store_true", default=False)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--start", default=0, type=int, help="Which video to start")
    parser.add_argument("--end", default=-1, type=int, help="Which video to end")
    parser.add_argument("--len_timestep", type=int, default=150, help="Which frame to end. Start is 0")
    parser.add_argument(
        "-m", "--multisequence", type=str, required=True, help="multisequence0000001"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="1 is before undistort. 2 assumes already undistorted"
    )