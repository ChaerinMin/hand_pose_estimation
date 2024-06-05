import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import ipdb

class Reader():
    iterator = []

    def __init__(
            self, inp_type: str, path: str, undistort: bool=False, cams_to_remove=[], ith: int=0, start_frame_path=None
        ):
        """ith: the ith video in each folder will be processed."""
        self.type = inp_type
        self.ith = ith
        self.frame_count = int(1e9)
        self.start_frames = None
        if self.type == "video":
            self.streams = {}
            self.vids = []
            for cam in os.listdir(path):
                if 'imu' not in cam and len(glob(f"{path}/{cam}/*.mp4")) > self.ith:
                    if cam not in cams_to_remove:
                        self.vids.append(natsorted(glob(f"{path}/{cam}/*.mp4"))[self.ith])
            self.init_videos()
            if start_frame_path:
                with open(start_frame_path, 'r') as file:
                    start_frames = json.load(file)
                self.start_frames = start_frames
        else:
            pass


        # Sanity checks
        assert (self.frame_count > 0) and (self.frame_count < int(1e9)), "No frames found"

        self.cur_frame = 0
    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.frame_count:
            return {}

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            if self.start_frames:
                start_frame = self.start_frames.get(cam_name, [0, 0])[0]
            else:
                start_frame = 0
            cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + start_frame)
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
            frames[cam_name] = frame
        
        return frames

    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_videos()

        self.cur_frame = 0

    def init_videos(self):
        """ Create video captures for each video
                ith: the ith video in each folder will be processed."""
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            frame_count = int(ffmpeg.probe(vid, cmd="ffprobe")["streams"][0]["nb_frames"])
            print(frame_count, self.frame_count, vid)
            self.frame_count = min(self.frame_count, frame_count)
            cam_name = os.path.basename(vid).split(".")[0]
            self.streams[cam_name] = cap
            
        self.frame_count -= 5 # To account for the last few frames that are corrupted

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames
        
        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
                
            yield frame, self.cur_frame

        # Reinitialize the videos
        self.reinit()

if __name__ == "__main__":
    reader = Reader("video", "/hdd_data/common/BRICS/hands/peisen/actions/abduction_adduction/", 5, 16, 3)
    for i in range(len(reader)):
        frames, frame_num = reader.get_frames()
        print(frame_num)
