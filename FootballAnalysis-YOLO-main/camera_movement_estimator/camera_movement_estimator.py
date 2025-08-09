import cv2 as cv
import pickle
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_distance, measure_xy_distance, get_centre_of_bbox

class CameraMovementEstimator:
    def __init__(self, frame):
        first_frame_grayscale  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask_features  = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.minimum_distance_moved = 5

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask = mask_features
        )

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03 )
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames,read_from_stubs = False, stub_path = None):
        #Read the stub
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)
        old_gray = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
        old_features  = cv.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv.cvtColor(frames[frame_num], cv.COLOR_BGR2GRAY)
            new_features, _, _ = cv.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features, None,**self.lk_params)
            max_distance = 0
            camera_movement_x,camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(new_features_point, old_features_point)
            
            if max_distance > self.minimum_distance_moved:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay  = frame.copy()

            cv.rectangle(overlay, (0,0), (500,100),(255,255,255),-1)
            alpha = 0.6
            cv.addWeighted(overlay, alpha, frame, 1-alpha,0,frame)
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv.putText(frame, f"Camera Movement X: {x_movement:.2f} ", (10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv.putText(frame, f"Camera Movement Y: {y_movement:.2f} ", (10,60),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)
        
        return output_frames

    