import cv2 as cv
import os
from ultralytics import YOLO
import supervision as sv
import pickle
from utils import get_width_of_bbox, get_centre_of_bbox, get_foot_positions
import numpy as np
import pandas as pd


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        detections = [] 
        batch_size = 20 # 20 frames will be detected at a time
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
            # break ##JUST FOR NOW
        return detections

    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        # tracks = {
        #     'players': [
        #         {0:{1,2,3,4}, <track_id>: <bbox>, } # for the first frame
        #         {<track_id>: <bbox> for each player present in frame } # for the second frame
        #         # if player moves out of frame in consecutive frames, its tracker id wont be shown
        #     ],
        #     #same format for referees   
        #     'referee': [],
        #     'ball': []
        # }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names # 2:player 1:goalkeeper 0:ball
            class_names_inversed = {v:k for k,v in class_names.items()} # player:2 goalkeeper:1 ball:0

            # converting into supervision detection format
            supervision_detection = sv.Detections.from_ultralytics(detection)

            # convert the goalkeeper into player
            for object_index, class_id in enumerate(supervision_detection.class_id):
                if class_names[class_id] == 'goalkeeper':
                    supervision_detection.class_id[object_index] = class_names_inversed['player']

            # with Tracks
            detection_with_tracks = self.tracker.update_with_detections(supervision_detection)
            # details about the above detection with tracks object
                # 0 contains the bounding boxes
                # 1 contains the mask attribute
                # 2 contains the confidence arrays
                # 3 contains the class id
                # 4 containst he tracker id
                # 5 contains the data dictionary

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inversed['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if class_id == class_names_inversed['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in supervision_detection:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inversed['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks # a dictionary of list of dictionaries
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, y_center = get_centre_of_bbox(bbox=bbox)
        width = get_width_of_bbox(bbox=bbox)
        
        cv.ellipse(frame,
                   (x_center,y2), 
                   axes = (int(width), int(0.35*width)),
                   angle=0,
                   startAngle=-45,
                   endAngle=235,
                   color=color,
                   thickness=2,
                   lineType= cv.LINE_4
                   )
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y_center - rectangle_height//2) + 15
        y2_rect = (y_center + rectangle_height//2) + 15

        if track_id is not None:
            cv.rectangle(frame,
                         (x1_rect,y1_rect),
                         (x2_rect,y2_rect),
                         color,
                         cv.FILLED
                         )
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10
            cv.putText(frame,f"{track_id}",(int(x1_text), int(y1_rect+15)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,0),2) 
                         

        return frame
        
    def draw_triangle(self, frame, bbox, color, track_id=None):
        y = int(bbox[1])
        x,_ = get_centre_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
            ]
        )  

        cv.drawContours(frame,[triangle_points],0,color,cv.FILLED)
        cv.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw semi transparent rectangle
        overlay = frame.copy()
        cv.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        transparency = 0.4
        cv.addWeighted(overlay, transparency,frame,1-transparency,0,frame)


        team_ball_control_till_current_frame = team_ball_control[:frame_num+1]
        #get the number of times each team has the ball
        team1_posession = team_ball_control_till_current_frame[team_ball_control_till_current_frame==1].shape[0]
        team2_posession = team_ball_control_till_current_frame[team_ball_control_till_current_frame==2].shape[0]
        team1 = (team1_posession)/(team1_posession+team2_posession)
        team2 = (team2_posession)/(team1_posession+team2_posession)
        cv.putText(frame, f"Team 1 Ball Posession:{team1*100:.2f}%", (1400,900),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv.putText(frame, f"Team 2 Ball Posession:{team2*100:.2f}%", (1400,950),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame

    def draw_annotations(self,video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num,frame in enumerate(video_frames):     #~ loop thru the frames
            frame_copy = frame.copy() # so that we dont draw on the original frames
            player_dict = tracks['players'][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():    #~ loop thru each player tracker in the frame
                player_color = player.get('team_color',(0,0,255))
                frame_copy = self.draw_ellipse(frame,player["bbox"],player_color,track_id)

                if player.get('has_ball', False):
                    self.draw_triangle(frame_copy, player['bbox'],(255,0,0))

            #draw the referee markings
            for _, referee in referee_dict.items():    #~ loop thru each referee tracker in the frame
                frame_copy = self.draw_ellipse(frame,referee["bbox"],(255,0,0))

            #draw the ball pointer
            for _,ball in ball_dict.items():
                frame_copy = self.draw_triangle(frame, ball["bbox"], (0,255,0) , _)
            
            #draw team ball control
            frame_copy = self.draw_team_ball_control(frame_copy,frame_num,team_ball_control)

            output_video_frames.append(frame_copy)

        return output_video_frames

    def add_positions_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_centre_of_bbox(bbox)
                    else:
                        position = get_foot_positions(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball(self, ball_positions):
        # get the bbox of the first track id and make an array of it
        # if not found the track no.1, return {},  if not found the bbox subsequently return []
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]

        # convert the array into a pandas dataframe
        df = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        
        # interpolate the missing values
        df = df.interpolate()
        # backfilling can be done for the edge cases
        df = df.bfill()

        # make a dictionary of 1:bbox:[] for all the frames of the ball positions
        interpolated_positions = [{1:{'bbox':x}} for x in df.to_numpy().tolist()]
        # return the data in the aforementioned original format 
        return interpolated_positions