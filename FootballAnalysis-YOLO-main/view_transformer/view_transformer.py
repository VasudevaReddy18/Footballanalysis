import numpy as np
import cv2 as cv

class ViewTransformer:
    def __init__(self):
        self.pitch_length = 23.32
        self.pitch_width = 68
        
        self.pixel_vertices = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
        ])

        self.target_vertices = np.array([
            [0,self.pitch_width],
            [0,0],
            [self.pitch_length,0],
            [self.pitch_length, self.pitch_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    
    def transform_point(self, position):
        p = (int(position[0]), int(position[1]))
        is_inside = cv.pointPolygonTest(self.pixel_vertices,p,False) >= 0
        if not is_inside:
            return None
        reshaped_point = position.reshape(-1,1,2).astype(np.float32)
        transform_point = cv.perspectiveTransform(reshaped_point,self.perspective_transformer)
        

        return transform_point.reshape(-1,2)



    def add_transformed_positions_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position  = track_info['position_adjusted']
                    position = np.array(position)
                    position_tranformed = self.transform_point(position)
                    if position_tranformed is not None:
                        position_tranformed = position_tranformed.squeeze().tolist()

                    tracks[object][frame_num][track_id]['position_transformed'] = position_tranformed