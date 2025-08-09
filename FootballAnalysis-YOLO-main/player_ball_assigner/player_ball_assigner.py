import sys
sys.path.append('../')
from utils import get_centre_of_bbox, measure_distance

class PlayerBallAsgn:
    def __init__(self):
        self.max_player_ball_dist = 70
    
    def assign_ball_to_player(self, ball_positions, player_positions):
        ball_positions = get_centre_of_bbox(ball_positions)
        minimum_distance = 99999
        assigned_player = -1

        for track_id, player in player_positions.items():
            player_bbox = player['bbox']
            
            #measure distance between left foot and ball
            distance_left = measure_distance((player_bbox[0],player_bbox[-1]), ball_positions)
            #measure distance between right foot and ball
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]), ball_positions)
            # get minimum distance
            distance = min(distance_left, distance_right)
            if distance < self.max_player_ball_dist:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = track_id
        
        return assigned_player