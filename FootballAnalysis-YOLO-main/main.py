from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAsgn
from camera_movement_estimator import CameraMovementEstimator
# from view_transformer import ViewTransformer


def main():
    video_frames = read_video("input_videos/input.mp4")
    tracker = Tracker(model_path='models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stubs.pkl")
    #get object positions
    tracker.add_positions_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(frame=video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                              read_from_stubs=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_positions_to_tracks(tracks=tracks)
    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])
    #save the cropped iamge of the player
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],track['bbox'],player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assign ball possession to current player
    player_ball_assigner = PlayerBallAsgn()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_ball_assigner.assign_ball_to_player(ball_positions=ball_bbox, player_positions=player_track)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # else we add the last person that had the ball 
            # this case is triggered when ball in mid-pass
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)


    # draw output and object tracks
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks, team_ball_control = team_ball_control)

    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
                                    
    
    save_video(output_video_frames,"output_videos/output.avi")

if __name__ == "__main__":
    main()