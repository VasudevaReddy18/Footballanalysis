from utils import read_video, save_video
from trackers import Tracker
import cv2 as cv
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAsgn
from camera_movement_estimator import CameraMovementEstimator
import os


def main():
<<<<<<< HEAD
    video_frames = read_video("input_videos/43499-360.mp4")
=======
    video_path = "input_videos/223067_small.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_frames = read_video(video_path)
    if len(video_frames) == 0:
        raise ValueError(f"No frames read from video: {video_path}. Check file format/codecs.")
    
    print(f"Number of frames read: {len(video_frames)}")

    # Initialize tracker and get object tracks
>>>>>>> ffefc1e (Football Analysis)
    tracker = Tracker(model_path='yolov8l.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/tracks_stubs.pkl")
    tracker.add_positions_to_tracks(tracks)

    # Camera movement
    camera_movement_estimator = CameraMovementEstimator(frame=video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stubs=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])

    # Team assignment
    team_assigner = TeamAssigner()
    if len(tracks['players']) > 0:
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    num_frames = min(len(video_frames), len(tracks['players']))

    # Assign team colors safely
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        for player_id, track in player_track.items():
            bbox = track.get('bbox', None)
            if bbox is None:
                continue
            team = team_assigner.get_player_team(video_frames[frame_num], bbox, player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, [0, 0, 255])

    # Assign ball possession safely
    player_ball_assigner = PlayerBallAsgn()
    num_frames_ball = min(num_frames, len(tracks['ball']))
    team_ball_control = []

    for frame_num in range(num_frames_ball):
        player_track = tracks['players'][frame_num]
        ball_data = tracks['ball'][frame_num]
        if len(ball_data) == 0:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            continue

        ball_bbox = ball_data[1].get('bbox', None)
        if ball_bbox is None:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
            continue

        assigned_player = player_ball_assigner.assign_ball_to_player(ball_positions=ball_bbox, player_positions=player_track)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    # Draw annotations
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks, team_ball_control=team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Ensure output folder exists
    os.makedirs("output_videos", exist_ok=True)
    save_video(output_video_frames, "output_videos/output.avi")

    print("Processing complete! Output saved to output_videos/output.avi")


if __name__ == "__main__":
    main()
