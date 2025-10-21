from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}  # player_id: team1 or team2

    def get_clustering_model(self, top_half):
        top_half_2d = top_half.reshape(-1, 3)
        if top_half_2d.shape[0] == 0:
            return None
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(top_half_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        # ensure bbox is within frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None  # skip invalid bbox

        image = frame[y1:y2, x1:x2]
        # handle very small boxes
        if image.size == 0:
            return None

        top_half_image = image[:max(1, image.shape[0]//2), :]  # take top half
        if top_half_image.size == 0:
            return None

        kmeans = self.get_clustering_model(top_half_image)
        if kmeans is None:
            return None

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0], clustered_image[-1,0], clustered_image[0,-1], clustered_image[-1,-1]]
        background_cluster = max(corner_clusters, key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color        

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        valid_players = []
        for track_id, player in player_detections.items():
            bbox = player['bbox']
            color = self.get_player_color(frame, bbox)
            if color is not None:
                player_colors.append(color)
                valid_players.append(track_id)

        if len(player_colors) < 2:
            # fallback if too few valid players
            self.team_colors = {1: np.array([0,255,0]), 2: np.array([0,0,255])}
            self.kmeans = None
            return

        player_colors = np.array(player_colors)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=0)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            # fallback default team
            team_id = 1
        else:
            player_color = self.get_player_color(frame, player_bbox)
            if player_color is None:
                team_id = 1
            else:
                team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
