from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        # self.kmeans
        self.player_team_dict = {} # player_id:team1 or team 2, eg: 17:1

    def get_clustering_model(self, top_half):
        top_half_2d = top_half.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init='k-means++', n_init=1, random_state=0)
        kmeans.fit(top_half_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image = image[0: int(image.shape[1]//2),:]


        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0],clustered_image[-1,0],clustered_image[0,-1],clustered_image[-1,-1]]
        background_cluster = max(corner_clusters, key=corner_clusters.count)
        player_cluster = 1 - background_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color        

    def assign_team_color(self, frame,player_detections):
        player_colors = []
        for track_id,player in player_detections.items():
            bbox=  player['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init='k-means++',n_init=1)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        # 1 means team1 and 2 means team 2
        #example of color dictionary: {1:green, 2:yellow}
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        #use this if posession issue happens
        goalkeeper_id = 1231231
        if player_id == goalkeeper_id:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id