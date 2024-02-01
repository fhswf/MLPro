import numpy as np

class ClusterGenerator:
    def __init__(self, num_dimensions, num_clusters, num_points_per_cluster=1000, num_point_anomalies=10):
        self.num_dim = num_dimensions
        self.num_clusters = num_clusters
        self.num_points_per_cluster = num_points_per_cluster
        self.clusters = []
        self.num_point_anom = num_point_anomalies
        self.index = 0
        self.cluster = 0
        self.cycle = 0

    def add_cluster(self, center=None,
                    radius = 100,
                    velocity=0,
                    change_in_radius=False,
                    change_in_velocity=False,
                    change_in_density=False,
                    appears_later=False,
                    disappears=False,
                    merge=False):
        cluster = {
            "dimension": self.num_dim,
            "center": center,
            "num_points": self.num_points_per_cluster,
            "radius" : radius,
            "velocity": velocity,
            "change_in_radius" : change_in_radius,
            "change_in_velocity" : change_in_velocity,
            "change_in_density" : change_in_density,
            "appears_later" : appears_later,
            "disappears" : disappears,
            "merge" : merge,
        }
        self.clusters.append(cluster)

    def generate_clusters(self):
        data = []
        for cluster in self.clusters:
            center = cluster["center"]
            velocity = cluster["velocity"]
            num_points = self.num_points_per_cluster
            cluster_data = self.generate_cluster_data(center, velocity, num_points)
            data.extend(cluster_data)



            # Generate anomalies
        anomalies = np.random.rand(self.num_point_anom, self.num_dim)
        for i in range(self.num_dim):
            anomalies[:, i] = anomalies[:, i] * (max(X[:, i]) - min(X[:, i])) + min(X[:, i])
        return data

    def generate_cluster_data(self, center, velocity, num_points):
        data = []
        for _ in range(num_points):
            point = center + np.random.normal(0, 0.1, 2)  # Add some random noise
            data.append(point)
            center += velocity  # Move the center according to the velocity
        return data
    
    def get_instance():
        pass