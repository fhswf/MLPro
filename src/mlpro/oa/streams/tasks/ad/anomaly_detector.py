from mlpro.oa.streams.tasks.ad.cluster_generator import ClusterGenerator
from mlpro.oa.streams.tasks.ad.cluster_analyzer import ClusterAnalyzer
import numpy as np

cluster_gen = ClusterGenerator(num_clusters=3)

# Add clusters with their centers, velocities, and number of data points
cluster_gen.add_cluster(center=[0, 0], velocity=[0.02, 0.02], num_points=100)
cluster_gen.add_cluster(center=[5, 5], velocity=[-0.04, -0.08], num_points=80)
cluster_gen.add_cluster(center=[-5, 5], velocity=[0.0, 0.0], num_points=50)

generated_data = cluster_gen.generate_clusters()

# Print the generated data
for point in generated_data:
    print(point)

    # Generate some sample data
np.random.seed(0)
data = np.random.rand(200, 2) * 10

    # Create a ClusterAnalyzer
cluster_analyzer = ClusterAnalyzer(data)

    # Find the optimal number of clusters
num_clusters = cluster_analyzer.find_optimal_clusters(max_clusters=10)
print(f"The optimal number of clusters is: {num_clusters}")

    # Analyze the clusters
cluster_analyzer.analyze_clusters(num_clusters)

    # Calculate cluster density
cluster_density = cluster_analyzer.calculate_density()
print(f"Cluster Density: {cluster_density}")

    # Calculate cluster velocity (assuming a time interval of 1)
previous_centers = None  # Set previous_centers to the centers at the previous time step
cluster_velocity = cluster_analyzer.calculate_velocity(previous_centers)
print(f"Cluster Velocity: {cluster_velocity}")

