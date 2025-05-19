.. _target_oa_cbad:
Cluster-based anomaly detection
===============================

Cluster-based anomaly detection uses clustering algorithms to form clustered data and identify anomolous behaviours of these clusters, which are flagged as anomalies.

Traditional anomaly detection methods often assume specific distributions or focus on individual data points. Cluster-based methods leverage group behaviors, identifying anomalies as deviations from the characteristics of a cluster. This makes them particularly useful for complex datasets with latent group structures.

Potential:

Dynamic anomaly detection: Works well with multi-dimensional data where relationships between features may indicate normal or anomalous behavior.
Adaptability: Can handle evolving data distributions by recalculating clusters dynamically.
Broad applicability: Used in diverse fields such as fraud detection, intrusion detection, and fault detection in industrial systems.

Advantages:

Can uncover contextual anomalies (data points that are anomalous in one cluster but normal in another).
Can identify global anomalies (points far from all clusters) and local anomalies (outliers within a cluster).
Scalability with various clustering algorithms, such as k-means for simpler scenarios or DBSCAN for non-spherical and dense data distributions.


**New types of anomalies**

  Cluster-based methods introduce nuanced anomaly categorizations:
  
  (a) Cluster-Centric Anomalies: Points significantly deviating from their assigned cluster’s centroid.
  
  (b) Inter-Cluster Anomalies: Points that do not fit well into any cluster, often lying between clusters.
  
  (c) Cluster Structural Anomalies: Unusual clusters themselves, such as unexpected densities, shapes, or sizes, signaling broader irregularities.


**Special dependencies on cluster algorithms**

Cluster-based anomaly detection heavily depends on the choice of clustering algorithm, as it directly impacts the detection process:

k-means: Effective for spherical clusters but may miss anomalies in datasets with non-convex shapes or varying densities.
DBSCAN: Ideal for discovering density-based anomalies but sensitive to hyperparameter tuning (e.g., minPts, ε).
Hierarchical Clustering: Useful for anomalies appearing at different levels of granularity.
Gaussian Mixture Models (GMM): Handles soft clustering and detects probabilistic anomalies but assumes data follows Gaussian distributions.
Spectral Clustering: Good for identifying anomalies in non-linear manifolds but computationally intensive for large datasets.
Effective anomaly detection requires understanding the clustering algorithm’s limitations and ensuring it aligns with the data characteristics and problem context.


**Cross reference**

- Howtos
- API
