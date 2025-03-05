import math
from cluster import Cluster


def h_clustering(dim, k, points, dist, clusts=[]):
    """
    Performs hierarchical clustering on a set of given points.

    Args:
        dim: Dimension of the points.
        k: Desired number of clusters. If None, then a stopping value is chosen (defaulting to 1).
        points: List of points to cluster.
        dist: Distance function between two points. If None, Euclidean distance is used.
        clusts (default=[]): Output list of clusters (each as a Cluster object).
    """
    # If no distance function is provided, use Euclidean distance.
    if dist is None:
        def euclidean_distance(p1, p2):
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
        dist = euclidean_distance

    # If k is None, choose a default stopping value (e.g., 1 cluster).
    if k is None:
        k = 1

    # Initialize each point as its own cluster.
    clusters: list[Cluster] = [Cluster([list(p)]) for p in points]

    # Helper: compute the distance between two clusters using the provided distance function.
    def cluster_distance(c1: Cluster, c2: Cluster) -> float:
        return min(dist(p1, p2) for p1 in c1.points for p2 in c2.points)

    # Merge clusters until the number of clusters is k.
    while len(clusters) > k:
        min_dist = float('inf')
        merge_idx = (-1, -1)
        # Find the two clusters with the minimum distance.
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j])
                if d < min_dist:
                    min_dist = d
                    merge_idx = (i, j)

        i, j = merge_idx
        # Merge the two clusters by combining their points.
        merged_cluster = Cluster(clusters[i].points + clusters[j].points)
        # Remove the merged clusters (remove the higher index first to preserve order)
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(merged_cluster)

    # Store the final clusters in the output list.
    clusts.clear()
    clusts.extend([cluster.points for cluster in clusters])


if __name__ == '__main__':
    import csv
    from utils import generate_2d_data, plot_clusters

    # Step 1: Generate synthetic 2D data.
    true_k = 3              # True number of clusters in the synthetic data.
    points_per_cluster = 50
    cluster_std = 1.0
    generate_2d_data(true_k, points_per_cluster, cluster_std,
                     points_csv='points.csv',
                     ground_truth_csv='ground_truth.csv')

    # Step 2: Load points from the generated CSV file.
    points = []
    with open('points.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            x, y = map(float, row)
            points.append((x, y))

    # Step 3: Run k-means clustering.
    final_clusters = []
    n = len(points)
    h_clustering(dim=2, k=true_k, dist=None, points=points, clusts=final_clusters)
    print("Final clusters from h-clustering:")
    for idx, cluster in enumerate(final_clusters):
        print(f"Cluster {idx}: {cluster}")

    # Step 4: Save the k-means clustering output to a CSV file for plotting.
    kmeans_csv = 'h_clustering_output.csv'
    with open(kmeans_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["Cluster_ID", "Dim_0", "Dim_1"])
        for cluster_id, cluster_points in enumerate(final_clusters):
            for p in cluster_points:
                writer.writerow([cluster_id] + p)

    # Step 5: Plot the clusters from the k-means output.
    plot_clusters(kmeans_csv)