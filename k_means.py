from cluster import Cluster
from metrics import euclidean_distance, get_k_upper_bound, silhouette_score
from metrics import Point
import random

MAX_ITERATIONS = 100


def k_means(dim: int, k: int|None, n: int, points: list[tuple[float, ...]], clusts=[]) -> None:
    """
    Perform k-means clustering on a set of n-dimensional points.

    :param dim: The dimension of the points.
    :param k: The number of clusters to form.
    :param n: The number of points.
    :param points: The points to cluster.
    :param clusts: Holds the final clusters list[list[tuple[float]]]
    :return: None
    """
    clusts.clear()

    if k is None:
        k = get_best_k(dim, points)
    k = max(1, min(k, n))  # bound k in [1, n]

    points_list = [list(p) for p in points]

    # Randomly assign points to k clusters initially
    init_clusters = [[] for _ in range(k)]
    for p in points_list:
        init_clusters[random.randrange(k)].append(p)

    for i in range(k):
        if not init_clusters[i]:
            init_clusters[i].append(random.choice(points_list))

    cluster_objs = [Cluster(cluster) for cluster in init_clusters]

    for _ in range(MAX_ITERATIONS):
        new_clusters = [[] for _ in range(k)]
        for p in points_list:
            distances = [euclidean_distance(p, cluster.centroid) for cluster in cluster_objs]
            min_index = distances.index(min(distances))
            new_clusters[min_index].append(p)
        changed = any(sorted(new_clusters[i]) != sorted(cluster_objs[i].points) for i in range(k))
        if not changed:
            break
        for i in range(k):
            if not new_clusters[i]:
                new_clusters[i].append(random.choice(points_list))
        cluster_objs = [Cluster(cluster) for cluster in new_clusters]

    clusts.extend([cluster.points for cluster in cluster_objs])

def get_best_k(dim: int, points: list[Point]) -> int:
    if len(points) < 2:
        return 1
    scores = dict()
    n = len(points)
    k_upper_bound = get_k_upper_bound(n)
    for k in range(2, k_upper_bound):
        clusters = []
        k_means(dim, k, n, points, clusters)
        scores[k] = silhouette_score(clusters)

    return max(scores, key=scores.get)

def find_optimal_k(dim: int, points: list[tuple[float]], k_values: range) -> None:
    """
    Try multiple k values, run k-means for each, and compute SSE.
    This is commonly used with the Elbow Method: pick the k
    where SSE stops decreasing significantly.

    :param dim: Dimension of each point.
    :param points: The data points to be clustered.
    :param k_values: A range or list of k values to test, e.g., range(1, 10).
    :return: None (prints the SSE for each k).
    """
    n = len(points)

    def calc_SSE(dim: int, clusters: list[list[list]], centroids: list[list[float]]):
        """
        Calculate the sum of squared errors for the given clusters and centroids.
        :param dim: Dimension of each point.
        :param clusters: The list of clusters (each cluster is a list of points).
        :param centroids: The list of centroid coordinates for each cluster.
        :return: SSE (float).
        """
        sse = 0.0
        for cluster_idx, cluster in enumerate(clusters):
            c = centroids[cluster_idx]
            for p in cluster:
                dist_sq = sum((p[i] - c[i])**2 for i in range(dim))
                sse += dist_sq
        return sse

    for k in k_values:
        clusts = []
        k_means(dim, k, n, points, clusts)
        centroids = calculate_centroids(dim, clusts)
        sse = calc_SSE(dim, clusts, centroids)

        print(f"For k = {k}, SSE = {sse:.2f}")

    return k



def calculate_centroids(dim: int, clusts: list[list[tuple[float]]]) -> list[tuple[float]]:
    """
    Calculate the centroid of each cluster.

    :param dim: The dimension of the points.
    :param clusts: The list of clusters.
    :return: A list of centroids.
    """
    centroids = []
    for cluster in clusts:
        if cluster:
            c_sum = [0.0] * dim
            for p in cluster:
                for i in range(dim):
                    c_sum[i] += p[i]
            for i in range(dim):
                c_sum[i] /= len(cluster)
            centroids.append(tuple(c_sum))
        else:
            centroids.append(tuple([0.0]*dim))

    return centroids




if __name__ == '__main__':
    points = [
        [1, 2], [2, 1], [3, 2],
        [8, 9], [9, 8], [10, 10],
        [50, 50], [49, 52]
    ]
    
    dim = len(points[0])
    k = None
    n = len(points)
    clusts = []
    k_means(dim, k, n, points, clusts)
    
    for i, cluster in enumerate(clusts, start=1):
        print(f"Cluster {i}: {cluster}")
