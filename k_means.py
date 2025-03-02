import math
import random

def k_means(dim: int, k: int|None, n: int, points: list[tuple[float]], clusts=[]) -> None:
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
        k = find_optimal_k(dim, points, range(1, 6))
    k = max(1, min(k, n))  # bound k in [1, n]

    points_list = [list(p) for p in points]

    centroids = random.sample(points_list, k)

    def distance(point1, point2):
        return math.sqrt(sum((point1[i] - point2[i])**2 for i in range(dim)))


    for _iter in range(n):
        new_clusters = [[] for _ in range(k)]

        for p in points_list:
            min_dist = float('inf')
            min_idx = 0
            for i, c in enumerate(centroids):
                d = distance(p, c)
                if d < min_dist:
                    min_dist = d
                    min_idx = i
            new_clusters[min_idx].append(p)

        if _iter > 0:  
            same = True
            prev_sorted = sorted([sorted(cluster) for cluster in clusts], key=len)
            new_sorted = sorted([sorted(cluster) for cluster in new_clusters], key=len)
            
            
            if len(prev_sorted) == len(new_sorted):
                for old_cl, new_cl in zip(prev_sorted, new_sorted):
                    if old_cl != new_cl:
                        same = False
                        break
            else:
                same = False

            if same:
                clusts = new_clusters
                break

        new_centroids = []
        for cluster in new_clusters:
            if len(cluster) > 0:
                c_sum = [0.0]*dim
                for p in cluster:
                    for i in range(dim):
                        c_sum[i] += p[i]
                for i in range(dim):
                    c_sum[i] /= len(cluster)
                new_centroids.append(c_sum)
            else:
                new_centroids.append(random.choice(points_list))

        centroids = new_centroids

        clusts.clear()
        for cluster in new_clusters:
            clusts.append(cluster)


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
