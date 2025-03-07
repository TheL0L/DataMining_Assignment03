from typing import List, Tuple
import math

Point = Tuple[float, ...]
Cluster = List[Point]

def euclidean_distance(p1: Point, p2: Point) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

def average_distance(point: Point, cluster: Cluster) -> float:
    """Compute average distance from a point to all other points in a cluster."""
    if len(cluster) <= 1:
        return 0  # If the cluster has only one point, distance is 0
    return sum(euclidean_distance(point, other) for other in cluster if other != point) / (len(cluster) - 1)

def nearest_cluster_distance(point: Point, clusters: List[Cluster], own_cluster: Cluster) -> float:
    """Find the minimum average distance from the point to the nearest other cluster."""
    min_dist = float("inf")
    for cluster in clusters:
        if cluster is own_cluster:  # Skip the cluster containing the point
            continue
        avg_dist = sum(euclidean_distance(point, other) for other in cluster) / len(cluster)
        min_dist = min(min_dist, avg_dist)
    return min_dist

def silhouette_score(clusters: List[Cluster]) -> float:
    """Compute the average silhouette score for all points in all clusters."""
    total_score = 0
    num_points = sum(len(cluster) for cluster in clusters)  # Total number of points

    if num_points <= 1:
        return 0  # Silhouette score is undefined for a single point

    for cluster in clusters:
        for point in cluster:
            a = average_distance(point, cluster)  # Intra-cluster distance
            b = nearest_cluster_distance(point, clusters, cluster)  # Nearest other cluster distance
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0  # Silhouette formula
            total_score += s

    return total_score / num_points  # Average over all points
