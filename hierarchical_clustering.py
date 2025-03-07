from metrics import euclidean_distance, get_k_upper_bound, silhouette_score
from metrics import Point
import heapq

def h_clustering(dim, k, points, dist, clusts=[]):
    """
    Performs hierarchical clustering (centroid linkage) on a set of given points.

    Args:
        dim: Points dimension.
        k: Desired number of clusters. If None or less than 1, an optimal value is used.
        points: List of points to cluster (each point is a tuple of floats).
        dist: Distance function. If None, Euclidean distance is used.
        clusts (default=[]): Output list of clusters. Each cluster is a list of points.
    
    The function updates the `clusts` list in place.
    """
    if not points:
        return

    if dim < 1:
        raise ValueError('Dimension must be greater than zero.')
    
    if dist is None:
        dist = euclidean_distance

    if k is None or k < 1:
        k = get_best_k(dim, points)
    k = min(k, len(points))

    def get_centroid(cluster):
        return tuple(sum(coords) / len(cluster) for coords in zip(*cluster))
    
    # initialize active clusters: each point is its own cluster
    # assign each cluster a unique id, for efficient memory management
    active_clusters = {i: [point] for i, point in enumerate(points)}
    next_cluster_id = len(points)

    # build the initial priority queue (min-heap) of distances between each pair of clusters
    # each heap entry is tuple[distance, cluster_id1, cluster_id2]
    heap = []
    cluster_ids = list(active_clusters.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id1, id2 = cluster_ids[i], cluster_ids[j]
            centroid1 = get_centroid(active_clusters[id1])
            centroid2 = get_centroid(active_clusters[id2])
            d = dist(centroid1, centroid2)
            heapq.heappush(heap, (d, id1, id2))

    # merge clusters until there are exactly k clusters
    while len(active_clusters) > k:
        # pop the smallest distance pair
        d, id1, id2 = heapq.heappop(heap)
        # if either cluster is no longer active (already merged), skip it
        if id1 not in active_clusters or id2 not in active_clusters:
            continue

        # merge clusters id1 and id2
        merged_cluster = active_clusters[id1] + active_clusters[id2]
        # remove the old clusters
        del active_clusters[id1]
        del active_clusters[id2]
        # add the new merged cluster with a new unique id
        new_id = next_cluster_id
        next_cluster_id += 1
        active_clusters[new_id] = merged_cluster

        # calculate distances from the new cluster to all other active clusters
        new_centroid = get_centroid(merged_cluster)
        for other_id, other_cluster in active_clusters.items():
            if other_id == new_id:
                continue
            other_centroid = get_centroid(other_cluster)
            d = dist(new_centroid, other_centroid)
            # push the new pair to the heap
            heapq.heappush(heap, (d, other_id, new_id))
    
    # update the output clusters list with the final clusters
    clusts.clear()
    for cluster in active_clusters.values():
        clusts.append(cluster)

def get_best_k(dim: int, points: list[Point]) -> int:
    if len(points) < 2:
        return 1
    scores = dict()
    k_upper_bound = get_k_upper_bound(len(points))
    for k in range(2, k_upper_bound):
        clusters = []
        h_clustering(dim, k, points, euclidean_distance, clusters)
        scores[k] = silhouette_score(clusters)

    return max(scores, key=scores.get)


if __name__ == '__main__':
    ...

