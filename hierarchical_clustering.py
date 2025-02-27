
def h_clustering(dim, k, points, dist, clusts=[]):
    """
    Performs hierarchical clustering on a set of given points.

    Args:
        dim - points dimension
        k - clusters count, if None, then algorithmically pick a stopping value
        points - list of points to cluster
        dist - distance function, if None, then assume euclidian distance
        clusts (default=[]) - output list of clusters
    """
    if not points:
        return
    if dim < 1:
        raise ValueError('Dimension must be greater than zero.')
    if not (0 < k < len(points)):
        raise ValueError('K must be between 1 and len(points).')
    if dist is None:
        dist = lambda p, q: sum([(p[i] - q[i])**2 for i in range(dim)])
    
    def similarity(cluster1, cluster2):
        """Calculate the similarity of two clusters."""
        get_centroid = lambda c: tuple(sum(axis) for axis in zip(*c))
        return dist(get_centroid(cluster1), get_centroid(cluster2))

    # HAC algorithm, each point starts in it own cluster
    #clusts = [[p] for p in points]  # reference reassignment
    clusts.clear()  # clearing old values, just in case the function is reused during tests...
    for p in points:
        clusts.append([p])

    while len(clusts) > k:
        for i, c in enumerate(clusts):
            if c is None:
                continue
            best_index = -1
            best_value = float('inf')
            for j in range(i+1, len(clusts)):
                sim = similarity(c, clusts[j])
                if sim < best_value:
                    best_index = j
                    best_value = sim
            if best_index > 0:
                c.extend(clusts[best_index])
                clusts[best_index] = None
            #clusts = [c for c in clusts if c]  # reference reassignment
            temp = [c for c in clusts if c]
            clusts.clear()
            for c in temp:
                if c:
                    clusts.append(c)


if __name__ == '__main__':
    points = [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)]
    clusters = [None, None]
    h_clustering(1, 2, points, None, clusters)

    print(clusters)