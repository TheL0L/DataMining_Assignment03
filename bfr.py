import csv, math
from small_data import read_single_point
from k_means import k_means
from metrics import euclidean_distance

class _BFR_Cluster:
    index = 0
    def __init__(self, dim):
        self.index = _BFR_Cluster.index
        _BFR_Cluster.index += 1
        self.n = 0
        self.sum = [0, ] * dim
        self.sumsq = [0, ] * dim
    
    def add_points(self, points):
        for i, axis in enumerate(zip(*points)):
            self.sum[i] += sum(axis)
            self.sumsq[i] += sum(x ** 2 for x in axis)
        self.n += len(points)
    
    def get_variance(self):
        return [
            (axis_sumsq / self.n) - (axis_sum / self.n)**2
            for axis_sum, axis_sumsq in zip(self.sum, self.sumsq)
        ]
    
    def get_standard_deviation(self):
        return [math.sqrt(axis) for axis in self.get_variance()]
    
    def get_centroid(self):
        return [axis / self.n for axis in self.sum]
    
    def merge(self, cluster):
        self.n += cluster.n
        self.sum = [sum(axis) for axis in zip(self.sum, cluster.sum)]
        self.sumsq = [sum(axis) for axis in zip(self.sumsq, cluster.sumsq)]


def bfr_cluster(dim, k, n, block_size, in_path, out_path):
    if dim < 1:
        raise ValueError('Dimension must be greater than zero.')
    
    input_handler = open(in_path, 'r')
    output_handler = open(out_path, 'w', newline='')
    reader = csv.reader(input_handler)
    writer = csv.writer(output_handler)

    cluster_index_mapping = {}  # the mapping will serve for translating indexes

    def read_batch():
        batch = []
        for _ in range(block_size):
            try:
                batch.append(read_single_point(next(reader), dim))
            except StopIteration:
                break
        return batch

    # initialize clusters via running k-means
    initial_clusters = []
    batch = read_batch()
    k_means(dim, k, len(batch), batch, initial_clusters)
    k = len(initial_clusters)  # grab k value decided by k-means if none was given

    # convert traditional clusters to bfr format
    _bfr_clusters = []  # Discard-Set
    for cluster in initial_clusters:
        _bfr_cluster = _BFR_Cluster(dim)
        _bfr_cluster.add_points(cluster)
        _bfr_clusters.append(_bfr_cluster)
        # iterate over points in new DS clusters and assign them an index before discarding
        for point in cluster:
            writer.writerow([*point, _bfr_cluster.index])  # write (point, cluster_index) to output

    points_clustered = len(batch)
    _bfr_miniclusters = []  # Compressed-Set
    _bfr_retained = []  # Retained-Set

    # keep iterating in batches, updating the bfr clusters
    while points_clustered < n:
        batch = read_batch()
        if not batch:
            break

        # 1. assign points to DS
        unclustered_points = []
        for point in batch:
            assigned = False
            for cluster in _bfr_clusters:
                centroid = cluster.get_centroid()
                std_dev = cluster.get_standard_deviation()
                
                # check if the point is within 2 standard deviations (mahalanobis distance)
                if all(abs(point[i] - centroid[i]) <= 2 * std_dev[i] for i in range(dim)):
                    cluster.add_points([point])
                    assigned = True
                    writer.writerow([*point, cluster.index])  # write (point, cluster_index) to output
                    break
            
            # if none of the clusters were close enough, keep it for the next step
            if not assigned:
                unclustered_points.append(point)

        # 2. cluster remaining points to CS and expand RS
        unclustered_points.extend(_bfr_retained)
        _bfr_retained.clear()  # clear RS, the points will be re-clustered

        if len(unclustered_points) > 1:
            miniclusters = []
            k_means(dim, k, len(unclustered_points), unclustered_points, miniclusters)

            for cluster in miniclusters:
                if len(cluster) > 1:  # if there are multiple points, expand CS
                    new_cs_cluster = _BFR_Cluster(dim)
                    new_cs_cluster.add_points(cluster)
                    _bfr_miniclusters.append(new_cs_cluster)
                    # iterate over points in new CS cluster and assign them an index before discarding
                    for point in cluster:
                        writer.writerow([*point, new_cs_cluster.index])  # write (point, cluster_index) to output
                else:  # otherwise, expand RS
                    _bfr_retained.append(cluster[0])
        
        # 3. merge close miniclusters in CS
        merged_clusters = set()
        for i, cluster1 in enumerate(_bfr_miniclusters):
            for j, cluster2 in enumerate(_bfr_miniclusters):
                if i >= j or j in merged_clusters:
                    continue  # skip duplicate comparisons
                
                centroid1 = cluster1.get_centroid()
                centroid2 = cluster2.get_centroid()
                std_dev1 = cluster1.get_standard_deviation()
                std_dev2 = cluster2.get_standard_deviation()
                
                # merge if centroids are close enough (within 2 standard deviations)
                if all(abs(centroid1[d] - centroid2[d]) <= 2 * max(std_dev1[d], std_dev2[d]) for d in range(dim)):
                    cluster1.merge(cluster2)
                    merged_clusters.add(j)
                    cluster_index_mapping[cluster2.index] = cluster1.index  # add index mapping for later translation
        
        # remove merged clusters from CS
        _bfr_miniclusters = [c for i, c in enumerate(_bfr_miniclusters) if i not in merged_clusters]

        points_clustered += len(batch)

    # write all points from DS with index=-1
    for point in _bfr_retained:
        writer.writerow([*point, -1])  # write (point, cluster_index) to output

    print(f'|DS|={len(_bfr_clusters)}    |CS|={len(_bfr_miniclusters)}    |RS|={len(_bfr_retained)}')
    print(f'DS: {[c.n for c in _bfr_clusters]}')
    print(f'CS: {[c.n for c in _bfr_miniclusters]}')

    # map each CS to the closest DS
    for minicluster in _bfr_miniclusters:
        centroid = minicluster.get_centroid()
        best_d = float('inf')
        best_index = -1
        for cluster in _bfr_clusters:
            d = euclidean_distance(centroid, cluster.get_centroid())
            if d < best_d:
                best_d = d
                best_index = cluster.index
        
        cluster_index_mapping[minicluster.index] = best_index
    
    # collapse (flatten) the mapping
    cluster_index_mapping = collapse_mapping(cluster_index_mapping)
    print(f'\nmapping:\n{cluster_index_mapping}')

    # close file handles and return mapping for cleanup
    input_handler.close()
    output_handler.close()
    return cluster_index_mapping


def collapse_mapping(mapping: dict[int, int]) -> dict[int, int]:
    collapsed = {}
    for swapped in mapping:
        current = swapped
        while current in mapping and current != mapping[current]:
            current = mapping[current]
        collapsed[swapped] = current
    return collapsed


def bfr_cleanup(input_path: str, output_path: str, mapping: dict[int, int]) -> None:
    input_handler = open(input_path, 'r')
    output_handler = open(output_path, 'w', newline='')
    reader = csv.reader(input_handler)
    writer = csv.writer(output_handler)

    for line in reader:
        axis, cluster = line[:-1], int(line[-1])
        new_line = [*axis, mapping[cluster] if cluster in mapping else cluster]
        writer.writerow(new_line)

    input_handler.close()
    output_handler.close()


if __name__ == '__main__':
    ...
