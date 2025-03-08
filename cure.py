import random, csv
from small_data import read_single_point
from hierarchical_clustering import h_clustering as hac
from metrics import euclidean_distance
from metrics import Point, Cluster
from typing import List

_R = 10
_ALPHA = 0.2
_MERGE_THRESHOLD = 1.2

class _CURE_Cluster:
    index = 0
    def __init__(self, points: List[Point]):
        self.index = _CURE_Cluster.index
        _CURE_Cluster.index += 1
        self.n = len(points)
        self.points = points.copy()
        self.centroid = self.__compute_centroid()
        self.radius = self.__compute_radius()
        self.representatives = self.__select_representatives()
    
    def __compute_radius(self) -> float:
        return max(euclidean_distance(self.centroid, p) for p in self.points)
    
    def __compute_centroid(self) -> Point:
        return tuple(sum(axis) / self.n for axis in zip(*self.points))
    
    def __select_representatives(self):
        reps_count = max(self.n, _R)
        if reps_count < 1:
            reps = []
        else:
            # start with a random point
            reps = [random.choice(self.points)]
            
            while len(reps) < reps_count:
                # find the point in the cluster that is farthest from the closest representative
                farthest_point = max(
                    self.points,
                    key=lambda p: min(euclidean_distance(p, rep) for rep in reps)
                )
                reps.append(farthest_point)
        
        # shrink representatives
        shrunk = []
        for rep in reps:
            new_point = tuple((1 - _ALPHA) * r + _ALPHA * c for r, c in zip(rep, self.centroid))
            shrunk.append(new_point)
        
        return shrunk

    def get_closest_distance(self, points: List[Point]) -> float:
        """
        return the closest possible distance between the points and the representatives.
        """
        best_overall = float('inf')
        for rep in self.representatives:
            min_distance = min(euclidean_distance(rep, p) for p in points)
            if min_distance < best_overall:
                best_overall = min_distance
        return best_overall


def reservoir_sample_points(file_path: str, sample_size: int, dim: int) -> List[Point]:
    sample_points = []
    count = 0
    with open(file_path, 'r') as input_handler:
        reader = csv.reader(input_handler)
        for row in reader:
            point = read_single_point(row, dim)
            count += 1
            if len(sample_points) < sample_size:
                sample_points.append(point)
            else:
                # randomly replace an element in the reservoir with decreasing probability
                r = random.randint(0, count - 1)
                if r < sample_size:
                    sample_points[r] = point
    return sample_points

def cure_cluster(dim: int, k: int, n: int, block_size: int, in_path: str, out_path: str) -> None:
    """
    this implementation will guarantee "at most k clusters".
    couldn't think of a way to guarantee exactly k clusters in time,
        as it is highly depended on many factors (merge_threshold for example).
    """
    if dim < 1:
        raise ValueError('Dimension must be greater than zero.')
    
    # 1. make initial clustering based on a sample of the data
    sample_points = reservoir_sample_points(in_path, block_size, dim)
    clusters = []
    hac(dim, k, sample_points, euclidean_distance, clusters)
    clusters = [_CURE_Cluster(c) for c in clusters]  # convert to CURE format

    # 2. merge clusters based on representative proximity
    for i, cluster in enumerate(clusters):
        distance = float('inf')
        closest_cluster = -1
        for j in range(i + 1, len(clusters)):
            distance = cluster.get_closest_distance(clusters[j].representatives)

        # merge clusters only if they're "close enough"
        # close enough is defined as "within at least one of the radii * scalar"
        if distance < _MERGE_THRESHOLD \
            * min(cluster.radius, clusters[closest_cluster].radius):
            # the later cluster will get assigned the earlier index
            clusters[closest_cluster].index = cluster.index

    input_handler = open(in_path, 'r')
    output_handler = open(out_path, 'w', newline='')
    reader = csv.reader(input_handler)
    writer = csv.writer(output_handler)

    def read_batch():
        batch = []
        for _ in range(block_size):
            try:
                batch.append(read_single_point(next(reader), dim))
            except StopIteration:
                break
        return batch
    
    # 3. assign clusters to all points (since there is no reason to expand the clusters, this will suffice)
    points_clustered = 0
    while points_clustered < n:
        batch = read_batch()
        if not batch:
            break

        # for each point in the batch
        for point in batch:
            # find cluster with closest representative
            distances = {
                c.index:c.get_closest_distance([point,])
                for c in clusters
            }
            index = min(distances, key=distances.get)
            # and write the point assignment to output
            writer.writerow([*point, index])
        
        points_clustered += len(batch)

    input_handler.close()
    output_handler.close()


if __name__ == '__main__':
    import plot_util, large_data, comparison

    large_data.generate_data(2, 5, 1000, './data/test_points.csv', None, None)
    cure_cluster(
        2, 5, 1000, 100, 
        './data/test_points.csv',
        './data/results/test_points_cure.csv'
    )
    plot_util.plot_from_storage_2d('./data/test_points.csv')
    plot_util.plot_from_storage_2d('./data/results/test_points_cure.csv')

    actual_clusters = comparison.construct_clustering('./data/test_points.csv')
    predicted_clusters = comparison.construct_clustering('./data/results/test_points_cure.csv')
    acc = comparison.compare_clusterings(actual_clusters, predicted_clusters)
    print(f'{acc=}')
