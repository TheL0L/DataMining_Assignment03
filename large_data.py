import csv, random
from typing import List, Tuple

__MEMORY_USAGE_LIMIT = 256 * 1024**2  # 256MB
__PYTHON_FLOAT_SIZE = 8  # 8 Bytes

def generate_data(dim, k, n, out_path, points_gen=None, extras={}):
    """
    Generate N random points, uniformly distributed, across K clusters.

    Args:
        dim - point dimension
        k - number of clusters
        n - number of points
        out_path - output csv file path
    """
    def generate_random_centroid():
        return tuple([random.uniform(-100, 100) for i in range(dim)])

    def generate_random_point(offset, j):
        return tuple([offset[i] + random.uniform(-10, 10) for i in range(dim)] + [j])

    # calculate batch size based on expected float size and dimentionality.
    BATCH_SIZE = __MEMORY_USAGE_LIMIT // (dim * __PYTHON_FLOAT_SIZE)
    if BATCH_SIZE < 1:
        raise Exception('Insufficient memory to generate a single point.')
    print(f'Required batches {n // BATCH_SIZE} of size {BATCH_SIZE}.')
    print(f'Estimated max memory  {n * dim * __PYTHON_FLOAT_SIZE / 1024**3:.3f} GB.')
    print(f'Estimated max storage {n * dim * 20 / 1024**3:.3f} GB.')

    centroids = [generate_random_centroid() for _ in range(k)]

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)

        points_written = 0
        while points_written < n:
            print(f'Progress: {points_written} / {n}')
            
            batch = []
            for _ in range(min(BATCH_SIZE, n - points_written)):
                j = random.randint(0, k - 1)
                batch.append(generate_random_point(centroids[j], j))
            
            writer.writerows(batch)
            points_written += len(batch)
    
    return centroids

def save_points(clusters: List[List[Tuple[float, ...]]], output_path: str) -> None:
    """
    Write clustered points into a file in the following format:
        row: `axis1, axis2, ..., axisN, clusterID`
    
    Args:
        clusters - list of clusters
        output_path - output file path
    """
    output_handler = open(output_path, 'w', newline='')
    writer = csv.writer(output_handler)

    for k, cluster in enumerate(clusters):
        for point in cluster:
            writer.writerow([*point, k])

    output_handler.close()

if __name__ == '__main__':
    ...

