import csv
from math import ceil

from cluster import Cluster


def cure_cluster_k(dim: int, k: int, n: int, block_size: int, in_path: str):
    num_blocks = ceil(n / block_size)
    clusters: list[Cluster] = []

    # Step 1: Read and process data in blocks
    for block_idx in range(num_blocks):
        with open(in_path, 'r', newline='') as file:
            reader = csv.reader(file)

            block = []
            for i, raw_row in enumerate(reader):
                if i >= n:
                    break

                if block_size * block_idx <= i < block_size * (block_idx + 1):
                    block.append([float(value) for value in raw_row[:dim]])

        block_clusters = [Cluster([point]) for point in block]
        clusters.extend(block_clusters)

        print(f'Processed Block {block_idx}: {len(block)} points')

    # Step 2: Perform hierarchical clustering
    if k is not None and k < len(clusters):
        clusters = hierarchical_clustering(clusters, k)

    return clusters


def cure_cluster(dim: int, k: int, n: int, block_size: int, in_path: str, out_path: str):
    if k is not None:
        clusters = cure_cluster_k(dim, k, n, block_size, in_path)
    else:
        return

    # Step 3: Save final clusters with cluster IDs
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cluster_ID"] + [f"Dim_{i}" for i in range(dim)])  # Header

        for cluster_id, cluster in enumerate(clusters):
            for point in cluster.points:
                writer.writerow([cluster_id] + point)  # Add cluster ID to each point

    print(f'Final clusters saved to {out_path}')


def hierarchical_clustering(clusters: list[Cluster], k: int) -> list[Cluster]:
    """Performs hierarchical clustering until only k clusters remain."""
    while len(clusters) > k:
        min_dist = float('inf')
        merge_idx = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = clusters[i].distance(clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)

        i, j = merge_idx
        merged_cluster = Cluster(clusters[i].points + clusters[j].points)
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(merged_cluster)

    return clusters


if __name__ == '__main__':
    from utils import generate_2d_data, plot_clusters

    generate_2d_data(k=3, points_per_cluster=50, cluster_std=1.5)
    cure_cluster(dim=2, k=3, n=150, block_size=50, in_path='points.csv', out_path='out.csv')
    plot_clusters('points_ground_truth.csv')
    plot_clusters('out.csv')
