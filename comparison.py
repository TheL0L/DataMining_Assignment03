import small_data, large_data
from hierarchical_clustering import h_clustering as hac
from k_means import k_means
from bfr import bfr_cluster, bfr_cleanup
from cure import cure_cluster

from random import seed
import hashlib, os, csv
from small_data import read_single_point
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_file_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_file(file_path: str, expected_hash: str) -> bool:
    if not os.path.exists(file_path):
        return False
    file_hash = get_file_sha256(file_path)
    return file_hash == expected_hash.lower()

def generate_small_data_files() -> List[str]:
    # file 1 requirements: dim=2 k>=5 n≈1000
    # file 2 requirements: dim>3 k>4 dim+k>=10
    file_names = [
        './data/small_1.csv',
        './data/small_2a.csv',
        './data/small_2b.csv',
        './data/small_2c.csv'
    ]
    checksums = [
        '2E023461402672B9CD234242C87A8E3CD973243EE92090D906302A61B71C6E38',
        '9E370A848B6A4DCAF9CDA692E901FCC031C83112BF2181655B6441BB1F62DBD4',
        '45357C720FD3E818B3AE98171538EFA167E62C78132D189EBE9DDEA876A10C08',
        '0CE683C251AFE737D032472C974DC6AA6E78511619D52EA851C519D0F4574178'
    ]

    if not verify_file(file_names[0], checksums[0]):
        small_data.generate_data(
            dim=2, k=5, n=1_000,
            out_path=file_names[0],
            points_gen=None, extras=None
        )

    if not verify_file(file_names[1], checksums[1]):
        small_data.generate_data(
            dim=5, k=10, n=10_000,
            out_path=file_names[1],
            points_gen=None, extras=None
        )

    if not verify_file(file_names[2], checksums[2]):
        small_data.generate_data(
            dim=6, k=15, n=25_000,
            out_path=file_names[2],
            points_gen=None, extras=None
        )

    if not verify_file(file_names[3], checksums[3]):
        small_data.generate_data(
            dim=10, k=20, n=50_000,
            out_path=file_names[3],
            points_gen=None, extras=None
        )

    return file_names

def generate_large_data_files() -> List[str]:
    # file 1,2 requirements: at least 10GB, dim>5 k>5
    file_names = [
        './data/large_1.csv',
        './data/large_2.csv'
    ]
    checksums = [
        'D834C4D287EA2F99136E9672C81D0FAF1D5CEFBEDD94A518279417709B912EE6',
        '65B27B92C0B91A173E16ADBF7DFC7DE86C1131FF532F68B3609F5277299CEB78'
    ]

    if not verify_file(file_names[0], checksums[0]):
        large_data.generate_data(
            dim=75, k=75, n=10**7,
            out_path=file_names[0],
            points_gen=None, extras=None
        )

    if not verify_file(file_names[1], checksums[1]):
        large_data.generate_data(
            dim=100, k=100, n=10**7,
            out_path=file_names[1],
            points_gen=None, extras=None
        )

    return file_names


def construct_clustering(file_path: str) -> List[List[Tuple[float, ...]]]:
    """
    Strong assumption: there is enough memory to construct the clustering.
    """
    clusters = dict()
    input_handler = open(file_path, 'r')
    reader = csv.reader(input_handler)

    for row in reader:
        point = read_single_point(row)
        k = int(point[-1])
        point = tuple(point[:-1])

        if k not in clusters:
            clusters[k] = []
        clusters[k].append(point)

    input_handler.close()
    return list(clusters.values())

def compare_clusterings(
        actual: List[List[Tuple[float, ...]]],
        predicted: List[List[Tuple[float, ...]]]) -> float:
    """
    Compare two clusterings based on how similar `predicted` is to `actual`.

    This function constructs a contingency table for the clusters by mapping each point
    to its cluster label (using the index of the cluster in the list) for both the actual
    and predicted clusterings. It then uses the Hungarian algorithm to find the optimal
    one-to-one mapping between the clusters. The clustering accuracy is computed as the
    sum of counts along the optimally matched clusters divided by the total number of points.
    """
    # Map each point to its cluster label for both clusterings.
    actual_label = {}
    for cluster_idx, cluster in enumerate(actual):
        for point in cluster:
            actual_label[point] = cluster_idx

    predicted_label = {}
    for cluster_idx, cluster in enumerate(predicted):
        for point in cluster:
            predicted_label[point] = cluster_idx

    # Consider only the common points in both clusterings.
    all_points = set(actual_label.keys()) & set(predicted_label.keys())
    total_points = len(all_points)
    if total_points == 0:
        return 0.0

    # Get sorted lists of unique cluster labels.
    actual_clusters = sorted({actual_label[p] for p in all_points})
    predicted_clusters = sorted({predicted_label[p] for p in all_points})
    n_actual = len(actual_clusters)
    n_predicted = len(predicted_clusters)

    # Map the cluster labels to indices in our contingency matrix.
    actual_index = {cluster: i for i, cluster in enumerate(actual_clusters)}
    predicted_index = {cluster: j for j, cluster in enumerate(predicted_clusters)}

    # Build the contingency matrix.
    contingency = np.zeros((n_actual, n_predicted), dtype=int)
    for point in all_points:
        i = actual_index[actual_label[point]]
        j = predicted_index[predicted_label[point]]
        contingency[i, j] += 1

    # Use the Hungarian algorithm (via linear_sum_assignment) to find the optimal cluster matching.
    # We negate the contingency matrix to convert our maximization problem into a minimization problem.
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched_count = contingency[row_ind, col_ind].sum()

    # Calculate and return the accuracy.
    accuracy = matched_count / total_points
    return accuracy


if __name__ == '__main__':
    seed('big data')  # initialize the rng seed for reproducibility

    small_files = generate_small_data_files()
    large_files = generate_large_data_files()

    # ☐ show heuristic/metric for a range of k values for small_data.algs
    # ☑ show clustering accuracy for small_data.algs
    # ☑ show clustering accuracy for large_data.algs

