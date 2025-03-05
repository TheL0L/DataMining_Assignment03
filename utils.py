import csv
import random
from math import sqrt
from typing import List

import matplotlib.pyplot as plt


def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """Calculates the Euclidean distance between two points."""
    return sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))


def generate_2d_data(
        k: int,
        points_per_cluster: int,
        cluster_std: float,
        points_csv: str = 'points.csv',
        ground_truth_csv: str = 'points_ground_truth.csv'
):
    """
    Generates synthetic 2D data with k clusters and saves two CSV files:

    1. points_csv: Contains only the 2D points (Dim_0 and Dim_1).
    2. ground_truth_csv: Contains the points along with the true Cluster_ID.

    Parameters:
        k (int): Number of clusters.
        points_per_cluster (int): Number of points per cluster.
        cluster_std (float): Standard deviation for the Gaussian noise around each cluster center.
        points_csv (str): File path for CSV containing points only.
        ground_truth_csv (str): File path for CSV containing points with cluster IDs.
    """
    # Define cluster centers randomly in the range [-10, 10] for both dimensions.
    centers = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(k)]

    # Lists to hold the data for the two CSV files.
    points_data = []  # Only x, y coordinates
    ground_truth_data = []  # cluster_id, x, y

    for cluster_id, (center_x, center_y) in enumerate(centers):
        for _ in range(points_per_cluster):
            # Generate each coordinate using a Gaussian distribution around the cluster center.
            x = random.gauss(center_x, cluster_std)
            y = random.gauss(center_y, cluster_std)
            points_data.append([x, y])
            ground_truth_data.append([cluster_id, x, y])

    # Write points only CSV (input for CURE)
    with open(points_csv, 'w', newline='') as f_points:
        writer = csv.writer(f_points)
        # writer.writerow(["Dim_0", "Dim_1"])
        writer.writerows(points_data)

    # Write ground truth CSV (for plotting and evaluation)
    with open(ground_truth_csv, 'w', newline='') as f_gt:
        writer = csv.writer(f_gt)
        writer.writerow(["Cluster_ID", "Dim_0", "Dim_1"])
        writer.writerows(ground_truth_data)

    print(f"Generated {k} clusters with {points_per_cluster} points each.")
    print(f"Points only CSV saved to: {points_csv}")
    print(f"Ground truth CSV saved to: {ground_truth_csv}")


def plot_clusters(csv_file: str):
    """
    Plots 2D clusters from a CSV file output by the CURE algorithm or the data generator.

    Assumes the CSV file has a header with columns:
        Cluster_ID, Dim_0, Dim_1
    """
    clusters = {}

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            cluster_id = int(row[0])
            x = float(row[1])
            y = float(row[2])
            if cluster_id not in clusters:
                clusters[cluster_id] = {"x": [], "y": []}
            clusters[cluster_id]["x"].append(x)
            clusters[cluster_id]["y"].append(y)

    # Plot each cluster with a different color.
    for cid, points in clusters.items():
        plt.scatter(points["x"], points["y"], label=f"Cluster {cid}")

    plt.xlabel("Dim 0")
    plt.ylabel("Dim 1")
    plt.title("Clusters from CURE Algorithm Output")
    plt.legend()
    plt.show()
