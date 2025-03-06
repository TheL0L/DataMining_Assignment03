from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
from collections import defaultdict
from typing import List, Tuple


def plot_from_memory_2d(clusters: List[List[Tuple[float, ...]]]) -> None:
    """
    Plots 2D clusters from memory, using a unique color for each cluster.

    :param clusters: A list of clusters, where each cluster is a list of (x, y) tuples.
    """
    colors = plt.cm.get_cmap("tab20", len(clusters))  # Get distinct colors
    
    plt.figure(figsize=(8, 6))
    
    for i, cluster in enumerate(clusters):
        x, y = zip(*cluster)  # Unpack points
        plt.scatter(x, y, color=colors(i), label=f"Cluster {i+1}")
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Cluster Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_from_memory_3d(clusters: List[List[Tuple[float, ...]]]) -> None:
    """
    Plots 3D clusters from memory, using a unique color for each cluster.

    :param clusters: A list of clusters, where each cluster is a list of (x, y, z) tuples.
    """
    colors = plt.cm.get_cmap("tab20", len(clusters))  # Get distinct colors
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    for i, cluster in enumerate(clusters):
        x, y, z = zip(*cluster)  # Unpack points
        ax.scatter(x, y, z, color=colors(i), label=f"Cluster {i+1}")
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Cluster Plot")
    ax.legend()
    plt.show()

    
def plot_from_storage_2d(clusters_file: str, chunk_size: int = 10_000) -> None:
    """
    Reads a large CSV file in chunks and plots 2D clusters.

    :param clusters_file: Path to the CSV file containing 2D cluster data.
    :param chunk_size: Number of rows to read per chunk.
    """
    clusters = defaultdict(list)

    with open(clusters_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            *point, cluster_idx = row
            clusters[int(cluster_idx)].append(tuple(map(float, point[:2])))  # Only take x, y for 2D

    colors = plt.cm.get_cmap("tab20", len(clusters))  # Get distinct colors

    plt.figure(figsize=(8, 6))
    for i, (cluster_idx, points) in enumerate(clusters.items()):
        x, y = zip(*points)
        plt.scatter(x, y, color=colors(i), label=f"Cluster {cluster_idx}")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Cluster Plot")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_from_storage_3d(clusters_file: str, chunk_size: int = 10_000) -> None:
    """
    Reads a large CSV file in chunks and plots 3D clusters.

    :param clusters_file: Path to the CSV file containing 3D cluster data.
    :param chunk_size: Number of rows to read per chunk.
    """
    clusters = defaultdict(list)

    with open(clusters_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            *point, cluster_idx = row
            clusters[int(cluster_idx)].append(tuple(map(float, point[:3])))  # Only take x, y, z for 3D

    colors = plt.cm.get_cmap("tab20", len(clusters))  # Get distinct colors

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, (cluster_idx, points) in enumerate(clusters.items()):
        x, y, z = zip(*points)
        ax.scatter(x, y, z, color=colors(i), label=f"Cluster {cluster_idx}")

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Cluster Plot")
    ax.legend()
    plt.show()

