import random
from typing import List

from utils import euclidean_distance


class Cluster:
    """Represents a cluster with multiple representative points."""
    def __init__(self, points: List[List[float]]):
        self.points = points
        self.centroid = self.compute_centroid()
        self.representative_points = self.compute_representatives()

    def compute_centroid(self) -> List[float]:
        """Computes the centroid (average position) of the cluster."""
        num_points = len(self.points)
        num_dimensions = len(self.points[0])
        centroid = [0] * num_dimensions

        for point in self.points:
            for i in range(num_dimensions):
                centroid[i] += point[i]

        return [x / num_points for x in centroid]

    def compute_representatives(self, shrink_factor: float = 0.2) -> List[List[float]]:
        """Selects representative points and moves them toward the centroid."""
        num_reps = min(len(self.points), 5)
        representatives = random.sample(self.points, num_reps)

        for i in range(len(representatives)):
            representatives[i] = [
                self.centroid[d] + shrink_factor * (representatives[i][d] - self.centroid[d])
                for d in range(len(self.centroid))
            ]
        return representatives

    def distance(self, other: "Cluster") -> float:
        """Computes the minimum Euclidean distance between representative points of two clusters."""
        return min(euclidean_distance(p1, p2) for p1 in self.representative_points for p2 in other.representative_points)
