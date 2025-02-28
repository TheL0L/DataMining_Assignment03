import csv, random

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

    def generate_random_point(offset):
        return tuple([offset[i] + random.uniform(-10, 10) for i in range(dim)])

    centroids = [generate_random_centroid() for _ in range(k)]
    points = [generate_random_point(random.choice(centroids)) for _ in range(n)]

    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(points)
    
    return centroids, points

def load_points(in_path, n=-1, points=[]):
    """
    Loads N points from a CSV file.

    Args:
        in_path - path for CSV file containing the points
        n (default=-1) - amount of points to load, if -1, then load all available
        points (default=[]) - output list of points
    """
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if n == 0:
                break
            point = []
            for axis in row:
                point.append(int(axis))
            points.append(tuple(point))
            n -= 1

def save_points(clusts, out_path, out_path_tagged):
    """
    Write two CSV files describing the given clusters.

    Args:
        clusts - list of clustsers
        out_path - output path for CSV file containing all points within the clusters
        out_path_tagged - output path for CSV file containing all clusters
    """
    with open(out_path, 'w', newline='') as file:
        writer = csv.writer(file)
        all_points = []
        for cluster in clusts:
            all_points.extend(cluster)
        random.shuffle(all_points)
        writer.writerows(all_points)

    with open(out_path_tagged, 'w', newline='') as file:
        writer = csv.writer(file)
        for k, cluster in enumerate(clusts):
            writer.writerow([*cluster, k])


if __name__ == '__main__':
    ...

