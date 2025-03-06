import small_data, large_data
from hierarchical_clustering import h_clustering as hac
from k_means import k_means
from bfr import bfr_cluster, bfr_cleanup
from cure import cure_cluster

from random import seed
import hashlib, os

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

def generate_small_data_files():
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

def generate_large_data_files():
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


if __name__ == '__main__':
    seed('big data')  # initialize the rng seed for reproducibility

    small_files = generate_small_data_files()
    large_files = generate_large_data_files()

