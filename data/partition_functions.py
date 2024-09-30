import math

import numpy as np


# Create Paritions
def split_art_data_noniid(n_partitions: int, train_data: tuple[list, list]) -> list:
    """
    Splitting only 1-2-3
    Non-indetical chunks
    """
    if n_partitions <= 1:
        raise ValueError(f"Can not create {n_partitions}, must be bigger than 2")

    x_train, y_train = train_data

    partition_size = math.floor(len(x_train) / n_partitions)
    partitions = []
    for cid in range(n_partitions):
        idx_from, idx_to = cid * partition_size, (cid + 1) * partition_size
        partitions.append((x_train[idx_from:idx_to], y_train[idx_from:idx_to]))

    return partitions


def split_art_data_iid(n_partitions: int, train_data: tuple[list, list]) -> list:
    """
    Splitting (1-2-3) - (1-2-3) - ... - (1-2-3)
    Identical chunks, with identical distribution
    """
    if n_partitions <= 1:
        raise ValueError(f"Can not create {n_partitions}, must be bigger than 2")

    results = [[] for i in range(n_partitions)]
    labels = [[] for i in range(n_partitions)]

    x_split, y_split = train_data

    for i in range(0, len(x_split) - n_partitions, n_partitions):
        for partition in range(n_partitions):
            results[partition].append(x_split[i + partition])
            labels[partition].append(y_split[i + partition])

    partitions = [(np.array(x), np.array(y)) for x, y in zip(results, labels)]
    return partitions
