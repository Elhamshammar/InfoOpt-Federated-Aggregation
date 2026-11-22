from __future__ import annotations
import numpy as np
from typing import Dict, Sequence


def dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float, min_samples_per_client: int = 10, seed: int = 42) -> Dict[int, list[int]]:
    """Create a Dirichlet non-IID partition by class proportions."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    for _ in range(100):
        client_indices = {i: [] for i in range(num_clients)}
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            splits = np.split(idx_c, cuts)
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
        sizes = [len(v) for v in client_indices.values()]
        if min(sizes) >= min_samples_per_client:
            for i in client_indices:
                rng.shuffle(client_indices[i])
            return client_indices
    raise RuntimeError("Could not create a valid partition. Increase alpha or reduce min_samples_per_client.")


def class_histogram(labels: Sequence[int], indices: Sequence[int], num_classes: int) -> np.ndarray:
    y = np.asarray(labels)[list(indices)]
    return np.bincount(y, minlength=num_classes).astype(float)


def writer_partition(dataset, num_clients: int, min_samples_per_client: int = 10, seed: int = 42) -> Dict[int, list[int]]:
    """Partition a dataset with writer IDs into client groups.

    Expects dataset.writers to be a list of writer identifiers.
    Groups writers into num_clients clusters, then collects their samples.
    """
    rng = np.random.default_rng(seed)
    writers = dataset.writers
    unique_writers = sorted(set(writers))
    rng.shuffle(unique_writers)

    # Assign writers to clients
    writer_to_client = {}
    writers_per_client = max(1, len(unique_writers) // num_clients)
    for i, writer in enumerate(unique_writers):
        client_id = min(i // writers_per_client, num_clients - 1)
        writer_to_client[writer] = client_id

    # Collect sample indices per client
    client_indices = {i: [] for i in range(num_clients)}
    for idx, writer in enumerate(writers):
        cid = writer_to_client[writer]
        client_indices[cid].append(idx)

    # Filter clients with too few samples
    valid_clients = {k: v for k, v in client_indices.items() if len(v) >= min_samples_per_client}
    if len(valid_clients) < num_clients:
        print(f"[writer_partition] Only {len(valid_clients)}/{num_clients} clients meet min_samples={min_samples_per_client}")

    for i in valid_clients:
        rng.shuffle(valid_clients[i])
    return valid_clients


def pathological_partition(labels: Sequence[int], num_clients: int, classes_per_client: int, seed: int = 42) -> Dict[int, list[int]]:
    """Assign each client a fixed number of classes (pathological non-IID).

    Each client gets exactly `classes_per_client` random classes.
    All samples of a class are divided evenly among the clients assigned that class.

    Algorithm:
      1. Shuffle class list.
      2. Assign classes_per_client classes to each client (round-robin or random).
      3. For each class, collect all its samples and split among its assigned clients.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max()) + 1

    # Step 1: Assign classes to clients
    # Each client gets classes_per_client random classes
    all_classes = np.arange(num_classes)
    rng.shuffle(all_classes)

    # Create a class -> list of clients mapping
    class_to_clients = {c: [] for c in range(num_classes)}

    # Assign classes to clients sequentially, then shuffle assignments
    class_pool = list(all_classes) * ((num_clients * classes_per_client // num_classes) + 1)
    rng.shuffle(class_pool)

    client_to_classes = {}
    for i in range(num_clients):
        assigned = []
        while len(assigned) < classes_per_client and class_pool:
            c = class_pool.pop()
            if c not in assigned:
                assigned.append(c)
                class_to_clients[c].append(i)
        client_to_classes[i] = assigned

    # Step 2: Distribute samples of each class among its assigned clients
    client_indices = {i: [] for i in range(num_clients)}
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        owners = class_to_clients[c]
        if len(owners) == 0:
            continue
        splits = np.array_split(idx_c, len(owners))
        for owner, split in zip(owners, splits):
            client_indices[owner].extend(split.tolist())

    for i in client_indices:
        rng.shuffle(client_indices[i])
    return client_indices
