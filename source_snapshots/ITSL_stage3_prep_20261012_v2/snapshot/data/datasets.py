from __future__ import annotations
import csv
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

CIFAR_MEAN = {
    10: (0.4914, 0.4822, 0.4465),
    100: (0.5071, 0.4865, 0.4409),
}
CIFAR_STD = {
    10: (0.2470, 0.2435, 0.2616),
    100: (0.2673, 0.2564, 0.2762),
}


def _cifar_transforms(num_classes: int):
    mean = CIFAR_MEAN[num_classes]
    std = CIFAR_STD[num_classes]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def get_cifar(num_classes: int, root: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_tf, test_tf = _cifar_transforms(num_classes)
    if num_classes == 10:
        train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    elif num_classes == 100:
        train = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unsupported CIFAR variant: {num_classes}")
    return train, test


class _MedMNIST(torch.utils.data.TensorDataset):
    def __init__(self, images, labels, transform=None, targets_list=None):
        self.transform = transform
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        labels_np = labels.squeeze()
        super().__init__(images, torch.from_numpy(labels_np).long())
        self.targets = labels_np.tolist()

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        return img, label


class _PILTransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class _NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.targets = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class _NumpyAugment:
    def __init__(self, mean, std, augment=True, seed=42):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.augment and self.rng.random() < 0.5:
            img = transforms.functional.hflip(img)
        if self.augment:
            angle = self.rng.uniform(-15, 15)
            img = transforms.functional.rotate(img, angle)
        return (img - self.mean) / self.std


class _CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, base_dir, class_to_idx, transform=None):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.samples = []
        self._labels = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["Filename"]
                label = int(row["Label"])
                img_path = self.base_dir / fname
                self.samples.append((str(img_path), label))
                self._labels.append(label)

        csv_labels = set(self._labels)
        folder_labels = set(class_to_idx.values())
        if csv_labels != folder_labels:
            raise AssertionError(
                f"CSV label set {sorted(csv_labels)} does not match "
                f"ImageFolder label set {sorted(folder_labels)}. "
                f"CSV encoding may be scrambled relative to directory labels."
            )

        self.targets = self._labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_pathmnist(root: str, expected_num_classes: int | None = None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    npz_path = Path(root) / "pathmnist" / "pathmnist.npz"
    npz_path_alt = Path(root) / "pathmnist.npz"

    if npz_path.exists():
        data = np.load(npz_path)
    elif npz_path_alt.exists():
        data = np.load(npz_path_alt)
    else:
        raise FileNotFoundError(
            f"PathMNIST npz not found at {npz_path} or {npz_path_alt}. "
            "Download from: https://zenodo.org/records/10519652/files/pathmnist.npz "
            "Place at datasets/pathmnist/pathmnist.npz"
        )

    train_images = data["train_images"]
    train_labels = data["train_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    actual_min = int(train_labels.min())
    actual_max = int(train_labels.max())
    actual_classes = actual_max + 1
    if expected_num_classes is not None:
        if actual_max >= expected_num_classes or actual_min < 0:
            raise AssertionError(
                f"PathMNIST label range [{actual_min}, {actual_max}] does not match "
                f"expected num_classes={expected_num_classes}. Check dataset root and config."
            )
        if actual_classes < expected_num_classes:
            raise AssertionError(
                f"PathMNIST has {actual_classes} unique labels (0-{actual_max}) but "
                f"config specifies num_classes={expected_num_classes}. "
                f"Config underestimates dataset classes."
            )

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    test_tf = transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train = _MedMNIST(train_images, train_labels, transform=train_tf)
    test = _MedMNIST(test_images, test_labels, transform=test_tf)
    return train, test


def get_eurosat(root: str, expected_num_classes: int | None = None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    mean = (0.3610, 0.3714, 0.3534)
    std = (0.1908, 0.1782, 0.1859)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    csv_path = Path(root) / "train.csv"
    test_csv = Path(root) / "test.csv"

    if csv_path.exists() and test_csv.exists():
        base_ds = datasets.ImageFolder(str(root))
        train_ds = _CSVDataset(csv_path, root, base_ds.class_to_idx, train_tf)
        test_ds = _CSVDataset(test_csv, root, base_ds.class_to_idx, test_tf)
        actual_max = max(train_ds.targets)
        actual_min = min(train_ds.targets)
        actual_classes = actual_max + 1
        if expected_num_classes is not None:
            if actual_max >= expected_num_classes or actual_min < 0:
                raise AssertionError(
                    f"EuroSAT label range [{actual_min}, {actual_max}] does not match "
                    f"expected num_classes={expected_num_classes}. Check dataset root and config."
                )
            if actual_classes < expected_num_classes:
                raise AssertionError(
                    f"EuroSAT has {actual_classes} unique labels (0-{actual_max}) but "
                    f"config specifies num_classes={expected_num_classes}. "
                    f"Config underestimates dataset classes."
                )
        return train_ds, test_ds

    raise FileNotFoundError(
        f"EuroSAT CSV files not found at {csv_path} / {test_csv}. "
        "Place the dataset as ImageFolder with train.csv and test.csv in the root."
    )


def get_femnist(root: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load LEAF FEMNIST with natural writer-based splits.

    Expects leaf-preprocessed data at {root}/femnist/.
    If not found, falls back to torchvision EMNIST 'byclass' with a note.
    """
    leaf_dir = Path(root) / "femnist"
    leaf_json = leaf_dir / "all_data.json"

    if leaf_json.exists():
        return _get_femnist_leaf(str(leaf_json))

    # Fallback: EMNIST byclass (not true writer partitions, but same class set)
    print(f"[FEMNIST] LEAF data not found at {leaf_dir}. Using EMNIST byclass fallback.")
    print(f"[FEMNIST] For natural writer partitions, download LEAF FEMNIST from:")
    print(f"[FEMNIST] https://github.com/karpathy/leaf/data/femnist")
    return _get_femnist_emnist_fallback(root)


def _get_femnist_leaf(json_path: str):
    import json
    with open(json_path) as f:
        data = json.load(f)

    users = data["users"]
    num_samples = data["num_samples"]
    user_data = data["user_data"]

    all_images = []
    all_labels = []
    all_writers = []  # writer ID per sample

    for writer_id, n_samples in zip(users, num_samples):
        xs = user_data[writer_id]["x"]
        ys = user_data[writer_id]["y"]
        for x, y in zip(xs, ys):
            # x is a 784-length flattened 28x28 grayscale image
            img = np.array(x, dtype=np.float32).reshape(1, 28, 28)  # CHW
            all_images.append(img)
            all_labels.append(int(y))
            all_writers.append(writer_id)

    images = np.stack(all_images)  # (N, 1, 28, 28)
    labels = np.array(all_labels, dtype=int)

    # LEAF FEMNIST uses first 80% of writers for train, last 20% for test
    # But we'll use a random split for consistency
    rng = np.random.default_rng(42)
    n_total = len(labels)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    n_train = int(0.8 * n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_ds = _NumpyDataset(images[train_idx], labels[train_idx], transform=None)
    test_ds = _NumpyDataset(images[test_idx], labels[test_idx], transform=None)

    # Store writer IDs as attributes for partitioning
    train_ds.writers = [all_writers[i] for i in train_idx]
    test_ds.writers = [all_writers[i] for i in test_idx]

    return train_ds, test_ds


def _get_femnist_emnist_fallback(root: str):
    """Fallback using torchvision EMNIST byclass. Not true writer partitions."""
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    train = datasets.EMNIST(root=root, split="byclass", train=True, download=False, transform=tf)
    test = datasets.EMNIST(root=root, split="byclass", train=False, download=False, transform=tf)
    return train, test


def get_dataset(name: str, root: str, expected_num_classes: int | None = None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if name == "CIFAR10":
        return get_cifar(10, root)
    elif name == "CIFAR100":
        return get_cifar(100, root)
    elif name == "PathMNIST":
        return get_pathmnist(root, expected_num_classes)
    elif name == "EuroSAT":
        return get_eurosat(root, expected_num_classes)
    elif name == "FEMNIST":
        return get_femnist(root)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: CIFAR10, CIFAR100, PathMNIST, EuroSAT, FEMNIST"
        )


def get_labels(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, list):
            return targets
        if hasattr(targets, "tolist"):
            return targets.tolist()
        return list(targets)
    return [int(dataset[i][1]) for i in range(len(dataset))]