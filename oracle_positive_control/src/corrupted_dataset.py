from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import Dataset


class CorruptedDataset(Dataset):
    """Wrap a training dataset so that a subset of labels are replaced.

    The wrapper exposes the *original* ``targets`` attribute so that server-side
    histograms and InfoOpt constraints are computed from uncorrupted labels.
    Only the value returned by ``__getitem__`` is corrupted.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        corruption_map: Mapping[int, int],
    ) -> None:
        self.base = base_dataset
        self.corruption_map = dict(corruption_map)

        # Expose original targets for histogram computation.
        if hasattr(base_dataset, "targets"):
            targets = base_dataset.targets
            if hasattr(targets, "tolist"):
                self.targets = targets.tolist()
            else:
                self.targets = list(targets)
        else:
            self.targets = [int(base_dataset[i][1]) for i in range(len(base_dataset))]

        self.classes = getattr(base_dataset, "classes", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        x, y = self.base[idx]
        if idx in self.corruption_map:
            y = self.corruption_map[idx]
        return x, int(y)
