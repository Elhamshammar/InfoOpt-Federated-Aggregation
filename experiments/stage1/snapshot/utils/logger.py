from __future__ import annotations
import csv
import json
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


def save_json(obj: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2)
