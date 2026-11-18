"""PneumoniaMNIST loading, augmentation, and stratified subsampling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

MEAN, STD = 0.5, 0.5


def build_transforms(train: bool):
    # Grayscale X-ray -> 3 channels so ImageNet ResNet weights can be reused.
    aug = []
    if train:
        # No horizontal flip: chest anatomy is asymmetric.
        aug = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    return transforms.Compose(
        aug
        + [
            transforms.ToTensor(),
            transforms.Normalize([MEAN], [STD]),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )


def load_split(split: str, root: str = "data", size: int = 128, train_aug: bool | None = None):
    from medmnist import PneumoniaMNIST

    Path(root).mkdir(parents=True, exist_ok=True)
    aug = (split == "train") if train_aug is None else train_aug
    return PneumoniaMNIST(split=split, root=root, size=size, download=True, transform=build_transforms(aug))


def stratified_fraction(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Indices of a class-stratified subset containing `frac` of the data (at least 1 per class)."""
    if frac >= 1.0:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        k = max(1, int(round(frac * len(idx))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int = 2, indices=None):
    if indices is not None:
        ds = Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    for s in ("train", "val", "test"):
        ds = load_split(s)
        y = ds.labels.ravel()
        print(f"{s:5s} n={len(y):5d}  pneumonia={y.mean():.1%}")
