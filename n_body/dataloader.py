# Code from https://github.com/gerkone/egnn-jax

from typing import Tuple
from .dataset_nbody import NBodyDataset
from torch.utils.data import DataLoader
import numpy as np


def numpy_collate(batch):
    if isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.vstack(batch)


# Utility function added for fetching datasets
def get_nbody_dataloaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_train = NBodyDataset(partition="train", max_samples=args.max_samples)
    dataset_val = NBodyDataset(partition="val")
    dataset_test = NBodyDataset(partition="test")

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )
    return loader_train, loader_val, loader_test


if __name__ == "__main__":
    """
    from dataset_nbody import NBodyDataset

    dataset_train = NBodyDataset(partition='train')
    dataloader_train = Dataloader(dataset_train)
    for i, (loc, vel, edges) in enumerate(dataset_train):
        print(i)
    """
