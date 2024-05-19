# from https://github.com/flooreijkelboom/equivariant-simplicial-mp/blob/main/utils.py

import os
import torch
import random
import numpy as np
import torch.nn as nn
from typing import Tuple
from argparse import Namespace
from torch_geometric.loader import DataLoader
from n_body import get_nbody_dataloaders
import copy


class NodeDistance:
    def __init__(self, normalize=False) -> None:
        self.normalize = normalize

    def __call__(self, data):
        data = copy.copy(data)
        node_com_distances = torch.linalg.vector_norm(data.pos - data.pos.mean(dim=0), dim=-1)
        if self.normalize:
            node_com_distances = node_com_distances / node_com_distances.max()
        data.x = torch.cat([data.x, node_com_distances], dim=-1)
        return data


def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == "qm9":
        num_out = 1
    elif args.dataset == "charged":
        num_out = 3
    else:
        raise ValueError(f"Do not recognize dataset {args.dataset}.")

    if args.model_name == "egnn":
        from models.egnn_jax import EGNN

        model = EGNN(
            hidden_nf=args.num_hidden,
            out_node_nf=num_out,
            n_layers=args.num_layers,
        )

    else:
        raise ValueError(f"Model type {args.model_name} not recognized.")

    return model


def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == "qm9":
        from torch_geometric.datasets import QM9
        import torch_geometric.transforms as T
        # Distance transform handles distances between atoms
        dataset = QM9(root='data/QM9', pre_transform=T.Compose([T.Distance(), NodeDistance(normalize=True)]))
        num_train = 100000
        num_val = 10000

        train_loader = DataLoader(dataset[:num_train], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[num_train:num_train+num_val], batch_size=args.batch_size)
        test_loader = DataLoader(dataset[num_train+num_val:], batch_size=args.batch_size)
    elif args.dataset == "charged":
        train_loader, val_loader, test_loader = get_nbody_dataloaders(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

    return train_loader, val_loader, test_loader


def set_seed(seed: int = 42) -> None:

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
