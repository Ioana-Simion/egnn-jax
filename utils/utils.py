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
        from qm9.utils import generate_loaders_qm9

        train_loader, val_loader, test_loader = generate_loaders_qm9(args)
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
