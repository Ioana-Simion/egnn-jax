# from https://github.com/flooreijkelboom/equivariant-simplicial-mp/blob/main/utils.py

import os
import torch
import random
import numpy as np
import torch.nn as nn
from typing import Tuple
from argparse import Namespace
from torch.utils.data import DataLoader
from n_body import get_nbody_dataloaders
import copy
from torch.nn.utils.rnn import pad_sequence
from qm9.utils import RemoveNumHs

class NodeDistance:
    def __init__(self, normalize=False) -> None:
        self.normalize = normalize

    def __call__(self, data):
        data = copy.copy(data)
        node_com_distances = torch.linalg.vector_norm(data.pos - data.pos.mean(dim=0), dim=-1).view(-1, 1)
        if self.normalize:
            node_com_distances = node_com_distances / node_com_distances.max()
        data.x = torch.cat([data.x, node_com_distances], dim=-1)
        return data
    

def collate_fn(data_list):
    x_list = [d.x for d in data_list]
    x = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    mask = torch.zeros_like(x)
    for i, d in enumerate(data_list):
        mask[i, d.x.size(0):] = -torch.inf

    y = torch.stack([d.y for d in data_list])
    edge_attr_list = [d.edge_attr for d in data_list]
    edge_attr = pad_sequence(edge_attr_list, batch_first=True, padding_value=0.0)
    edge_mask = torch.zeros_like(edge_attr)
    for i, d in enumerate(data_list):
        edge_mask[i, d.edge_attr.size(0):] = -torch.inf
    
    pos = [d.pos for d in data_list]

    return x, edge_attr, pos, mask, edge_mask, y


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
        dataset = QM9(root='data/QM9', pre_transform=T.Compose([T.Distance(), RemoveNumHs(), NodeDistance(normalize=True)]))
        num_train = 100000
        num_val = 10000

        train_loader = DataLoader(dataset[:num_train], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset[num_train:num_train+num_val], batch_size=args.batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(dataset[num_train+num_val:], batch_size=args.batch_size, collate_fn=collate_fn)
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
