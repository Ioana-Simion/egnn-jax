# from https://github.com/flooreijkelboom/equivariant-simplicial-mp/blob/main/utils.py

import os
import torch
import random
import numpy as np
import torch.nn as nn
from typing import Tuple
from qm9.utils import RemoveNumHs
import copy
from torch.utils.data import DataLoader
from n_body import get_nbody_dataloaders
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.loader import DataLoader as GDataLoader
import torch.nn.functional as F
from argparse import Namespace


class NodeDistance:
    def __init__(self, normalize=False) -> None:
        self.normalize = normalize

    def __call__(self, data):
        data = copy.copy(data)
        node_com_distances = torch.linalg.vector_norm(
            data.pos - data.pos.mean(dim=0), dim=-1
        ).view(-1, 1)
        if self.normalize:
            node_com_distances = node_com_distances / node_com_distances.max()
        data.x = torch.cat([data.x, node_com_distances], dim=-1)
        return data


def collate_fn(data_list):
    x_list = [d.x for d in data_list]
    x = pad_sequence(x_list, batch_first=True, padding_value=0.0)
    mask = torch.zeros_like(x)
    for i, d in enumerate(data_list):
        mask[i, d.x.size(0) :] = -torch.inf

    y = torch.stack([d.y for d in data_list])
    edge_attr_list = [d.edge_attr for d in data_list]
    edge_attr = pad_sequence(edge_attr_list, batch_first=True, padding_value=0.0)
    edge_mask = torch.zeros_like(edge_attr)
    for i, d in enumerate(data_list):
        edge_mask[i, d.edge_attr.size(0) :] = -torch.inf

    pos = [d.pos for d in data_list]

    return x, edge_attr, pos, mask, edge_mask, y

def get_collate_fn_egnn(dataset):
    print("calculating max num nodes and edges")
    max_num_nodes = max([len(x.x) for x in dataset])
    max_num_edges = max([x.edge_index.shape[-1] for x in dataset])
    print("max num nodes", max_num_nodes)
    print("max num edges", max_num_edges)

    def _collate_fn_egnn(data_list):
        x = [d.x for d in data_list]
        x[0] = nn.ConstantPad1d((0, max_num_nodes - x[0].size(0)), 0.0)(x[0])
        x = pad_sequence(x, batch_first=True, padding_value=0.0).reshape(x.shape[0] * x.shape[1], -1)

        y = torch.stack([d.y for d in data_list])

        edge_index = []
        start_idx = 0
        for d in data_list:
            num_edges = d.edge_index.size(1)
            padded_edges = torch.cat([d.edge_index, torch.full((2, max_num_edges - num_edges), -1)], dim=1)
            padded_edges = torch.where(padded_edges != -1, padded_edges + start_idx, padded_edges)
            edge_index.append(padded_edges)
            start_idx += num_edges
        edge_index = torch.stack(edge_index).transpose(0, 1).reshape(2, -1)

        edge_attr = [d.edge_attr for d in data_list]
        edge_attr[0] = nn.ConstantPad1d((0, max_num_edges - edge_attr[0].size(0)), 0.0)(edge_attr[0])
        edge_attr = pad_sequence(edge_attr, batch_first=True, padding_value=0.0).reshape(edge_attr.shape[0] * edge_attr.shape[1], -1)
        
        pos = [d.pos for d in data_list]
        pos[0] = nn.ConstantPad1d((0, max_num_nodes - pos[0].size(0)), 0.0)(pos[0])
        pos = pad_sequence(pos, batch_first=True, padding_value=0.0).reshape(pos.shape[0] * pos.shape[1], -1)
        
        #node_mask = torch.where(x.sum(dim=-1) == 0, 1, 0)
        #edge_mask = torch.where(edge_attr.sum(dim=-1) == 0, 1, 0)
        return x, edge_attr, edge_index, pos, y
    return _collate_fn_egnn

def get_model(args: Namespace) -> nn.Module:
    """Return model based on name."""
    if args.dataset == "qm9":
        num_out = 1
    elif args.dataset == "charged":
        num_out = 3
    else:
        raise ValueError(f"Do not recognize dataset {args.dataset}.")

    if args.model_name == 'egnn':
        from models.egnn_jax import EGNN

        model = EGNN(
            hidden_nf=args.num_hidden,
            out_node_nf=num_out,
            n_layers=args.num_layers,
        )
    elif args.model_name == 'transformer':
        from models.transformer import EGNNTransformer

        model = EGNNTransformer(
            num_edge_encoder_blocks=args.num_edge_encoders,
            num_node_encoder_blocks=args.num_node_encoders,
            num_combined_encoder_blocks= args.num_combined_encoder_blocks,

            model_dim=args.dim,
            num_heads=args.heads,
            dropout_prob=args.dropout,
            edge_input_dim= args.edge_input_dim,
            node_input_dim= args.node_input_dim,
        )
    else:
        raise ValueError(f"Model type {args.model_name} not recognized.")

    return model


def get_loaders(
    args: Namespace, transformer=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return dataloaders based on dataset."""
    if args.dataset == "qm9":
        from torch_geometric.datasets import QM9
        import torch_geometric.transforms as T

        num_train = 10000
        num_val = 1000
        num_test= 1000

        if transformer:
            # Distance transform handles distances between atoms
            dataset = QM9(root='data/QM9', pre_transform=T.Compose([T.Distance(), RemoveNumHs(), NodeDistance(normalize=True)]))
            
            train_loader = DataLoader(dataset[:num_train], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(dataset[num_train:num_train+num_val], batch_size=args.batch_size, collate_fn=collate_fn)
            test_loader = DataLoader(dataset[num_train+num_val:num_train+num_val+num_test], batch_size=args.batch_size, collate_fn=collate_fn)

        else:
            dataset = QM9(root='data/QM9', pre_transform=RemoveNumHs())
            collate_fn_egnn = get_collate_fn_egnn(dataset)

            train_loader = DataLoader(
                dataset[:num_train],
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_fn_egnn,
            )
            val_loader = DataLoader(
                dataset[num_train : num_train + num_val],
                batch_size=args.batch_size,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_fn_egnn,
            )
            test_loader = DataLoader(
                dataset[num_train + num_val :],
                batch_size=args.batch_size,
                drop_last=True,
                pin_memory=True,
                collate_fn=collate_fn_egnn,
            )
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
