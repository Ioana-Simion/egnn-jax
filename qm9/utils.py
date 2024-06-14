import copy
import jax.numpy as jnp
import numpy as np  # Make sure to import numpy
from typing import Callable
import torch


def GraphTransform(property_idx) -> Callable:
    """
    Build a function that converts torch geometric DataBatch into jraph.GraphsTuple.
    Args:
    property_idx: Index of the property to predict.
    
    Returns:
    h : the one hot encoding of atomic numbers (dataset.x)
    pos : Atoms positions (dataset.pos)
    edge_indices : Edge indices (dataset.edge_index)
    edge_attr: Edge attributes (dataset.edge_attr)
    targets: Selected target property (dataset.y[:, property_idx])
    """
    def _to_jax(data):
        jax_data = {key: jnp.array(np.array(value)) for key, value in data.items() if torch.is_tensor(value)}
        
        nodes = jax_data['x']
        pos = jax_data['pos']
        edge_indices = jax_data['edge_index']
        edge_attr = jax_data['edge_attr'] if 'edge_attr' in jax_data else None
        targets = jax_data['y'][:, property_idx]  # Select property to optimize for
        targets = targets.reshape(-1,1)

        return (nodes, pos, edge_indices, edge_attr), targets
    
    return _to_jax


def TransformDLBatches(property_idx):
    """
    Build a function that converts torch geometric DataBatch into jraph.GraphsTuple.
    Args:
    property_idx: Index of the property to predict.
    
    Returns:
    h : the one hot encoding of atomic numbers (dataset.x)
    pos : Atoms positions (dataset.pos)
    edge_indices : Edge indices (dataset.edge_index)
    edge_attr: Edge attributes (dataset.edge_attr)
    targets: Selected target property (dataset.y[:, property_idx])
    """
    def _to_jax(data):

        data = (jnp.array(np.array(x)) for x in data)
        nodes, edge_attr, edge_index, pos, targets, node_mask, edge_mask = data
    
        node_mask = (nodes.sum(axis=1) != 0).astype(jnp.float32)
        targets = targets[:, property_idx]  # Select property to optimize for
        #targets = targets.reshape(-1,1)
        pos = pos.reshape(-1,3)
        node_mask = node_mask.reshape(-1, 1)

        return (nodes, pos, edge_index, edge_attr, node_mask, edge_mask), targets
    
    return _to_jax


class RemoveNumHs:
    def __call__(self, data):
        data = copy.copy(data)
        data.x = data.x[:, :-1]
        return data
