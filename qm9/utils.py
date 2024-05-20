import jax.numpy as jnp
from typing import Callable
import jraph
import torch


def GraphTransform(batch_size: int) -> Callable:
    """
    Build a function that converts torch geometric DataBatch into jraph.GraphsTuple.
    Returns:
    h : the one hot encoding of atomic numbers (dataset.x)
    pos : Atoms positions (dataset.pos)
    edge_indices : Edge indices (dataset.edge_index)
    edge_attr: Edge attributes (dataset.edge_attr)
    targets: Target properties (dataset.y)
    """
    def _to_jax(data):
        jax_data = {key: jnp.array(value.numpy()) for key, value in data.items() if torch.is_tensor(value)}
        
        nodes = jax_data['x']
        pos = jax_data['pos']
        edge_indices = jax_data['edge_index']
        edge_attr = jax_data['edge_attr'] if 'edge_attr' in jax_data else None
        targets = jax_data['y']
        
        return (nodes, pos, edge_indices, edge_attr), targets
    
    return _to_jax

