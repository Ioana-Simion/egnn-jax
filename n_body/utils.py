# from https://github.com/gerkone/egnn-jax/blob/main/nbody/utils.py

import jax.numpy as jnp
import jraph
from typing import Callable, Dict, List, Tuple
import numpy as np


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = jnp.sum(va, axis=1, keepdims=True), jnp.sum(vb, axis=1, keepdims=True)
    return va


def NbodyGraphTransform(
    n_nodes: int,
    batch_size: int,
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    # charged system is a connected graph
    full_edge_indices = np.array(
        [
            (i + n_nodes * b, j + n_nodes * b)
            for b in range(batch_size)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        ]
    ).T

    def _to_jraph(
        data: List,
    ):
        pos, vel, edge_attribute, charges, targets = data

        pos = jnp.array(pos)
        vel = jnp.array(vel)
        edge_attribute = jnp.array(edge_attribute)
        targets = jnp.array(targets)

        cur_batch = int(pos.shape[0] / n_nodes)

        edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
        rows, cols = edge_indices[0], edge_indices[1]

        #edges = data.dataset.get_edges(batch_size, n_nodes)
        nodes = jnp.ones((pos.shape[0], 1))  # all input nodes are set to 1
        #rows, cols = edges
        loc_dist = jnp.sum((pos[rows] - pos[cols]) ** 2, axis=-1)[:, None]
        vel_attr = get_velocity_attr(pos, vel, rows, cols)
        # TODO velocity is not mandatory make optional
        edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], 1)

        return (nodes, pos, edge_indices, edge_attr), targets

    return _to_jraph
