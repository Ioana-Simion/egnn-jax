# based upon https://github.com/gerkone/egnn-jax/blob/main/nbody/utils.py

import jax.numpy as jnp
from typing import Callable, Dict, List, Tuple
from models.utils import mask_from_edges

def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = jnp.linalg.norm(diff, axis=1, keepdims=True)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = jnp.sum(va, axis=1, keepdims=True), jnp.sum(vb, axis=1, keepdims=True)
    return va


def NbodyBatchTransform(
    n_nodes: int,
    batch_size: int,
    model: str = 'egnn'
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    # charged system is a connected graph
    full_edge_indices = jnp.array(
        [
            (i + n_nodes * b, j + n_nodes * b)
            for b in range(batch_size)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        ]
    ).T

    batched_edge_indices = jnp.stack(
        [jnp.array(
            [
                (i, j)
                for i in range(n_nodes)
                for j in range(n_nodes)
                if i != j
            ]
        ).T 
        for _ in range(batch_size)]
    )  # Shape: (batch_size, 2, n_edges)

    mask_from_edges_batched = mask_from_edges()

    def _to_egnn(
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

        vel_attr = get_velocity_attr(pos, vel, rows, cols)

        #magnitudes = jnp.sqrt(jnp.sum(vel ** 2, axis=1))
        #nodes = jnp.expand_dims(magnitudes, axis=1)

        #nodes = jnp.concatenate((nodes, charges), axis=1)
        nodes = charges

        loc_dist = jnp.expand_dims(jnp.sum((pos[rows] - pos[cols]) ** 2, 1), axis=1)
        edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], axis=1)

        return (nodes, pos, edge_indices, vel, edge_attr), targets
