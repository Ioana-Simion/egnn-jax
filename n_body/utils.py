# based upon https://github.com/gerkone/egnn-jax/blob/main/nbody/utils.py

import jax.numpy as jnp
from typing import Callable, Dict, List, Tuple


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
    model: str = 'egnn'
) -> Callable:
    """
    Build a function that converts torch DataBatch into jraph.GraphsTuple.
    """

    # charged system is a connected graph
    #TODO check out if jnp or np
    full_edge_indices = jnp.array(
        [
            (i + n_nodes * b, j + n_nodes * b)
            for b in range(batch_size)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if i != j
        ]
    ).T

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

        # nodes = jnp.ones((pos.shape[0], 1))  # all input nodes are set to 1
        # loc_dist = jnp.sum((pos[rows] - pos[cols]) ** 2, axis=-1)[:, None]
        vel_attr = get_velocity_attr(pos, vel, rows, cols)
        # edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], 1)

        magnitudes = jnp.sqrt(jnp.sum(vel ** 2, axis=1))
        nodes = jnp.expand_dims(magnitudes, axis=1)

        loc_dist = jnp.expand_dims(jnp.sum((pos[rows] - pos[cols]) ** 2, 1), axis=1)
        edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], axis=1)

        return (nodes, pos, edge_indices, vel, edge_attr), targets

    def _to_transformer(
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

        # nodes = jnp.ones((pos.shape[0], 1))  # all input nodes are set to 1
        # loc_dist = jnp.sum((pos[rows] - pos[cols]) ** 2, axis=-1)[:, None]
        vel_attr = get_velocity_attr(pos, vel, rows, cols)
        # edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], 1)

        magnitudes = jnp.sqrt(jnp.sum(vel ** 2, axis=1))
        nodes = jnp.expand_dims(magnitudes, axis=1)
        nodes = jnp.concatenate((nodes, charges), axis=1)

        loc_dist = jnp.expand_dims(jnp.sum((pos[rows] - pos[cols]) ** 2, 1), axis=1)
        edge_attr = jnp.concatenate([edge_attribute, loc_dist, vel_attr], axis=1)

        features_node = nodes.shape[1]
        nodes = jnp.reshape(nodes, (batch_size, n_nodes, features_node))

        features_edges = edge_attr.shape[1]
        edge_attr = jnp.reshape(edge_attr, (batch_size, n_nodes * (n_nodes - 1), features_edges))

        dim_target = targets.shape[1]
        targets = jnp.reshape(targets, (batch_size, n_nodes, dim_target))
        return (nodes, edge_attr, pos), targets

    if model == 'egnn':
        return _to_egnn
    else:
        return _to_transformer
