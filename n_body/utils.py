# from https://github.com/gerkone/egnn-jax/blob/main/nbody/utils.py

import jax.numpy as jnp
import jraph
from typing import Callable, Dict, List, Tuple


def NbodyGraphTransform(
    n_nodes: int,
    batch_size: int,
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

    def _to_jraph(
        data: List,
    ) -> Tuple[jraph.GraphsTuple, Dict[str, jnp.ndarray], jnp.ndarray]:
        props = {}
        pos, vel, edge_attribute, a, targets = data

        cur_batch = int(pos.shape[0] / n_nodes)

        edge_indices = full_edge_indices[:, : n_nodes * (n_nodes - 1) * cur_batch]
        senders, receivers = edge_indices[0], edge_indices[1]

        #pos = jnp.array(pos)
        #vel = jnp.array(vel)
        #edge_attribute = jnp.array(edge_attribute)

        # relative distances between particles
        pos_dist = jnp.sum((pos[senders] - pos[receivers]) ** 2, axis=-1)[:, None]
        props["edge_attribute"] = jnp.concatenate([edge_attribute, pos_dist], axis=-1)
        props["pos"] = pos
        props["vel"] = vel

        graph = jraph.GraphsTuple(
            # velocity magnitude as node features (scalar)
            nodes=jnp.sqrt(jnp.sum(vel**2, axis=-1))[:, None],
            edges=None,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([n_nodes] * cur_batch),
            n_edge=jnp.array([len(senders) // cur_batch] * cur_batch),
            globals=None,
        )

        return (
            graph,
            props,
            targets,
        )

    return _to_jraph