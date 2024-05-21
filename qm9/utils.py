import copy
import jax.numpy as jnp
from typing import Callable


def GraphTransform(
    batch_size: int,
) -> Callable:
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
        jax_data = {}
        jax_data["x"] = jnp.array(data.x.numpy())
        jax_data["pos"] = jnp.array(data.pos.numpy())
        jax_data["edge_index"] = jnp.array(data.edge_index.numpy())
        jax_data["edge_attr"] = jnp.array(data.edge_attr.numpy())
        jax_data["y"] = jnp.array(data.y.numpy())

        return (
            jax_data["x"],
            jax_data["pos"],
            jax_data["edge_index"],
            jax_data["edge_attr"],
        ), jax_data["y"]

    return _to_jax


class RemoveNumHs:
    def __call__(self, data):
        data = copy.copy(data)
        data.x = data.x[:, :-1]
        return data
