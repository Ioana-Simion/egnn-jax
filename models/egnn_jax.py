import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Callable


def xavier_init(gain=1.0):
    def init(key, shape, dtype):
        bound = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Sum elements along segments of a tensor."""
    result = jnp.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    return result.at[segment_ids].add(data)

class EdgeModel(nn.Module):
    hidden_nf: int
    act_fn: Callable

    def setup(self):
        self.edge_mlp = nn.Sequential([nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(self.hidden_nf), self.act_fn])
    
    def __call__(self, edge_index, h, coord, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        radial = self.coord2radial(edge_index, coord)
        radial = jnp.expand_dims(radial, axis=-1)
        radial = jnp.tile(radial, (1, 1, h.shape[-1]))
        out = jnp.concatenate([source, target, radial], axis=1)
        return self.edge_mlp(out)

    def coord2radial(self, edge_index, coord):
        senders, receivers = edge_index
        coord_i, coord_j = coord[senders], coord[receivers]
        distance = jnp.sum((coord_i - coord_j) ** 2, axis=1, keepdims=True)
        return distance

class NodeModel(nn.Module):
    hidden_nf: int
    act_fn: Callable

    def setup(self):
        self.node_mlp = nn.Sequential([nn.Dense(self.hidden_nf), self.act_fn, nn.Dense(self.hidden_nf)])
    
    def __call__(self, edge_index, edge_attr, x):
        row, col = edge_index
        num_nodes = x.shape[0]

        agg = unsorted_segment_sum(edge_attr, row, num_segments=num_nodes)
        if agg.shape[-1] != x.shape[-1]:
            agg = jnp.pad(agg, ((0, 0), (0, x.shape[-1] - agg.shape[-1])))

        combined = jnp.concatenate([x, agg], axis=1)
        out = self.node_mlp(combined)
        
        return out, combined

class E_GCL(nn.Module):
    hidden_nf: int
    act_fn: Callable

    def setup(self):
        self.edge_model = EdgeModel(self.hidden_nf, self.act_fn)
        self.node_model = NodeModel(self.hidden_nf, self.act_fn)
        
    def __call__(self, h, edge_index, coord, edge_attr):
        m_ij = self.edge_model(edge_index, h, coord, edge_attr)
        h, agg = self.node_model(edge_index, m_ij, h)
        return h, coord, agg

class EGNN_equiv(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu
    n_layers: int = 4
    velocity: bool = False

    @nn.compact
    def __call__(self, h, x, edges, vel, edge_attr):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL(self.hidden_nf, act_fn=self.act_fn, velocity=self.velocity)(h, edges, x, vel, edge_attr=edge_attr)
        h = nn.Dense(self.out_node_nf)(h)
        return h, x

class EGNN_QM9(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu
    n_layers: int = 4

    @nn.compact
    def __call__(self, h, x, edges, edge_attr, node_mask, n_nodes):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL(self.hidden_nf, act_fn=self.act_fn)(h, edges, x, edge_attr=edge_attr)
        h = h * node_mask[:, None]
        h = h.reshape(-1, n_nodes, self.hidden_nf)
        h = jnp.sum(h, axis=1)
        h = nn.Dense(self.out_node_nf)(h)
        return h, x

def unsorted_segment_sum(data, segment_ids, num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments)

# def unsorted_segment_sum(data, segment_ids, num_segments):
#     result_shape = (num_segments,) + data.shape[1:]
#     result = jnp.zeros(result_shape, dtype=data.dtype)
#     result = result.at[segment_ids].add(data)
#     return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    seg_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    seg_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    seg_count = jnp.maximum(seg_count, 1)  # Avoid 0 division
    return seg_sum / seg_count


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]

    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = jnp.ones(len(edges[0]) * batch_size, dtype=jnp.float32).reshape(-1, 1)
    edges = [
        jnp.array(edges[0]).astype(jnp.int32),
        jnp.array(edges[1]).astype(jnp.int32)
    ]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
    return edges, edge_attr

def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charges = jnp.clip(charges, 0, charge_power) / charge_scale
    charges = jax.nn.one_hot(charges.astype(jnp.int32), num_classes=charge_power + 1)
    return jnp.concatenate([one_hot, charges], axis=-1)

if __name__ == "__main__":
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3
    charge_power = 2
    charge_scale = 1.0

    h = jnp.ones((batch_size * n_nodes, n_feat))
    x = jnp.ones((batch_size * n_nodes, x_dim))
    charges = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])
    one_hot = jax.nn.one_hot(charges, num_classes=3)

    h = preprocess_input(one_hot, charges, charge_power, charge_scale)

    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    rng = jax.random.PRNGKey(42)

    # Initialize EGNN
    egnn = EGNN_equiv(hidden_nf=32, out_node_nf=1)

    params = egnn.init(rng, h, x, edges, edge_attr)["params"]

    # Now you can use the model's `apply` method with these parameters
    output = egnn.apply({"params": params}, h, x, edges, edge_attr)
