import jax
import jax.numpy as jnp
import flax.linen as nn


def xavier_init(gain=1.0):
    def init(key, shape, dtype):
        bound = gain * jnp.sqrt(6.0 / (shape[0] + shape[1]))
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init

def unsorted_segment_sum(data, segment_ids, num_segments):
    return jax.ops.segment_sum(data, segment_ids, num_segments)


def unsorted_segment_mean(data, segment_ids, num_segments):
    seg_sum = jax.ops.segment_sum(data, segment_ids, num_segments)
    seg_count = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
    seg_count = jnp.maximum(seg_count, 1)  # Avoid 0 division
    return seg_sum / seg_count


class E_GCL(nn.Module):
    """
    E(n) Equivariant Message Passing Layer
    """
    hidden_nf: int
    act_fn: callable
    velocity: bool = False

    def setup(self):
        self.edge_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.hidden_nf),
            self.act_fn,
        ])
        self.node_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.hidden_nf),
        ])

    def edge_model(self, edge_index, h, coord, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        radial = self.coord2radial(edge_index, coord)
        out = jnp.concatenate([source, target, radial], axis=-1)
        return self.edge_mlp(out)

    def node_model(self, edge_index, edge_attr, x):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        combined = jnp.concatenate([x, agg], axis=-1)
        out = self.node_mlp(combined)
        return out, combined

    def coord_model(self, edge_index, edge_feat, coord):
        row, col = edge_index
        coord_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(1, kernel_init=xavier_init(gain=0.001)),
        ])
        coord_out = coord_mlp(edge_feat)
        trans = (coord[row] - coord[col]) * coord_out
        agg = unsorted_segment_mean(trans, row, num_segments=coord.shape[0])
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        senders, receivers = edge_index
        coord_i, coord_j = coord[senders], coord[receivers]
        distance = jnp.sum((coord_i - coord_j) ** 2, axis=-1, keepdims=True)
        return distance

    def coord_vel_model(self, coord, h, vel):
        coord_mlp_vel = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(1)
        ])
        coord += coord_mlp_vel(h) * vel
        return coord

    @nn.compact
    def __call__(self, h, edge_index, coord, vel=None, edge_attr=None):
        m_ij = self.edge_model(edge_index, h, coord, edge_attr)
        h, agg = self.node_model(edge_index, m_ij, h)
        if self.velocity:
            coord = self.coord_vel_model(coord, h, vel)
        return h, coord, m_ij


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
            h, x, _ = E_GCL(self.hidden_nf, act_fn=self.act_fn, velocity=self.velocity)(
                h, edges, x, vel, edge_attr=edge_attr)
        h = nn.Dense(self.out_node_nf)(h)
        return h, x


class EGNN_QM9(nn.Module):
    hidden_nf: int
    out_node_nf: int
    act_fn: callable = nn.relu  # default activation function
    n_layers: int = 4

    @nn.compact
    def __call__(self, h, x, edges, edge_attr, node_mask, n_nodes):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            h, x, _ = E_GCL(self.hidden_nf, act_fn=self.act_fn)(
                h, edges, x, edge_attr=edge_attr)

        h = h * node_mask[:, None]
        h = h.reshape(-1, n_nodes, self.hidden_nf)
        h = jnp.sum(h, axis=1)
        h = nn.Dense(self.out_node_nf)(h)
        return h, x
def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges[..., None] / charge_scale) ** jnp.arange(charge_power + 1)
    charge_tensor = charge_tensor.reshape(*charges.shape, -1)
    atom_scalars = (one_hot[..., None] * charge_tensor[..., None, :]).reshape(charges.shape[0], -1)
    return atom_scalars


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


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3
    charge_power = 2
    charge_scale = 1.0

    # Dummy variables h, x and fully connected edges
    charges = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])
    one_hot = jax.nn.one_hot(charges, num_classes=charge_power + 1)
    h = preprocess_input(one_hot, charges, charge_power, charge_scale)
    h = h.reshape(batch_size * n_nodes, -1)
    x = jnp.ones((batch_size * n_nodes, x_dim))
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    rng = jax.random.PRNGKey(42)

    # Initialize EGNN
    egnn = EGNN_equiv(hidden_nf=32, out_node_nf=1)

    params = egnn.init(rng, h, x, edges, edge_attr)["params"]

    # Now you can use the model's `apply` method with these parameters
    output = egnn.apply({"params": params}, h, x, edges, edge_attr)