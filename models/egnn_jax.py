import flax.linen as nn
import jax
import jax.numpy as jnp


def xavier_init(gain):
    def init(key, shape, dtype):
        bound = gain * jnp.sqrt(6. / (shape[0] + shape[1]))
        jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
    return init


class E_GCL(nn.Module):
    """
    E(n) Equivariant Message Passing Layer
    """
    hidden_dim: int
    edges_in_d: int
    act_fn: callable
    residual: bool

    # def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.activation.silu, residual=True, coords_agg='mean'):
    #     super(E_GCL, self).__init__()
    #     input_edge = input_nf * 2
    #     self.residual = residual
    #     self.coords_agg = coords_agg
    #     self.epsilon = 1e-8
    #     edge_coords_nf = 1
    #
    #     self.edge_mlp = nn.Sequential([
    #         nn.Dense(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
    #         act_fn,
    #         nn.Dense(hidden_nf, hidden_nf),
    #         act_fn])
    #
    #     self.node_mlp = nn.Sequential([
    #         nn.Dense(hidden_nf + input_nf, hidden_nf),
    #         act_fn,
    #         nn.Dense(hidden_nf, output_nf)])
    #
    #     layer = nn.Dense(hidden_nf, 1, kernel_init=xavier_init(gain=0.001))
    #
    #     coord_mlp = []
    #     coord_mlp.append(nn.Dense(hidden_nf, hidden_nf))
    #     coord_mlp.append(act_fn)
    #     coord_mlp.append(layer)
    #     self.coord_mlp = nn.Sequential([*coord_mlp])


    def edge_model(self, edge_index, h, coord, edge_attr):
        row, col = edge_index
        source, target = h[row], h[col]
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.act_fn,
            nn.Dense(self.hidden_dim),
            self.act_fn
        ])
        out = jnp.concatenate([source, target, radial, edge_attr], axis=1)
        return edge_mlp(out)

    def node_model(self, edge_index, edge_attr, x):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        node_mlp = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.act_fn,
            nn.Dense(self.hidden_dim)
        ])

        agg = jnp.concatenate([x, agg], axis=1)
        out = node_mlp(agg)

        # TODO do we need to add x to out? to update it
        return out, agg

    def coord_model(self, edge_index, coord_diff, edge_feat, coord):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.sum(coord_diff**2, axis=1, keepdims=True)
        return radial, coord_diff


    @nn.compact
    def __call__(self, h, edge_index, coord, edge_attr=None):
        m_ij = self.edge_model(edge_index, h, coord, edge_attr)
        h, agg = self.node_model(edge_index, m_ij, x)








        return h, coord, edge_attr


class EGNN(nn.Module):
    in_node_nf: int
    hidden_nf: int
    out_node_nf: int
    in_edge_nf: int = 0
    act_fn = nn.silu  # default activation function
    n_layers: int = 4
    residual: bool = True


    @nn.compact
    def __call__(self, h, x, edges, edge_attr):
        h = nn.Dense(self.hidden_nf)(h)
        for i in range(self.n_layers):
            layer = E_GCL(self.hidden_nf, edges_in_d=self.in_edge_nf, act_fn=self.act_fn, residual=self.residual) #name=f"gcl_{i}"
            h, x, _ = layer(h, edges, x, edge_attr=edge_attr)
        h = nn.Dense(self.out_node_nf)(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, jax.ones_like(data))
    return result / count.clamp(min=1)


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
    edge_attr = jnp.ones((len(edges[0]) * batch_size, 1))
    edges = [jnp.array(edges[0], dtype=jnp.int32), jnp.array(edges[1], dtype=jnp.int32)]
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

    # Dummy variables h, x and fully connected edges
    h = jnp.ones((batch_size * n_nodes, n_feat))
    x = jnp.ones((batch_size * n_nodes, x_dim))
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    rng = jax.random.PRNGKey(42)

    # Initialize EGNN
    egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

    params = egnn.init(rng, h, x, edges, edge_attr)['params']

    # Now you can use the model's `apply` method with these parameters
    output = egnn.apply({'params': params}, h, x, edges, edge_attr)
