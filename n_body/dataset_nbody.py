# This script has been modified to return JAX arrays instead of torch Tensors.
import numpy as np
import jax.numpy as jnp


class NBodyDataset:
    """
    NBodyDataset

    """

    def __init__(self, partition="train", max_samples=1e8, dataset_name="nbody"):
        self.partition = partition
        if self.partition == "val":
            self.sufix = "valid"
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        loc = np.load("n_body/dataset/data/loc_" + self.sufix + ".npy")
        vel = np.load("n_body/dataset/data/vel_" + self.sufix + ".npy")
        edges = np.load("n_body/dataset/data/edges_" + self.sufix + ".npy")
        charges = np.load("n_body/dataset/data/charges_" + self.sufix + ".npy")

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to jax arrays and swap n_nodes <--> n_features dimensions
        loc, vel = jnp.array(loc).transpose((0, 1, 3, 2)), jnp.array(vel).transpose(
            (0, 1, 3, 2)
        )
        n_nodes = loc.shape[2]
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0 : self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = jnp.array(edge_attr).transpose(
            (1, 0)
        )  # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = jnp.expand_dims(edge_attr, axis=2)

        return (
            jnp.array(loc),
            jnp.array(vel),
            jnp.array(edge_attr),
            edges,
            jnp.array(charges),
        )

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].shape[1]

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]

    def __len__(self):
        return self.data[0].shape[0]

    def get_edges(self, batch_size, n_nodes):
        edges = [jnp.array(self.edges[0]), jnp.array(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
        return edges


if __name__ == "__main__":
    NBodyDataset()
