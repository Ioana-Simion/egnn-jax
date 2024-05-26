# This script has been modified to return JAX arrays instead of torch Tensors.
import torch
import numpy as np
import jax.numpy as jnp


class NBodyDataset:
    """
    NBodyDataset

    """

    def __init__(self, partition="train", max_samples=1e8, dataset_name="nbody_small", base_path='/n_body/dataset/data/'):
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
        self.base_path = base_path
        self.data, self.edges = self.load()


    def load(self):
        loc = np.load(self.base_path + "loc_" + self.sufix + ".npy")
        vel = np.load(self.base_path + "vel_" + self.sufix + ".npy")
        edges = np.load(self.base_path + "edges_" + self.sufix + ".npy")
        charges = np.load(self.base_path + "charges_" + self.sufix + ".npy")

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
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
        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = np.array(edge_attr)
        edge_attr = torch.Tensor(edge_attr).transpose(0, 1).unsqueeze(2)

        return (
            torch.Tensor(loc),
            torch.Tensor(vel),
            torch.Tensor(edge_attr),
            edges,
            torch.Tensor(charges),
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
            frame_0, frame_T = 6, 8
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
