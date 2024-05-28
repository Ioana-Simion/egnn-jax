import jax
import argparse
import jax.numpy as jnp
import torch
from torch import sin, cos
from n_body.utils import NbodyBatchTransform
from utils.utils import get_model, get_loaders, set_seed


# Seeding
jax_seed = jax.random.PRNGKey(42)

def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype = gamma.dtype)

def rot_y(beta):
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype = beta.dtype)

def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

def test_equivariance(model, args):
    train_loader, val_loader, test_loader = get_loaders(args, transformer=True)
    init_feat, _ = graph_transform(next(iter(train_loader)))
    rng, init_rng = jax.random.split(jax_seed)


    R = jnp.array(rot(*torch.rand(3)).cpu().numpy())
    T = jnp.array(torch.randn(1, 1, 3).cpu().numpy())

    feats, edges, coors, vel = init_feat
    params = model.init(init_rng, *(feats, edges, coors, vel))["params"]
    trans_out = model.apply(variables = {"params": params}, rngs = {"dropout": rng}, node_inputs = feats, coords = coors @ R + T, vel = vel, edge_inputs = edges)
    out = model.apply(variables = {"params": params}, rngs = {"dropout": rng}, node_inputs = feats, coords = coors, vel = vel, edge_inputs = edges)

    extra_trans_out = jnp.matmul(out, R.T) + T
    assert torch.allclose(torch.from_numpy(jax.device_get(trans_out.copy())), torch.from_numpy(jax.device_get(extra_trans_out.copy())), atol = 1e-6), 'coords are not equivariant'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size (number of graphs)."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--lr-scheduling", action="store_true", help="Use learning rate scheduling"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument(
        "--val_freq",
        type=int,
        default=1,
        help="Evaluation frequency (number of epochs)",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="charged",
        help="Dataset",
        choices=["qm9", "charged", "gravity"],
    )
    parser.add_argument(
        "--property",
        type=str,
        default="homo",
        help="Label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve",
    )
    parser.add_argument(
        "--nbody_name",
        type=str,
        default="nbody_small",
        help="Which n_body dataset to use",
        choices=["nbody", "nbody_small"],
    )
    parser.add_argument('--train_from_checkpoint', action='store_true', default=False,
                        help='Enables training form checkpoint')

    # Model parameters
    parser.add_argument(
        "--num_edge_encoders", type=int, default=1, help="Number of edge encoder blocks"
    )
    parser.add_argument(
        "--num_node_encoders", type=int, default=1, help="Number of node encoder blocks"
    )
    parser.add_argument(
        "--num_combined_encoder_blocks",
        type=int,
        default=1,
        help="Number of combined encoder blocks",
    )
    parser.add_argument("--dim", type=int, default=64, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--edge_input_dim", type=int, default=6, help="Dimension of the edge input"
    )
    parser.add_argument(
        "--node_input_dim", type=int, default=11, help="Dimension of the node input"
    )

    parser.add_argument(
        "--model_name", type=str, default="transformer", help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--nbody_path", default='n_body/dataset/data/')

    parsed_args = parser.parse_args()

    set_seed(parsed_args.seed)
    graph_transform = NbodyBatchTransform(n_nodes=5, batch_size=parsed_args.batch_size, model=parsed_args.model_name)
    train_loader, val_loader, test_loader = get_loaders(parsed_args, transformer=True)
    init_feat, a = graph_transform(next(iter(train_loader)))
    model = get_model(parsed_args)

    test_equivariance(model, parsed_args)
