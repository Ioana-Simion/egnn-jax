# jax grad process from https://github.com/gerkone/egnn-jax/blob/main/validate.py

import os
import jax
import jraph
import json
import torch
import pickle
import optax
import argparse
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from qm9.utils import GraphTransform
from flax.training import train_state
from models.egnn_jax import get_edges_batch
from typing import Dict, Callable, Tuple, Iterable
from utils.utils import get_model, get_loaders, set_seed

# Seeding
jax_seed = jax.random.PRNGKey(42)


def _get_config_file(model_path, model_name):
    # Name of the file for storing hyperparameter details
    return os.path.join(model_path, model_name + ".config")


def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")


def save_model(model, params, model_path, model_name):
    """
    Given a model, we save the parameters and hyperparameters.

    Inputs:
        model - Network object without parameters
        params - Parameters to save of the model
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
    """
    # config_dict = {'hidden_sizes': model.hidden_sizes,
    #               'num_classes': model.num_classes}
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(
        model_path, model_name
    )
    # with open(config_file, "w") as f:
    #    json.dump(config_dict, f)
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, "wb") as f:
        pickle.dump(params, f)


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


def load_model(model_path, model_name, state=None):
    """
    Loads a saved model from disk.

    Inputs:
        model_path - Path of the checkpoint directory
        model_name - Name of the model (str)
        state - (Optional) If given, the parameters are loaded into this training state. Otherwise,
                a new one is created alongside a network architecture.
    """
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(
        model_path, model_name
    )
    assert os.path.isfile(
        config_file
    ), f'Could not find the config file "{config_file}". Are you sure this is the correct path and you have your model config stored here?'
    assert os.path.isfile(
        model_file
    ), f'Could not find the model file "{model_file}". Are you sure this is the correct path and you have your model stored here?'
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    # TODO check this in depth
    net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, "rb") as f:
        params = pickle.load(f)
    state = state.replace(params=params)
    return state, net


@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params,
    feat,
    target: jnp.ndarray,
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
):
    loss, grads = jax.value_and_grad(loss_fn)(params, feat, target)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state

def get_property_index(property_name):
    property_dict = {
        'alpha': 0,
        'gap': 1,
        'homo': 2,
        'lumo': 3,
        'mu': 4,
        'Cv': 5,
        'G': 6,
        'H': 7,
        'r2': 8,
        'U': 9,
        'U0': 10,
        'zpve': 11
    }
    return property_dict[property_name]

def create_graph(h, x, edges, edge_attr):
    
    n_node = jnp.array([h.shape[0]])
    n_edge = jnp.array([edges.shape[1]])

    graphs_tuple = jraph.GraphsTuple(
            nodes=h,
            edges=edge_attr,
            senders=edges[0],
            receivers=edges[1],
            globals=None,
            n_node=n_node,
            n_edge=n_edge
        )
    
    return graphs_tuple

def create_padding_mask(h, x, edges, edge_attr):
    # num_nodes = h.shape[0]
    # num_edges = edges.shape[1]
    graph = create_graph(h,x,edges,edge_attr)

    node_mask = jraph.get_node_padding_mask(graph)
    edge_mask = jraph.get_edge_padding_mask(graph)
    
    return node_mask, edge_mask

def compute_mean_mad(dataloader, property_idx):
    values = []
    for batch in dataloader:
        values.append(jnp.array(batch.y[:, property_idx].numpy()))
    values = jnp.concatenate(values)
    meann = jnp.mean(values)
    ma = jnp.abs(values - meann)
    mad = jnp.mean(ma)
    return meann, mad


def normalize(pred, meann, mad):
    return (pred - meann) / mad

def denormalize(pred, meann, mad):
    return mad * pred + meann

@partial(jax.jit, static_argnames=["model_fn", "task", "training"])
def l1_loss(params, feat, target, model_fn, meann, mad, training=True, task="graph"):
    h, x, edges, edge_attr = feat
    pred = model_fn(params, h, x, edges, edge_attr)[0]

    node_mask, edge_mask = create_padding_mask(h, x, edges, edge_attr)
    
    if training:
        # Normalize prediction and target for training
        pred = normalize(pred, meann, mad)
        target = normalize(target, meann, mad)
    else:
        # only for eval
        pred = denormalize(pred, meann, mad)

    target_padded = jnp.pad(target, ((0, h.shape[0] - target.shape[0]), (0, 0)), mode='constant')
    
    # Apply the mask to the predictions and targets
    pred = pred * node_mask[:, None]
    target_padded = target_padded * node_mask[:, None]

    assert pred.shape == target_padded.shape, f"Shape mismatch: pred.shape = {pred.shape}, target_padded.shape = {target_padded.shape}"
    
    return jnp.mean(jnp.abs(pred - target_padded))

def evaluate(loader, params, loss_fn, graph_transform, meann, mad, task="graph"):
    eval_loss = 0.0
    for data in loader:
        feat, target = graph_transform(data)
        loss = jax.lax.stop_gradient(loss_fn(params, feat, target, meann, mad, training=False))
        eval_loss += jax.block_until_ready(loss)
    return eval_loss / len(loader)


def train_model(args, graph_transform, model_name, checkpoint_path):
    # Generate model
    model = get_model(args)

    # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)

    property_idx = get_property_index(args.property)
    meann, mad = compute_mean_mad(train_loader, property_idx)
    print(f"Mean: {meann}, MAD: {mad}")

    init_feat, _ = graph_transform(next(iter(train_loader)))
    
    opt_init, opt_update = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    params = model.init(jax_seed, *init_feat)
    opt_state = opt_init(params)

    loss_fn = partial(l1_loss, model_fn=model.apply, meann=meann, mad=mad, task=args.task)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, graph_transform=graph_transform, meann=meann, mad=mad, task=args.task)

    train_scores = []
    val_scores = []
    test_loss, best_val_epoch = 0, 0

    for epoch in tqdm(range(args.epochs)):
        ############
        # Training #
        ############
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feat, target = graph_transform(batch)
            loss, params, opt_state = update_fn(params=params, feat=feat, target=target, opt_state=opt_state)
            train_loss += loss
        train_loss /= len(train_loader)
        train_scores.append(train_loss)

        ##############
        # Validation #
        ##############
        if epoch % args.val_freq == 0:
            val_loss = eval_fn(val_loader, params)
            val_scores.append(val_loss)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")

            if len(val_scores) == 1 or val_loss < val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(model, params, checkpoint_path, model_name)
                best_val_epoch = epoch
                test_loss = eval_fn(test_loader, params)

    print(f"Final Performance [Epoch {epoch + 1:2d}] Training loss: {train_scores[best_val_epoch]:.6f}, "
          f"Validation loss: {val_scores[best_val_epoch]:.6f}, Test loss: {test_loss:.6f}")
    results = {"test_mae": test_loss, "val_scores": val_scores[best_val_epoch], "train_scores": train_scores[best_val_epoch]}
    with open(_get_result_file(checkpoint_path, model_name), "w") as f:
        json.dump(results, f)

    # Plot validation performance
    sns.set()
    plt.plot(range(1, len(train_scores) + 1), train_scores, label="Train")
    plt.plot(range(1, len(val_scores) + 1), val_scores, label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(min(val_scores), max(train_scores) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print(f" Test loss: {results['test_loss']:.6f} ".center(50, "=") + "\n")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (number of graphs).",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--lr-scheduling",
        action="store_true",
        help="Use learning rate scheduling",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="qm9",
        help="Dataset",
        choices=["qm9", "charged", "gravity"],
    )

    parser.add_argument("--property", type=str, default="homo", help="Label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve")
    
    # Model parameters
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=128,
        help="Number of values in the hidden layers",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="gaussian",
        choices=["gaussian", "bessel"],
        help="Radial basis function.",
    )

    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision",
    )
    parser.add_argument("--model_name", type=str, default="egnn", help="model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_samples", type=int, default=3000)

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)

    parsed_args.target = "pos"
    parsed_args.task = "node"
    parsed_args.radius = 1000.0
    parsed_args.node_type = "continuous"

    graph_transform = GraphTransform(batch_size=parsed_args.batch_size)

    train_model(parsed_args, graph_transform, "test", "assets")
