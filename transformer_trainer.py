import os
import jax
import optax
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from flax.training import checkpoints
from functools import partial
from qm9.utils import GraphTransform
from utils.utils import get_model, set_seed, NodeDistance, RemoveNumHs, get_loaders_and_statistics, get_property_index
import gc
import json
from torch_geometric.datasets import QM9
from torch.utils.data import DataLoader
import torch_geometric.transforms as T

# Seeding
jax_seed = jax.random.PRNGKey(42)

def _get_checkpoint_dir(model_path, model_name):
    return os.path.abspath(os.path.join(model_path, model_name))

def save_model(params, model_path, model_name):
    checkpoint_dir = _get_checkpoint_dir(model_path, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=params, step=0, overwrite=True)

def load_model(model_path, model_name):
    checkpoint_dir = _get_checkpoint_dir(model_path, model_name)
    params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)
    return params

def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")

@partial(jax.jit, static_argnames=["opt_update", "model_fn"])
def update(params, edge_attr, node_attr, cross_mask, target, opt_state, rng, model_fn, opt_update):
    rng, dropout_rng = jax.random.split(rng)
    grads = jax.grad(mse_loss)(params, edge_attr, node_attr, cross_mask, target, dropout_rng, model_fn)
    loss = mse_loss(params, edge_attr, node_attr, cross_mask, target, dropout_rng, model_fn)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state, rng

#use jit again? removed for debugging
def mse_loss(params, edge_attr, node_attr, cross_mask, target, dropout_rng, model_fn):
    variables = {'params': params}
    rngs = {'dropout': dropout_rng}
    pred = model_fn(variables, node_attr, edge_attr, None, None, cross_mask=cross_mask, train=True, rngs=rngs)

    return jnp.mean((pred - target) ** 2)

def normalize_data(data):
    """Normalize data for better training stability."""
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    return (data - mean) / std

def handle_nan(data):
    """Replace nan and inf values with 0."""
    data = jnp.nan_to_num(data, pos_inf=0, neg_inf=0)
    return data

def evaluate(loader, params, rng, model_fn):
    eval_loss = 0.0
    num_batches = len(loader)
    
    if num_batches == 0:
        print("Warning: The data loader is empty. Ensure the dataset and splits are correctly created.")
        return float('inf')  # Return infinity to show issue

    for data in tqdm(loader, desc="Evaluating", leave=False):
        edge_attr, node_attr, _, _, _, target = data
        target = jnp.array(target)
        
        # Handle nan and inf values
        #edge_attr = handle_nan(edge_attr)
        #node_attr = handle_nan(node_attr)
        #target = handle_nan(target)

        _, dropout_rng = jax.random.split(rng)
        loss = mse_loss(params, edge_attr, node_attr, target, dropout_rng, model_fn)
        eval_loss += loss
    return eval_loss / num_batches

def train_model(args, model, model_name, checkpoint_path):
    (
        train_loader, 
        val_loader, 
        test_loader, 
        meann, 
        mad, 
        max_num_nodes, 
        max_num_edges
    ) = get_loaders_and_statistics(args, transformer=True)

    property_idx = get_property_index(args.property)

    init_node_attr, init_edge_attr, edge_attn_mask, x, y = next(iter(train_loader))
    opt_init, opt_update = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    rng, init_rng = jax.random.split(jax_seed)
    params = model.init(init_rng, init_node_attr, init_edge_attr, coords=None, vel=None, cross_mask=edge_attn_mask)['params']
    opt_state = opt_init(params)

    train_scores = []
    val_scores = []
    test_loss, best_val_epoch = 0, 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            node_attr, edge_attr, cross_mask, pos, target = batch
            target = target[:, property_idx]

            # Handle nan and inf values
            #edge_attr = handle_nan(edge_attr)
            #node_attr = handle_nan(node_attr)
            #target = handle_nan(target)
            
            loss, params, opt_state, rng = update(params=params, edge_attr=edge_attr, node_attr=node_attr, cross_mask=cross_mask, target=target, opt_state=opt_state, rng=rng, model_fn=model.apply, opt_update=opt_update)
            train_loss += loss

            # Manually trigger garbage collection
            gc.collect()
            jax.clear_caches()

        train_loss /= len(train_loader)
        train_scores.append(float(jax.device_get(train_loss)))

        if epoch % args.val_freq == 0:
            val_loss = evaluate(val_loader, params, rng, model.apply)
            val_scores.append(float(jax.device_get(val_loss)))
            print(f"[Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")

            if len(val_scores) == 1 or val_loss < min(val_scores):
                print("\t   (New best performance, saving model...)")
                save_model(params, checkpoint_path, model_name)
                best_val_epoch = len(val_scores) - 1
                test_loss = evaluate(test_loader, params, rng, model.apply)
                jax.clear_caches()

    if val_scores:
        best_val_epoch = val_scores.index(min(val_scores))
    else:
        best_val_epoch = 0

    print(f"Final Performance [Epoch {epoch + 1:2d}] Training loss: {train_scores[best_val_epoch]:.6f}, "
          f"Validation loss: {val_scores[best_val_epoch]:.6f}, Test loss: {float(jax.device_get(test_loss)):.6f}")
    
    results = {
        "test_mae": float(jax.device_get(test_loss)),
        "val_scores": [float(val) for val in val_scores],
        "train_scores": [float(train) for train in train_scores]
    }

    with open(_get_result_file(checkpoint_path, model_name), "w") as f:
        json.dump(results, f)

    plt.plot(range(1, len(train_scores) + 1), train_scores, label="Train")
    plt.plot(range(1, len(val_scores) + 1), val_scores, label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(min(val_scores), max(train_scores) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print(f" Test loss: {results['test_mae']:.6f} ".center(50, "=") + "\n")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (number of graphs).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  # Reduced learning rate
    parser.add_argument("--lr-scheduling", action="store_true", help="Use learning rate scheduling")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument("--val_freq", type=int, default=10, help="Evaluation frequency (number of epochs)")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset", choices=["qm9", "charged", "gravity"])
    parser.add_argument("--property", type=str, default="homo", help="Label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve")
    parser.add_argument(
        "--nbody_name",
        type=str,
        default="nbody_small",
        help="Which n_body dataset to use",
        choices=["nbody", "nbody_small"],
    )

    # Model parameters
    parser.add_argument("--num_edge_encoders", type=int, default=3, help="Number of edge encoder blocks")
    parser.add_argument("--num_node_encoders", type=int, default=3, help="Number of node encoder blocks")
    parser.add_argument("--num_combined_encoder_blocks", type=int, default=3, help="Number of combined encoder blocks")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--edge_input_dim", type=int, default=6, help="Dimension of the edge input")
    parser.add_argument("--node_input_dim", type=int, default=11, help="Dimension of the node input")

    parser.add_argument("--model_name", type=str, default="transformer", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=3000)

    parsed_args = parser.parse_args()

    set_seed(parsed_args.seed)

    model = get_model(parsed_args)
    train_model(parsed_args, model, parsed_args.model_name, "assets")