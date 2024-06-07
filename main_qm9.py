import os
import jax
import jraph
import json
import optax
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from qm9.utils import GraphTransform, TransformDLBatches
from flax.training import checkpoints
from typing import Callable
from utils.utils import get_model, get_loaders_and_statistics, set_seed, get_property_index, denormalize, normalize
import gc
from optax import cosine_decay_schedule
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

@partial(jax.jit, static_argnames=["loss_fn", "opt_update", "max_num_nodes"])
def update(params, x, edge_attr, edge_index, pos, node_mask, max_num_nodes, target, opt_state, loss_fn, opt_update):
    #using jax grad only instead of value and grad
    grads = jax.grad(loss_fn)(params, x, edge_attr, edge_index, pos, node_mask, max_num_nodes, target)
    loss = loss_fn(params, x, edge_attr, edge_index, pos, node_mask, max_num_nodes, target)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state

@partial(jax.jit, static_argnames=["model_fn", "task", "training", "max_num_nodes"])
def l1_loss(params, h, edge_attr, edge_index, pos, node_mask, max_num_nodes, 
            target, model_fn, meann, mad, training=True, task="graph"):
    """
    Logic is : while training normalize the targets (we do this in collate)
               while evaluating denormalize the predictions
               we denormalize the target too because we normalize it in collate
    """
    if not training:
        pred = jax.lax.stop_gradient(model_fn(params, h, pos, edge_index, None, node_mask, max_num_nodes)[0])
        pred = denormalize(pred, meann, mad)
        target = denormalize(target, meann, mad)
    else:
        pred = model_fn(params, h, pos, edge_index, None, node_mask, max_num_nodes)[0]

    assert pred.shape == target.shape, f"Shape mismatch: pred.shape = {pred.shape}, target_padded.shape = {target.shape}"
    # TODO no_grad for training = false
    return jnp.mean(jnp.abs(pred - target))

def evaluate(loader, params, max_num_nodes, loss_fn, graph_transform, meann, mad, task="graph"):
    eval_loss = 0.0
    for data in tqdm(loader, desc="Evaluating", leave=False):
        feat, target = graph_transform(data)
        h, x, edges, edge_attr, node_mask = feat
        loss = loss_fn(params, h, edge_attr, edges, x, node_mask, max_num_nodes, target, meann=meann, mad=mad, training=False)
        eval_loss += loss
    return eval_loss / len(loader)

def train_model(args, model, graph_transform, model_name, checkpoint_path):
    train_loader, val_loader, test_loader, meann, mad, max_num_nodes, max_num_edges = get_loaders_and_statistics(args)
    property_idx = get_property_index(args.property)
    graph_transform_fn = graph_transform(property_idx)

    mad = jnp.maximum(mad, 1e-6)
    print(f"Mean: {meann}, MAD: {mad}")

    init_feat, _ = graph_transform_fn(next(iter(train_loader)))
    lr_schedule = cosine_decay_schedule(init_value=args.lr, decay_steps=args.epochs)

    opt_init, opt_update = optax.chain(
        optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay)
    )
    #opt_init, opt_update = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    params = model.init(jax_seed, *init_feat, max_num_nodes)
    opt_state = opt_init(params)

    loss_fn = partial(l1_loss, model_fn=model.apply, meann=meann, mad=mad, task=args.task)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, graph_transform=graph_transform_fn, meann=meann, mad=mad, task=args.task)

    train_scores = []
    val_scores = []
    test_loss, best_val_epoch = 0, 0
    best_val_loss = float('inf')

    global_step = 0
    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = tuple(tensor.to(args.device) for tensor in batch)
            feat, target = graph_transform_fn(batch)
            x, pos, edge_index, edge_attr, node_mask = feat
            loss, params, opt_state = update_fn(params, x, edge_attr, edge_index, pos, node_mask, max_num_nodes, target=target, opt_state=opt_state)
            loss_item = float(jax.device_get(loss))
            train_loss += loss_item
            writer.add_scalar('Loss/train', loss_item, global_step=global_step)

            # Manually trigger garbage collection
            gc.collect()
            jax.clear_caches()
            global_step += 1

        train_loss /= len(train_loader)
        train_scores.append(train_loss)
        writer.add_scalar('AvgLoss/train', train_loss, epoch)

        if epoch % args.val_freq == 0:
            val_loss = eval_fn(val_loader, params, max_num_nodes)
            val_loss_item = float(jax.device_get(val_loss))
            val_scores.append(val_loss_item)
            writer.add_scalar('Loss/val', val_loss_item, epoch)
            print(f"[Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")

            if val_loss_item < best_val_loss:
                print("\t   (New best performance, saving model...)")
                best_val_loss = val_loss_item
                save_model(params, checkpoint_path, model_name)
                test_loss = eval_fn(test_loader, params, max_num_nodes)
                jax.clear_caches()

    print(f"Final Performance [Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, "
          f"Validation loss: {best_val_loss:.6f}, Test loss: {float(jax.device_get(test_loss)):.6f}")

    writer.flush()
    writer.close()

    results = {
        "test_mae": float(jax.device_get(test_loss)),
        "val_scores": [float(val) for val in val_scores],
        "train_scores": [float(train) for train in train_scores]
    }

    with open(_get_result_file(checkpoint_path, model_name), "w") as f:
        json.dump(results, f)

    if val_scores:
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
    else:
        print("No validation scores to plot.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size (number of graphs).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr-scheduling", action="store_true", help="Use learning rate scheduling")
    parser.add_argument("--weight_decay", type=float, default=1e-16, help="Weight decay")
    parser.add_argument("--val_freq", type=int, default=1, help="Evaluation frequency (number of epochs)")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset", choices=["qm9", "charged", "gravity"])
    parser.add_argument("--property", type=str, default="homo", help="Label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve")
    parser.add_argument("--num_hidden", type=int, default=128, help="Number of values in the hidden layers")
    parser.add_argument("--num_layers", type=int, default=7, help="Number of message passing layers")
    parser.add_argument("--basis", type=str, default="gaussian", choices=["gaussian", "bessel"], help="Radial basis function.")
    parser.add_argument("--double-precision", action="store_true", help="Use double precision")
    parser.add_argument("--model_name", type=str, default="egnn", help="model")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--charge_power", type=int, default=2, help="Maximum power to take into one-hot features")
    parser.add_argument("--charge_scale", type=float, default=1.0, help="Scale for normalizing charges")

    parsed_args = parser.parse_args()

    set_seed(parsed_args.seed)

    parsed_args.target = "pos"
    parsed_args.task = "node"
    parsed_args.radius = 1000.0
    parsed_args.node_type = "continuous"

    graph_transform = TransformDLBatches

    # Unique datetime log
    log_dir = os.path.join("runs", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    model = get_model(parsed_args)
    train_model(parsed_args, model, graph_transform, "qm9_EGNN", "assets")