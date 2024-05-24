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
from utils.utils import get_model, get_loaders, set_seed
import gc
import json

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

@partial(jax.jit, static_argnames=["loss_fn", "opt_update"])
def update(params, x, y, opt_state, loss_fn, opt_update):
    grads = jax.grad(loss_fn)(params, x, y)
    loss = loss_fn(params, x, y)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state

@partial(jax.jit, static_argnames=["model_fn"])
def mse_loss(params, x, y, model_fn):
    pred = model_fn(params, x, train=True)
    return jnp.mean((pred - y) ** 2)

def evaluate(loader, params, loss_fn, model_fn):
    eval_loss = 0.0
    for data in tqdm(loader, desc="Evaluating", leave=False):
        x, edge_attr, pos, mask, edge_mask, y = data
        loss = loss_fn(params, (edge_attr, x), y, model_fn)
        eval_loss += loss
    return eval_loss / len(loader)

def train_model(args, model, model_name, checkpoint_path):
    train_loader, val_loader, test_loader = get_loaders(args, transformer=True)

    init_feat, _, _, _, _, _ = next(iter(train_loader))
    opt_init, opt_update = optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay)
    params = model.init(jax_seed, *init_feat)
    opt_state = opt_init(params)

    loss_fn = partial(mse_loss, model_fn=model.apply)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, model_fn=model.apply)

    train_scores = []
    val_scores = []
    test_loss, best_val_epoch = 0, 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x, edge_attr, pos, mask, edge_mask, y = batch
            loss, params, opt_state = update_fn(params=params, x=(edge_attr, x), y=y, opt_state=opt_state)
            train_loss += loss

            # Manually trigger garbage collection
            gc.collect()
            jax.clear_caches()

        train_loss /= len(train_loader)
        train_scores.append(float(jax.device_get(train_loss)))

        if epoch % args.val_freq == 0:
            val_loss = eval_fn(val_loader, params)
            val_scores.append(float(jax.device_get(val_loss)))
            print(f"[Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}")

            if len(val_scores) == 1 or val_loss < val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(params, checkpoint_path, model_name)
                best_val_epoch = epoch
                test_loss = eval_fn(test_loader, params)
                jax.clear_caches()

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
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr-scheduling", action="store_true", help="Use learning rate scheduling")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="Weight decay")
    parser.add_argument("--val_freq", type=int, default=10, help="Evaluation frequency (number of epochs)")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Dataset", choices=["qm9", "charged", "gravity"])
    parser.add_argument("--property", type=str, default="homo", help="Label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve")

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
