import math
import os
import gc
import jax
import json
import optax
import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial

from n_body.utils import NbodyGraphTransform
from qm9.utils import GraphTransform
from flax.training import checkpoints
from utils.utils import get_model, get_loaders, set_seed


# Seeding
jax_seed = jax.random.PRNGKey(42)


def _get_checkpoint_dir(model_path, model_name):
    return os.path.abspath(os.path.join(model_path, model_name))


def save_model(params, model_path, model_name):
    checkpoint_dir = _get_checkpoint_dir(model_path, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir, target=params, step=0, overwrite=True
    )


def load_model(model_path, model_name):
    checkpoint_dir = _get_checkpoint_dir(model_path, model_name)
    params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)
    return params


def _get_result_file(model_path, model_name):
    return os.path.join(model_path, model_name + "_results.json")


@partial(jax.jit, static_argnames=["opt_update", "model_fn"])
def update(params, feat, target, opt_state, rng, model_fn, opt_update):
    rng, dropout_rng = jax.random.split(rng)
    grads = jax.grad(mse_loss)(
        params, feat, target, dropout_rng, model_fn
    )
    loss = mse_loss(params, feat, target, dropout_rng, model_fn)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state, rng


@partial(jax.jit, static_argnames=["model_fn"])
def mse_loss(params, feat, target, dropout_rng, model_fn):
    variables = {"params": params}
    rngs = {"dropout": dropout_rng}
    pred = model_fn(variables, *feat, train=True, rngs=rngs)
    return jnp.mean((pred - target) ** 2)


def evaluate(loader, params, rng, model_fn):
    eval_loss = 0.0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        feat, target = graph_transform(batch)

        _, dropout_rng = jax.random.split(rng)
        loss = mse_loss(params, feat, target, dropout_rng, model_fn)
        eval_loss += loss
    return eval_loss / len(loader)


def train_model(args, model, model_name, graph_transform, checkpoint_path):
    train_loader, val_loader, test_loader = get_loaders(args, transformer=True)

    init_feat, _ = graph_transform(next(iter(train_loader)))
    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )
    rng, init_rng = jax.random.split(jax_seed)

    if args.train_from_checkpoint:
        params = load_model(model_path=checkpoint_path, model_name=model_name)
    else:
        params = model.init(init_rng, *init_feat)["params"]

    opt_state = opt_init(params)

    num_params = sum(math.prod(param.shape) for param in jax.tree_util.tree_leaves(params))

    print(f'Parameters: {num_params}')

    update_fn = partial(update, model_fn=model.apply, opt_update=opt_update)

    train_scores = []
    val_scores = []
    test_loss, best_val_epoch = 0, 0

    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feat, target = graph_transform(batch)

            loss, params, opt_state, rng = update_fn(
                params=params,
                feat=feat,
                target=target,
                opt_state=opt_state,
                rng=rng,
            )
            train_loss += loss

            # Manually trigger garbage collection
            gc.collect()
            jax.clear_caches()

        train_loss /= len(train_loader)
        train_scores.append(float(jax.device_get(train_loss)))

        if epoch % args.val_freq == 0:
            val_loss = evaluate(val_loader, params, rng, model.apply)
            val_scores.append(float(jax.device_get(val_loss)))
            print(
                f"[Epoch {epoch + 1:2d}] Training loss: {train_loss:.6f}, Validation loss: {val_loss:.6f}"
            )

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

    print(
        f"Final Performance [Epoch {epoch + 1:2d}] Training loss: {train_scores[best_val_epoch]:.6f}, "
        f"Validation loss: {val_scores[best_val_epoch]:.6f}, Test loss: {float(jax.device_get(test_loss)):.6f}"
    )

    results = {
        "test_mae": float(jax.device_get(test_loss)),
        "val_scores": [float(val) for val in val_scores],
        "train_scores": [float(train) for train in train_scores],
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
        "--model_name", type=str, default="transformer", help="Model name"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--nbody_path", default='n_body/dataset/data/')

    parsed_args = parser.parse_args()

    set_seed(parsed_args.seed)
    graph_transform = NbodyGraphTransform(n_nodes=5, batch_size=parsed_args.batch_size, model=parsed_args.model_name)
    train_loader, val_loader, test_loader = get_loaders(parsed_args, transformer=True)
    init_feat, a = graph_transform(next(iter(train_loader)))
    model = get_model(parsed_args)

    train_model(parsed_args, model, parsed_args.model_name, graph_transform, "assets")
