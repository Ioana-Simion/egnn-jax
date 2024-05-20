# jax grad process from https://github.com/gerkone/egnn-jax/blob/main/validate.py

import argparse
import json
from typing import Dict, Callable, Tuple, Iterable

import jraph
import torch
from tqdm import tqdm

from models.egnn_jax import get_edges_batch
from qm9.utils import GraphTransform
from utils.utils import get_model, get_loaders, set_seed
from flax.training import train_state
import jax.numpy as jnp
import jax
import optax
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from functools import partial

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
    #config_dict = {'hidden_sizes': model.hidden_sizes,
    #               'num_classes': model.num_classes}
    os.makedirs(model_path, exist_ok=True)
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    #with open(config_file, "w") as f:
    #    json.dump(config_dict, f)
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead save the parameters simply in a pickle file.
    with open(model_file, 'wb') as f:
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
    config_file, model_file = _get_config_file(model_path, model_name), _get_model_file(model_path, model_name)
    assert os.path.isfile(config_file), f"Could not find the config file \"{config_file}\". Are you sure this is the correct path and you have your model config stored here?"
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    with open(config_file, "r") as f:
        config_dict = json.load(f)
    # TODO check this in depth
    net = None
    # You can also use flax's checkpoint package. To show an alternative,
    # you can instead load the parameters simply from a pickle file.
    with open(model_file, 'rb') as f:
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
    opt_update: Callable
):
    loss, grads = jax.value_and_grad(loss_fn)(params, feat, target)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state

#TODO:check to see if we need something else
@partial(jax.jit, static_argnames=["model_fn"])
def mse(
    params,
    feat,
    target: jnp.ndarray,
    model_fn: Callable,
):
    # pred is tuple h,x
    pred = model_fn(
        params,
        *feat,
    )[1]
    return (jnp.power(pred - target, 2)).mean()


def evaluate(
    loader: Iterable,
    params,
    loss_fn: Callable,
    graph_transform: Callable,
) -> float:
    eval_loss = 0.0
    for data in loader:
        feat, target = graph_transform(data)
        loss = jax.lax.stop_gradient(loss_fn(params, feat, target))
        eval_loss += jax.block_until_ready(loss)
    return eval_loss / len(loader)


# @jax.jit
# def train_step(state, batch):
#     grad_fn = jax.value_and_grad(calculate_loss)
#     loss, grads = grad_fn(state.params, state.apply_fn, batch)
#     state = state.apply_gradients(grads=grads)
#     return state, loss
#
#
# @jax.jit
# def eval_step(state, batch):
#     loss = calculate_loss(state.params, state.apply_fn, batch)
#     return loss


# def test_model(state, data_loader):
#     """
#     Test a model on a specified dataset.
#
#     Inputs:
#         state - Training state including parameters and model apply function.
#         data_loader - DataLoader object of the dataset to test on (validation or test)
#     """
#     true_preds, count = 0., 0
#     for batch in data_loader:
#         acc = eval_step(state, batch)
#         batch_size = batch[0].shape[0]
#         true_preds += acc * batch_size
#         count += batch_size
#     test_acc = true_preds / count
#     return test_acc.item()



def train_model(args, graph_transform, model_name, checkpoint_path):
    # # Generate model
    model = get_model(args)  # .to(args.device)

    # # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)

    init_feat, _ = graph_transform(next(iter(train_loader)))

    # Get optimization objects
    #state = train_state.TrainState.create(
    #    apply_fn=model.apply, params=model.init(jax.random.PRNGKey(0), init_graph),
    #    tx=optax.adamw(args.lr, weight_decay=args.weight_decay)
    #)
    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    params = model.init(jax_seed, *init_feat)

    loss_fn = partial(mse, model_fn=model.apply)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)

    # TODO do cosine annealing
    # lr_schedule = optax.cosine_decay_schedule(init_value=args.lr,
    #                                           decay_steps=args.epochs * steps_per_epoch,
    #                                           alpha=final_lr / init_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # lr = args.lr
    # weight_decay = args.weight_decay
    #
    # # Total number of steps (epochs * steps_per_epoch)
    # # You would need to know or define steps_per_epoch based on your dataset
    # total_steps = args.epochs * steps_per_epoch
    #
    # # Setup cosine decay for the learning rate without restarts
    # lr_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps, alpha=0)
    #
    # # Create the optimizer with weight decay
    # optimizer = optax.chain(
    #     optax.adam(learning_rate=lr_schedule, weight_decay=weight_decay)
    # )

    train_scores = []
    val_scores = []
    test_loss = 0

    for epoch in tqdm(range(args.epochs)):
        ############
        # Training #
        ############
        train_loss, val_loss = 0, 0
        for batch in tqdm(train_loader.dataset, desc=f"Epoch {epoch+1}", leave=False):
            feat, target = graph_transform(batch)
            loss, params, opt_state = update_fn(
                params=params,
                feat=feat,
                target=target,
                opt_state=opt_state
            )
            train_loss += loss
        train_loss /= len(train_loader.dataset)
        train_scores.append(train_loss)

        ##############
        # Validation #
        ##############
        if epoch % args.val_freq == 0:
            val_loss = eval_fn(val_loader, params)

            val_scores.append(val_loss)
            print(f"[Epoch {epoch + 1:2d}] Training accuracy: {train_loss:05.2%}, Validation accuracy: {val_loss:4.2%}")

            if len(val_scores) == 1 or val_loss > val_scores[best_val_epoch]:
                print("\t   (New best performance, saving model...)")
                save_model(model, params, checkpoint_path, model_name)
                best_val_epoch = epoch
                test_loss = eval_fn(test_loader, params)

    print(f"Final Performance [Epoch {epoch + 1:2d}] Training accuracy: {train_scores[best_val_epoch]:05.2%}, "
          f"Validation accuracy: {val_scores[best_val_epoch]:4.2%}, Test accuracy: {test_loss:2.2%} ")
    results = {"test_mae": test_loss, "val_scores": val_scores[best_val_epoch],
               "train_scores": train_scores[best_val_epoch]}
    with open(_get_result_file(checkpoint_path, model_name), "w") as f:
        json.dump(results, f)

    # Plot a curve of the validation accuracy
    sns.set()
    plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
    plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
    plt.xlabel("Epochs")
    plt.ylabel("Validation accuracy")
    plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
    plt.title(f"Validation performance of {model_name}")
    plt.legend()
    plt.show()
    plt.close()

    print((f" Test accuracy: {results['test_acc']:4.2%} ").center(50, "=") + "\n")
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
    parser.add_argument('--model_name', type=str, default='egnn',
                        help='model')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument("--max_samples", type=int, default=3000)

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)

    parsed_args.target = "pos"
    parsed_args.task = "node"
    parsed_args.radius = 1000.0
    parsed_args.node_type = "continuous"

    graph_transform = GraphTransform(batch_size=parsed_args.batch_size)

    train_model(parsed_args, graph_transform, 'test', 'assets')