import jax
import copy
import torch
import argparse
import jax.numpy as jnp
from tqdm import tqdm
from qm9.utils import calc_mean_mad
from utils.utils import get_model, get_loaders, set_seed


def train_step(state, batch):
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, *batch)
        loss = jnp.mean(
            (preds - batch["labels"])
        )  # TODO since l1 loss, otherwise **2 missing
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def main(args):
    # Generate model
    model = get_model(args)  # .to(args.device)
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Commented this out bcs we don't have model.parameters()
    # print(f"Number of parameters: {num_params}")
    # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    mean, mad = calc_mean_mad(train_loader)
    # mean, mad = mean.to(args.device), mad.to(args.device)

    # Get optimization objects
    # state = train_state.TrainState.create(
    #     apply_fn=model.apply,
    #     params=model.init(jax.random.PRNGKey(0), jnp.ones([1, 5])),  # Example input
    #     tx=optax.adam(learning_rate=0.001),
    # )

    criterion = torch.nn.L1Loss(reduction="sum")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    _, best_val_mae, best_model = float("inf"), float("inf"), None

    for _ in tqdm(range(args.epochs)):
        epoch_mae_train, epoch_mae_val = 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(args.device)

            pred = model(batch)
            loss = criterion(pred, (batch.y - mean) / mad)
            mae = criterion(pred * mad + mean, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            epoch_mae_train += mae.item()

        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            pred = model(batch)
            mae = criterion(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader.dataset)
        epoch_mae_val /= len(val_loader.dataset)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model)

        scheduler.step()

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        pred = best_model(batch)
        mae = criterion(pred * mad + mean, batch.y)
        test_mae += mae.item()

    test_mae /= len(test_loader.dataset)
    print(f"Test MAE: {test_mae}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers")

    # Model parameters
    parser.add_argument("--model_name", type=str, default="egnn", help="model")
    parser.add_argument("--num_hidden", type=int, default=77, help="hidden features")
    parser.add_argument("--num_layers", type=int, default=7, help="number of layers")
    parser.add_argument(
        "--act_fn", type=str, default="silu", help="activation function"
    )

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-16, help="learning rate"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="qm9", help="dataset")
    parser.add_argument(
        "--target-name", type=str, default="mu", help="target feature to predict"
    )  # idk just guessing - Greg
    parser.add_argument(
        "--dim", type=int, default=1, help="dimension"
    )  # idk just guessing - Greg
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--property",
        type=str,
        default="homo",
        help="label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve",
    )

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the seed
    set_seed(parsed_args.seed)

    main(parsed_args)
