# The contents of this file are mostly taken from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html

import jax.numpy as jnp
import flax.linen as nn


def scaled_dot_product(q, k, v, mask=None):

    d_k = jnp.array(q.shape[-1])
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)

    return values, attention

def mask_from_edges(edge_index, num_nodes, num_edges):
    mask = jnp.zeros((edge_index.shape[0], num_nodes, num_edges))
    row, col = edge_index.transpose(1, 0)
    # (0,2), (1, 1)
    mask = mask.at[row, jnp.tile(jnp.arange(num_edges).astype(jnp.int32), (edge_index.shape[0], 1))].set(1)
    mask = mask.at[col, jnp.tile(jnp.arange(num_edges).astype(jnp.int32), (edge_index.shape[0], 1))].set(1)
    return mask

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):

    assert (
        mask.ndim >= 2
    ), "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, 1)

    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, 0)

    return mask


def cosine_warmup_schedule(base_lr: float, warmup: int, max_iters: int):
    assert (
        warmup > 0 and max_iters > 0
    ), "Ensure that the 'warmup' and 'max_iters' are above zero!"

    # Create function to return lr based on iteration count
    def get_lr(train_iter):
        lr_factor = 0.5 * (1 + jnp.cos(jnp.pi * train_iter / max_iters))
        if train_iter <= warmup:
            lr_factor *= train_iter * 1.0 / warmup
        return lr_factor * base_lr

    return get_lr
