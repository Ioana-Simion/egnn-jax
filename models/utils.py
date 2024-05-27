# The contents of this file are mostly taken from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html

import jax.numpy as jnp
import flax.linen as nn
from jax import vmap

def scaled_dot_product(q, k, v, mask=None):

    d_k = jnp.array(q.shape[-1])
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)

    return values, attention

def mask_from_edges():
    def _mask_from_edges(edge_index, num_nodes, num_edges):
        mask = jnp.zeros((num_nodes, num_edges))
        row, col = edge_index

        # Find valid (non-padding) indices
        valid_row_mask = (row != -1)
        valid_col_mask = (col != -1)
        
        valid_rows = jnp.where(valid_row_mask, row, 0)
        valid_cols = jnp.where(valid_col_mask, col, 0)

        # Set -inf for valid edges
        mask = mask.at[valid_rows, jnp.arange(num_edges).astype(jnp.int32)].set(-jnp.inf * valid_row_mask)
        mask = mask.at[valid_cols, jnp.arange(num_edges).astype(jnp.int32)].set(-jnp.inf * valid_col_mask)
        is_neg_inf = jnp.isneginf(mask)
        mask = jnp.where(is_neg_inf, 0.0, -jnp.inf)
        return mask
    return vmap(_mask_from_edges, in_axes=(0, None, None))



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
