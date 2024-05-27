# The contents of this file are mostly taken from:
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
# It has here been adapted into an EGNN framework.

import jax
import jax.numpy as jnp
import flax.linen as nn
from . import utils


class MultiheadAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        self.qkv_proj = nn.Dense(
            3 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )

        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        if mask is not None:
            mask = utils.expand_mask(mask)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = utils.scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o, attention


class MultiHeadCrossAttention(nn.Module):
    embed_dim: int  # Output dimension
    num_heads: int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.kv_proj = nn.Dense(
            2 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )

        self.q_proj = nn.Dense(
            1 * self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
        )

        self.o_proj = nn.Dense(
            self.embed_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, kv_inp, q_inp, mask=None):
        batch_size, seq_length_q, embed_dim = q_inp.shape
        seq_len_kv = kv_inp.shape[1]
        if mask is not None:
            mask = utils.expand_mask(mask)
        kv = self.kv_proj(kv_inp)
        q = self.q_proj(q_inp)

        # Separate K, V from linear output
        kv = kv.reshape(batch_size, seq_len_kv, self.num_heads, -1)
        kv = kv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        k, v = jnp.array_split(kv, 2, axis=-1)

        # Separate Q from linear output
        q = q.reshape(batch_size, seq_length_q, self.num_heads, -1)
        q = q.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]

        # Determine value outputs
        values, attention = utils.scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length_q, embed_dim)
        o = self.o_proj(values)

        return o, attention


class EncoderBlock(nn.Module):
    input_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        # Attention layer
        self.self_attn = MultiheadAttention(
            embed_dim=self.input_dim, num_heads=self.num_heads
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.input_dim),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        # Attention part
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        # MLP part
        linear_out = x
        for l in self.linear:
            linear_out = (
                l(linear_out)
                if not isinstance(l, nn.Dropout)
                else l(linear_out, deterministic=not train)
            )

        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.layers = (
            [
                EncoderBlock(
                    self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob
                )
                for _ in range(self.num_layers)
            ]
            if self.num_layers > 0
            else []
        )

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)

        return x

    def get_attention_maps(self, x, mask=None, train=True):
        # A function to return the attention maps within the model for a single application
        # Used for visualization purpose later
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask)
            attention_maps.append(attn_map)
            x = l(x, mask=mask, train=train)

        return attention_maps


class PositionalEncoding(nn.Module):
    d_model: int  # Hidden dimensionality of the input.
    max_len: int = 5000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, : x.shape[1]]

        return x


class TransformerPredictor(nn.Module):
    model_dim: int  # Hidden dimensionality to use inside the Transformer
    num_classes: int  # Number of classes to predict per sequence element
    num_heads: int  # Number of heads to use in the Multi-Head Attention blocks
    num_layers: int  # Number of encoder blocks to use
    dropout_prob: float = 0.0  # Dropout to apply inside the model
    input_dropout_prob: float = 0.0  # Dropout to apply on the input features

    def setup(self):
        # Input dim -> Model dim
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.model_dim)
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(self.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            dim_feedforward=2 * self.model_dim,
            num_heads=self.num_heads,
            dropout_prob=self.dropout_prob,
        )
        # Output classifier per sequence lement
        self.output_net = [
            nn.Dense(self.model_dim),
            nn.LayerNorm(),
            nn.relu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.num_classes),
        ]

    def __call__(self, x, mask=None, add_positional_encoding=True, train=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
            train - If True, dropout is stochastic
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=mask, train=train)
        for l in self.output_net:
            x = l(x) if not isinstance(l, nn.Dropout) else l(x, deterministic=not train)

        return x

    def get_attention_maps(
            self, x, mask=None, add_positional_encoding=True, train=True
    ):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)

        attention_maps = self.transformer.get_attention_maps(x, mask=mask, train=train)

        return attention_maps


class EGNNTransformer(nn.Module):
    num_edge_encoder_blocks: int = 2
    num_node_encoder_blocks: int = 2
    num_combined_encoder_blocks: int = 4

    model_dim: int = 128
    num_heads: int = 8
    dropout_prob: float = 0.0
    output_dim: int = 3

    input_dropout_prob: float = 0.0

    predict_pos: bool = False
    velocity: bool = False
    n_nodes: int = 5

    node_only: bool = False
    invariant_pos: bool = False

    def setup(self):

        # CLS token embedding
        self.cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.model_dim])

        # Node level
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer_nodes = nn.Dense(self.model_dim)

        # Node Encoder
        self.node_encoder = TransformerEncoder(
            num_layers=self.num_node_encoder_blocks,
            input_dim=self.model_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.model_dim,
            dropout_prob=self.dropout_prob,
        )

        # Output classifier
        self.output_net = nn.Dense(self.output_dim)

        if not self.node_only:
            self.input_layer_edges = nn.Dense(self.model_dim)

            # Edge Encoder
            if self.num_edge_encoder_blocks > 0:
                self.edge_encoder = TransformerEncoder(
                    num_layers=self.num_edge_encoder_blocks,
                    input_dim=self.model_dim,
                    num_heads=self.num_heads,
                    dim_feedforward=self.model_dim,
                    dropout_prob=self.dropout_prob,
                )

            # Combined Encoder
            self.combined_encoder = TransformerEncoder(
                num_layers=self.num_combined_encoder_blocks,
                input_dim=self.model_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.model_dim,
                dropout_prob=self.dropout_prob,
            )

            # Cross Attention
            self.cross_attention = MultiHeadCrossAttention(
                embed_dim=self.model_dim,
                num_heads=self.num_heads,
            )

    def __call__(self, node_inputs, edge_inputs, coords, vel, cross_mask=None, train=True):

        batch_size, _, _ = node_inputs.shape

        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))

        # Input layer
        node_inputs = self.input_dropout(node_inputs, deterministic=not train)
        node_encoded = self.input_layer_nodes(node_inputs)
        if not self.predict_pos:
            node_encoded = jnp.concatenate([cls_tokens, node_encoded], axis=1)
        # Node Encoder
        node_encoded = self.node_encoder(node_encoded, mask=None, train=train)
        if not self.node_only:
            # Input layer
            edge_inputs = self.input_dropout(edge_inputs, deterministic=not train)
            edge_encoded = self.input_layer_edges(edge_inputs)

            # Edge Encoder
            if self.num_edge_encoder_blocks > 0:
                edge_encoded = self.edge_encoder(edge_encoded, mask=None, train=train)

            # Cross Attention
            if not self.predict_pos:
                edge_enrichment, _ = self.cross_attention(edge_encoded, node_encoded[:,1:,:], mask=cross_mask)
            else:
                edge_enrichment, _ = self.cross_attention(edge_encoded, node_encoded, mask=cross_mask)
            node_encoded = node_encoded + edge_enrichment

            # Combined Encoder
            node_encoded = self.combined_encoder(
                node_encoded, mask=None, train=train
            )

        if self.predict_pos:
            output = self.output_net(node_encoded)
            if self.invariant_pos:
                return output
            else:
                if self.velocity:
                    return coords + output * vel
                else:
                    return coords + output
        else:
            return self.output_net(node_encoded[:, 0])
