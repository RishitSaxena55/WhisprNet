from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np


class MultiHeadAttention:
    """
    Multi Head Attention
    """

    def __init__(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention = ScaledDotProductAttention()

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        N, L, E = query.shape
        S = key.shape[1]

        Q = self.q_proj.forward(query)  # (N, L, E)
        K = self.k_proj.forward(key)    # (N, S, E)
        V = self.v_proj.forward(value)  # (N, S, E)

        Q = self._split_heads(Q)  # (N, H, L, D)
        K = self._split_heads(K)  # (N, H, S, D)
        V = self._split_heads(V)  # (N, H, S, D)

        mask = self._merge_masks(key_padding_mask, attn_mask, N, L, S)

        attn_output = self.attention.forward(Q, K, V, mask)  # (N, H, L, D)

        concat = self._concat_heads(attn_output)  # (N, L, E)
        output = self.out_proj.forward(concat)    # (N, L, E)

        return output

    def backward(self, d_output):
        # Backprop through output projection
        d_concat = self.out_proj.backward(d_output)  # (N, L, E)

        # Split into heads
        d_concat = self._split_heads(d_concat)  # (N, H, L, D)

        # Backprop through attention
        d_Q, d_K, d_V = self.attention.backward(d_concat)  # (N, H, L/S, D)

        # Concatenate gradients back
        d_Q = self._concat_heads(d_Q)  # (N, L, E)
        d_K = self._concat_heads(d_K)  # (N, S, E)
        d_V = self._concat_heads(d_V)  # (N, S, E)

        # Backprop through input projections
        d_q = self.q_proj.backward(d_Q)  # (N, L, E)
        d_k = self.k_proj.backward(d_K)  # (N, S, E)
        d_v = self.v_proj.backward(d_V)  # (N, S, E)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask, N, L, S):
        combined_mask = None

        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S)
            kpm = key_padding_mask[:, None, None, :].astype(bool)
            kpm = np.broadcast_to(kpm, (N, self.num_heads, L, S))
        else:
            kpm = None

        if attn_mask is not None:
            # (L, S) -> (1, 1, L, S)
            am = attn_mask[None, None, :, :].astype(bool)
            am = np.broadcast_to(am, (N, self.num_heads, L, S))
        else:
            am = None

        if kpm is not None and am is not None:
            combined_mask = np.logical_or(kpm, am)
        elif kpm is not None:
            combined_mask = kpm
        elif am is not None:
            combined_mask = am

        return combined_mask

    def _split_heads(self, x):
        N, L, E = x.shape
        H = self.num_heads
        D = self.head_dim
        x = x.reshape(N, L, H, D)
        x = x.transpose(0, 2, 1, 3)  # (N, H, L, D)
        return x

    def _concat_heads(self, x):
        N, H, L, D = x.shape
        x = x.transpose(0, 2, 1, 3)  # (N, L, H, D)
        x = x.reshape(N, L, H * D)
        return x
