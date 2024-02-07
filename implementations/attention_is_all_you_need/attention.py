r"""
Scaled Dot-Product Attention Module
===================================

This module defines the Scaled Dot-Product Attention mechanism, which is a key
component in transformer architectures. It follows the equations:

.. math::
    \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V


Multi-Head Attention Module
===========================

This module defines the Multi-Head Attention mechanism, which is a key
component in transformer architectures. It follows the equations:

.. math::
    \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H)W^O

where each head is computed as:

.. math::
    \text{head}_h = \text{Attention}(QW^Q_h, KW^K_h, VW^V_h) where h \in [1, H]

Here, Q, K, V are the query, key, and value vectors. W^Q_h, W^K_h, W^V_h are
parameter matrices. H is the number of heads.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

__all__ = ["MultiHeadedAttention", "ScaledDotProductAttention"]


class ScaledDotProductAttention(nn.Module):
    """Implements scaled dot-product attention mechanism.

    This class is a derived instance of the `Attention` class that computes the
    scaled dot-product attention, defined by the following operation:

    .. math::
        \\text{Attention}(Q, K, V) = \\text{softmax} \\left( \\frac{QK^T}{\\sqrt{d_k}} \\right) V

    where:

    -   Q is the query matrix
    -   K is the key matrix
    -   V is the value matrix
    -   d_k is the dimension of the keys

    Methods
    -------
    forward(query, key, value, mask)
        Computes the forward pass for the scaled dot-product attention.
    """
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass for scaled dot-product attention.

        This function applies the attention mechanism on the input tensors `query`,
        `key`, and `value`. It's worth noting that for cross-attention, the sequence
        lengths of `query` and `key`/`value` may differ. This is because `query` is
        usually projected from the decoder's states, while `key` and `value` are from
        the encoder's states.
        """
        # fmt: off
        d_q               = query.size(dim=-1)

        attention_scores  = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())
        attention_scores  = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector    = torch.matmul(attention_weights, value)
        # fmt: on
        return context_vector, attention_weights


class MultiHeadedAttention(nn.Module):
    __slots__ = [
        "d_model",
        "d_k",
        "d_q",
        "d_v",
        "H",
        "W_Q",
        "W_K",
        "W_V",
        "W_O",
        "attention",
        "dropout",
        "context_vector",
        "attention_weights",
    ]

    def __init__(
        self,
        attention: ScaledDotProductAttention,
        H: int,
        d_model: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert (
            d_model % H == 0
        ), "The number of heads must divide the embedding dimension."

        # fmt: off
        self.d_model   = d_model       # D
        self.d_k       = d_model // H  # stay true to notations
        self.d_q       = d_model // H
        self.d_v       = d_model // H

        self.H         = H             # number of heads

        # shadow my notations, actually they are of shape D x D.
        self.W_Q       = nn.Linear(self.d_model, self.d_q * self.H, bias=bias)  # D x D
        self.W_K       = nn.Linear(self.d_model, self.d_k * self.H, bias=bias)
        self.W_V       = nn.Linear(self.d_model, self.d_v * self.H, bias=bias)
        self.W_O       = nn.Linear(self.d_model, self.d_model, bias=bias)

        self.attention = attention
        self.dropout   = nn.Dropout(p=dropout, inplace=False)

        self.context_vector: torch.Tensor
        self.attention_weights: torch.Tensor

        self._init_weights()
        # fmt: on

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Notations
        ---------
        B:      Batch size
        S or L: Source sequence length
        T or L: Target sequence length
        D:      Embedding dimension
        H:      Number of heads

        Parameters
        ----------
        query:  The query tensor.
                type:  torch.Tensor
                shape: (B, S or T, D)
        key:    The key tensor.
                type:  torch.Tensor
                shape: (B, S or T, D)
        value:  The value tensor.
                type:  torch.Tensor
                shape: (B, S or T, D)
        mask:   Mask to be applied to the attention scores.
                type:  torch.BoolTensor
                shape: (B, 1, S or T, S or T)

        Returns
        -------
        O:  The output of the multi-headed attention mechanism.
            type:  torch.Tensor
            shape: (B, S or T, D)
        """
        # fmt: off
        if mask is not None:
            assert mask.ndim     == 4, f"Mask should have 4 dimensions but got {mask.ndim}."
            assert mask.shape[0] == query.shape[0], ("Batch size of mask and query must match.")
            assert mask.shape[1] == 1, ("Mask should have shape (batch_size, 1, seq_len, seq_len).")
            assert mask.shape[2] == mask.shape[3] == query.shape[1], ("Mask should have shape (batch_size, 1, seq_len, seq_len).")


        Q = self.W_Q(query).contiguous() # Z @ W_Q -> LxD @ DxD = LxD
        K = self.W_K(key).contiguous()   # Z @ W_K
        V = self.W_V(value).contiguous() # Z @ W_V

        Q = self.transpose_qkv(Q)        # [B, H, L, D]
        K = self.transpose_qkv(K)
        V = self.transpose_qkv(V)

        # Attention
        self.context_vector, self.attention_weights = self.attention(Q, K, V, mask)
        context_vector_concat                       = self.reverse_transpose_qkv(self.context_vector)
        # fmt: on

        O = self.W_O(
            context_vector_concat
        )  # context_vector_concat @ W_O -> LxD @ DxD = LxD
        return O  # type: ignore[no-any-return]

    def _init_weights(self) -> None:
        """See PyTorch's MultiHeadAttention code for reference."""
        # we assume _qkv_same_embed_dim is True
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads.
        Why does transpose allow parallel computation? So originally the shape of
        the query, key, and value is (B, L, D), and we want to split the D into H
        heads to become (B, L, H, D / H). But this is not the shape we want (could
        be due to efficiency reasons), so we transpose the shape to (B, H, L, D / H)
        so all heads can be computed in parallel (efficiently).

        Parameters
        ----------
        q_or_k_or_v: The query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, L, D)

        Returns
        -------
        q_or_k_or_v: The transposed query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, H, L, D / H)
        """
        # fmt: off
        # 1. q_or_k_or_v is shape (B, L, D)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        batch_size, seq_len, _ = q_or_k_or_v.shape
        q_or_k_or_v            = q_or_k_or_v.view(batch_size, seq_len, self.H, self.d_model // self.H)

        # 3. switch H from 3rd to 2nd dimension, or in python swap 2nd to 1st dimension and 1st to 2nd dimension
        #    shape (B, H, L, D / H = d_qkv)
        q_or_k_or_v            = q_or_k_or_v.permute(0, 2, 1, 3)
        # fmt: on
        return q_or_k_or_v

    def reverse_transpose_qkv(self, q_or_k_or_v: torch.Tensor) -> torch.Tensor:
        """Reverse the transposition operation for concatenating multiple attention heads.

        Parameters
        ----------
        q_or_k_or_v: The query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, H, L, D / H)

        Returns
        -------
        q_or_k_or_v: The transposed query, key, or value tensor.
            type:  torch.Tensor
            shape: (B, L, D)
        """
        # fmt: off
        # 1. q_or_k_or_v is shape (B, H, L, D / H = d_qkv)
        # 2. aim to make it of shape (B, L, H, D / H = d_qkv)
        q_or_k_or_v = q_or_k_or_v.permute(0, 2, 1, 3)

        # 3. Merge H and d_qkv into D
        batch_size, seq_len, _, _ = q_or_k_or_v.shape
        q_or_k_or_v = q_or_k_or_v.contiguous().view(batch_size, seq_len, self.d_model)
        # fmt: on
        return q_or_k_or_v

if __name__ == "__main__":
    B, H, L, D = 4, 8, 32, 512  # batch size, head, context length, embedding dimension
    dropout, bias = 0.0, False
    attention = ScaledDotProductAttention(dropout=dropout)

    mha = MultiHeadedAttention(attention=attention, H=H, d_model=D, dropout=dropout, bias=bias)
    z = torch.rand(B, L, D)
    output = mha(query=z, key=z, value=z, mask = None)
    assert output.shape == (B, L, D)
