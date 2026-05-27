import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable, Optional, Union

class SlotInitializer(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Parameters for Gaussian init, shared by all slots.
        self.slots_mu = nn.Parameter(torch.empty(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, slot_dim))

        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

    def forward(self, batch_size: int) -> Tensor:
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = torch.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        return slots

class SlotAttentionLayer(nn.Module):
    """
    PyTorch conversion of the official Google Research SlotAttention layer.

    Args:
        num_iterations: Number of attention refinement iterations.
        slot_dim: Dimensionality of slot feature vectors.
        mlp_hidden_dim: Hidden layer size of the per-slot MLP.
        epsilon: Offset for attention coefficients before normalization.

    Input:
        feats: [B, N, slot_dim]
        slots: [B, num_slots, slot_dim]

    Output:
        slots: [B, num_slots, slot_dim]
    """

    def __init__(
        self,
        num_iterations: int,
        slot_dim: int,
        mlp_hidden_dim: int,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.slot_dim = slot_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Linear maps for attention.
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

    def forward(self, feats: Tensor, slots: Tensor) -> Tensor:
        """
        Args:
            feats: [B, N, slot_dim]
            slots: [B, num_slots, slot_dim]

        Returns:
            slots: [B, num_slots, slot_dim]
        """
        assert feats.ndim == 3, f"Expected feats with shape [B, N, D], got {feats.shape}"
        assert slots.ndim == 3, f"Expected slots with shape [B, num_slots, D], got {slots.shape}"

        _, _, slot_dim = slots.shape
        assert slot_dim == self.slot_dim, f"Expected slot_dim={self.slot_dim}, got {slot_dim}"

        b, _, c = feats.shape
        assert c == self.slot_dim, f"Expected feat_dim={self.slot_dim}, got {c}"

        # Apply layer norm to input.
        feats = self.norm_inputs(feats)

        # Shape: [B, N, slot_dim]
        k = self.project_k(feats)
        v = self.project_v(feats)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots

            slots = self.norm_slots(slots)

            # Shape: [B, num_slots, slot_dim]
            q = self.project_q(slots)
            q = q * (self.slot_dim ** -0.5)

            # attn_logits: [B, N, num_slots]
            attn_logits = torch.bmm(k, q.transpose(1, 2))

            # Softmax over slots.
            attn = F.softmax(attn_logits, dim=-1)

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)

            # updates: [B, num_slots, slot_dim]
            updates = torch.bmm(attn.transpose(1, 2), v)

            # GRUCell expects [B*K, D].
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )

            slots = slots.reshape(b, -1, self.slot_dim)

            # MLP residual.
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

class CrossAttnLayer(nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the cross-attn layer.

        Args:
            tgt: the sequence to the cross-attn layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
    
    def _get_activation_fn(self, activation: str) -> Callable[[Tensor], Tensor]:
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError(f"activation should be relu/gelu, not {activation}")
