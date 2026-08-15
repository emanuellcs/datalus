# Adapted from TabFM (https://github.com/google-research/tabfm),
# Copyright 2026 Google LLC, Apache License 2.0.
"""TabFM-inspired attention denoiser for tabular latent vectors.

Ports TabFM's transformer blocks (RMSNorm, SwiGLU, RoPE, per-dimension scale,
induced set attention) while preserving the residual MLP denoiser's
``(x_t, timestep, context) -> predicted_noise`` contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from datalus.models.nn import SinusoidalTimeEmbedding


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the scale weight and epsilon."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize along the last axis in float32 and cast back."""

        dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(-1, keepdim=True)
        return ((xf * torch.rsqrt(variance + self.eps)) * self.weight.float()).to(dtype)


class RoPE(nn.Module):
    """Interleaved rotary position embeddings over a sequence axis."""

    def __init__(self, dim: int, base: float = 100_000.0) -> None:
        """Precompute the inverse-frequency buffer for the given head dim."""

        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even.")
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", inv)

    def rotate(self, x: Tensor) -> Tensor:
        """Rotate x of shape [..., T, D] over its sequence axis."""

        t = x.shape[-2]
        angles = torch.outer(torch.arange(t, device=x.device).float(), self.freqs)
        cos = angles.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
        sin = angles.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-x2, x1), -1).reshape_as(x)
        return x * cos + rotated * sin


class MultiheadAttention(nn.Module):
    """Attention with q/k RMSNorm and per-dimension scale (TabFM-style)."""

    def __init__(self, d_model: int, nhead: int, rope: RoPE | None = None) -> None:
        """Build projections, norms, and the optional shared RoPE module."""

        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead.")
        self.nhead = nhead
        self.hd = d_model // nhead
        self.rope = rope
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.query_ln = RMSNorm(self.hd)
        self.key_ln = RMSNorm(self.hd)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.hd))

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Apply multi-head attention with RoPE, q/k norm, and per-dim scale."""

        batch, tokens_q, d_model = query.shape
        q = self.q_proj(query).view(batch, tokens_q, self.nhead, self.hd)
        k = self.k_proj(key).view(batch, key.shape[1], self.nhead, self.hd)
        v = self.v_proj(value).view(batch, value.shape[1], self.nhead, self.hd)
        if self.rope is not None:
            q = self.rope.rotate(q)
            k = self.rope.rotate(k)
        q = self.query_ln(q)
        k = self.key_ln(k)
        scale = 1.442695041 / math.sqrt(self.hd) * F.softplus(self.per_dim_scale.float())
        q = q * scale.to(q.dtype)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens_q, d_model))


class MultiheadAttentionBlock(nn.Module):
    """Pre/post-norm attention block with a SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.0,
        rope: RoPE | None = None,
    ) -> None:
        """Build attention, residual norms, and the SwiGLU FFN."""

        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, rope)
        self.pre_attn_ln = RMSNorm(d_model)
        self.post_attn_ln = RMSNorm(d_model)
        self.pre_ff_ln = RMSNorm(d_model)
        self.post_ff_ln = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear1_gate = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn_chunk_size: int | None = None

    def _ff_impl(self, x: Tensor) -> Tensor:
        """Apply the SwiGLU feed-forward network to one token slice."""

        x_norm = self.pre_ff_ln(x)
        hidden = F.silu(self.linear1_gate(x_norm)) * self.linear1(x_norm)
        return self.post_ff_ln(self.linear2(self.dropout(hidden)))

    def _ff(self, x: Tensor) -> Tensor:
        """Apply the FFN, chunking the expanded activation over tokens."""

        if self.ffn_chunk_size is None:
            return self._ff_impl(x)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        out = torch.empty(
            flat.shape[0],
            self.linear2.out_features,
            dtype=flat.dtype,
            device=flat.device,
        )
        for start in range(0, flat.shape[0], self.ffn_chunk_size):
            out[start : start + self.ffn_chunk_size] = self._ff_impl(
                flat[start : start + self.ffn_chunk_size]
            )
        return out.reshape(shape)

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply self- or cross-attention followed by the SwiGLU FFN."""

        k = q if k is None else k
        v = q if v is None else v
        q_norm = self.pre_attn_ln(q)
        k_norm = self.pre_attn_ln(k)
        v_norm = self.pre_attn_ln(v)
        attended = self.attn(q_norm, k_norm, v_norm, attn_mask)
        h = q + self.post_attn_ln(attended)
        return h + self._ff(h)


class InducedSelfAttentionBlock(nn.Module):
    """Induced set-attention block (ISAB) with a learned memory bottleneck."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_inds: int,
        dropout: float = 0.0,
    ) -> None:
        """Build the inducing-point memory and the two cross-attention MABs."""

        super().__init__()
        self.ind_vectors = nn.Parameter(torch.zeros(num_inds, d_model))
        nn.init.normal_(self.ind_vectors, std=0.02)
        self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, dropout)

    def forward(self, src: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Summarize the source into inducing points and attend back to them."""

        ind = self.ind_vectors.unsqueeze(0).expand(src.shape[0], -1, -1)
        hidden = self.mab1(ind, src, src, attn_mask)
        return self.mab2(src, hidden, hidden)


class SetTransformer(nn.Module):
    """Stacked induced self-attention blocks (ISAB)."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_ff: int,
        num_inds: int,
        dropout: float = 0.0,
    ) -> None:
        """Build the requested number of induced-attention blocks."""

        super().__init__()
        self.blocks = nn.ModuleList(
            InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, dropout)
            for _ in range(num_blocks)
        )

    def forward(self, src: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run the stacked induced-attention blocks."""

        for block in self.blocks:
            src = block(src, attn_mask=attn_mask)
        return src


class TransformerEncoder(nn.Module):
    """Stacked attention blocks sharing one optional RoPE module."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.0,
        rope_base: float | None = None,
    ) -> None:
        """Build the shared RoPE and the stacked attention blocks."""

        super().__init__()
        self.rope = RoPE(d_model // nhead, rope_base) if rope_base is not None else None
        self.blocks = nn.ModuleList(
            MultiheadAttentionBlock(d_model, nhead, dim_ff, dropout, self.rope)
            for _ in range(num_blocks)
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run the stacked attention blocks."""

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x


class TabularTransformerDenoiser(nn.Module):
    """Attention-based epsilon predictor for tabular latent vectors.

    Tokenizes each latent feature, mixes the tokens with attention, and reads
    out the noise from learned CLS tokens. RoPE is opt-in (``rope_base``)
    because feature order is arbitrary for tables.
    """

    def __init__(
        self,
        d_in: int,
        num_dim: int,
        cat_dims: list[tuple[int, int]] | None = None,
        dim_t: int = 128,
        d_model: int = 256,
        num_blocks: int = 4,
        nhead: int = 8,
        dim_ff: int | None = None,
        num_inds: int | None = 16,
        num_cls: int = 2,
        dropout: float = 0.1,
        context_dim: int | None = None,
        rope_base: float | None = None,
        ffn_chunk_size: int | None = None,
        row_chunk_size: int | None = None,
    ) -> None:
        """Build token projections, attention stages, and the readout head."""

        super().__init__()
        if num_dim < 0:
            raise ValueError("num_dim must be non-negative.")
        self.d_in = d_in
        self.num_dim = num_dim
        self.cat_dims = list(cat_dims or [])
        self.num_cls = num_cls
        self.context_dim = context_dim
        time_emb_dim = dim_t * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim_t),
            nn.Linear(dim_t, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.context_proj = (
            nn.Sequential(
                nn.Linear(context_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
            if context_dim is not None
            else None
        )
        self.num_proj = nn.Linear(1, d_model) if num_dim > 0 else None
        self.cat_projs = nn.ModuleList(
            nn.Linear(emb_dim, d_model) for _, emb_dim in self.cat_dims
        )
        self.time_token_proj = nn.Linear(time_emb_dim, d_model)
        self.cls_tokens = nn.Parameter(torch.zeros(num_cls, d_model))
        nn.init.normal_(self.cls_tokens, std=0.02)
        dim_ff = dim_ff or d_model * 2
        self.ind_stage = (
            SetTransformer(1, d_model, nhead, dim_ff, num_inds, dropout)
            if num_inds is not None
            else None
        )
        self.encoder = TransformerEncoder(
            num_blocks,
            d_model,
            nhead,
            dim_ff,
            dropout,
            rope_base,
        )
        self.head = nn.Linear((num_cls + 1) * d_model, d_in)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.row_chunk_size = row_chunk_size
        if ffn_chunk_size is not None:
            for module in self.modules():
                if hasattr(module, "ffn_chunk_size"):
                    module.ffn_chunk_size = ffn_chunk_size

    def _tokenize(self, x: Tensor) -> Tensor:
        """Split the flat latent into per-feature tokens of shape [B, T, d_model]."""

        parts: list[Tensor] = []
        if self.num_proj is not None:
            parts.append(self.num_proj(x[:, : self.num_dim].unsqueeze(-1)))
        cursor = self.num_dim
        for idx, (_, emb_dim) in enumerate(self.cat_dims):
            parts.append(self.cat_projs[idx](x[:, cursor : cursor + emb_dim])[:, None, :])
            cursor += emb_dim
        return torch.cat(parts, dim=1)

    def _run_encoder(self, tokens: Tensor) -> Tensor:
        """Apply the encoder with optional exact chunking over the batch axis."""

        if self.row_chunk_size is None or tokens.shape[0] <= self.row_chunk_size:
            return self.encoder(tokens)
        parts = [
            self.encoder(tokens[start : start + self.row_chunk_size])
            for start in range(0, tokens.shape[0], self.row_chunk_size)
        ]
        return torch.cat(parts, dim=0)

    def forward_with_hidden(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Predict noise and return the pre-readout hidden representation."""

        t_emb = self.time_embed(t)
        if self.context_proj is not None:
            if c is None:
                c = torch.zeros((x.shape[0], self.context_dim), device=x.device, dtype=x.dtype)
            t_emb = t_emb + self.context_proj(c.to(dtype=x.dtype))
        features = self._tokenize(x)
        time_token = self.time_token_proj(t_emb)[:, None, :]
        cls = self.cls_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
        tokens = torch.cat([time_token, cls, features], dim=1)
        if self.ind_stage is not None:
            tokens = self.ind_stage(tokens)
        tokens = self._run_encoder(tokens)
        hidden = tokens[:, : self.num_cls + 1].reshape(x.shape[0], (self.num_cls + 1) * tokens.shape[-1])
        return self.head(hidden), hidden

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        """Predict the noise for a latent vector given a timestep and context."""

        return self.forward_with_hidden(x, t, c)[0]
