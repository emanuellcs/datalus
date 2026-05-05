"""Neural components for DATALUS tabular diffusion."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer diffusion timesteps with sinusoidal features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("Time embedding dimension must be at least 2.")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half_dim = self.dim // 2
        scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            embeddings = torch.cat(
                [embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1
            )
        return embeddings


class ResidualMLPBlock(nn.Module):
    """Residual denoising block with timestep injection."""

    def __init__(
        self, d_in: int, d_out: int, time_emb_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.linear2 = nn.Linear(d_out, d_out)
        self.norm2 = nn.LayerNorm(d_out)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, d_out))
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.linear1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.norm2(h)
        return h + self.residual(x)


class TabularDenoiserMLP(nn.Module):
    """Residual MLP epsilon predictor for tabular latent vectors."""

    def __init__(
        self,
        d_in: int,
        dim_t: int = 128,
        hidden_dims: tuple[int, ...] = (256, 512, 512, 256),
        dropout: float = 0.1,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty.")
        self.d_in = d_in
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
        self.input_proj = nn.Linear(d_in, hidden_dims[0])
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(left, right, time_emb_dim, dropout)
            for left, right in zip(hidden_dims[:-1], hidden_dims[1:])
        )
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], d_in),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        t_emb = self.time_embed(t)
        if self.context_proj is not None:
            if c is None:
                c = torch.zeros(
                    (x.shape[0], self.context_dim), device=x.device, dtype=x.dtype
                )
            t_emb = t_emb + self.context_proj(c.to(dtype=x.dtype))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.final(h)


class FeatureProjector(nn.Module):
    """Project encoded numerical and categorical columns into one latent vector."""

    def __init__(
        self,
        schema_metadata: dict[str, dict[str, Any]],
        numerical_columns: list[str],
        categorical_columns: list[str],
        embedding_rule: str = "log2",
    ) -> None:
        super().__init__()
        self.schema_metadata = schema_metadata
        self.numerical_columns = list(numerical_columns)
        self.categorical_columns = list(categorical_columns)
        self.num_dim = len(self.numerical_columns)
        self.cat_dims: list[tuple[int, int]] = []
        self.embeddings = nn.ModuleList()
        for column in self.categorical_columns:
            cardinality = int(schema_metadata[column].get("cardinality") or 2)
            if embedding_rule == "fastai":
                emb_dim = min(50, max(2, int(1.6 * (cardinality**0.56))))
            else:
                emb_dim = max(2, math.ceil(math.log2(max(cardinality, 2))))
            self.cat_dims.append((cardinality, emb_dim))
            layer = nn.Embedding(cardinality, emb_dim)
            nn.init.xavier_uniform_(layer.weight)
            self.embeddings.append(layer)
        self.total_latent_dim = self.num_dim + sum(dim for _, dim in self.cat_dims)

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        parts: list[Tensor] = []
        if self.num_dim:
            if x_num is None:
                raise ValueError("Numerical tensor is required by this projector.")
            parts.append(x_num.float())
        if self.categorical_columns:
            if x_cat is None:
                raise ValueError("Categorical tensor is required by this projector.")
            for idx, embedding in enumerate(self.embeddings):
                values = x_cat[:, idx].long().clamp(0, embedding.num_embeddings - 1)
                parts.append(embedding(values))
        if not parts:
            raise ValueError("At least one active feature is required.")
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def nearest_category_indices(self, latent: Tensor) -> Tensor | None:
        """Decode categorical latent slices by nearest learned embedding."""

        if not self.categorical_columns:
            return None
        current = self.num_dim
        decoded: list[Tensor] = []
        for embedding, (_, emb_dim) in zip(self.embeddings, self.cat_dims):
            chunk = latent[:, current : current + emb_dim]
            current += emb_dim
            distances = torch.cdist(chunk.float(), embedding.weight.float())
            decoded.append(distances.argmin(dim=1))
        return torch.stack(decoded, dim=1)

    def split_numerical(self, latent: Tensor) -> Tensor | None:
        if not self.num_dim:
            return None
        return latent[:, : self.num_dim]


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(
                    self.shadow[name].to(device=param.device, dtype=param.dtype)
                )
