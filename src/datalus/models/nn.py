"""Neural components for DATALUS tabular diffusion."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer diffusion timesteps with sinusoidal features."""

    def __init__(self, dim: int) -> None:
        """Initialize the embedding with the requested output dimension."""

        super().__init__()
        if dim < 2:
            raise ValueError("Time embedding dimension must be at least 2.")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Encode timesteps into a sinusoidal feature vector."""

        half_dim = self.dim // 2
        scale = math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings


class ResidualMLPBlock(nn.Module):
    """Residual denoising block with timestep injection."""

    def __init__(self, d_in: int, d_out: int, time_emb_dim: int, dropout: float) -> None:
        """Initialize the residual block with a matching residual projection."""

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
        """Apply the residual transformation with a timestep injection."""

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
        """Initialize the denoiser with time and optional context projections."""

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
            ResidualMLPBlock(left, right, time_emb_dim, dropout) for left, right in pairwise(hidden_dims)
        )
        self.final = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dims[-1], d_in),
        )
        nn.init.zeros_(self.final[-1].weight)
        nn.init.zeros_(self.final[-1].bias)

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> Tensor:
        """Predict the noise for a latent vector given a timestep and context."""

        return self.forward_with_hidden(x, t, c)[0]

    def forward_with_hidden(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Predict noise and the pre-readout hidden for categorical logit heads."""

        t_emb = self.time_embed(t)
        if self.context_proj is not None:
            if c is None:
                c = torch.zeros((x.shape[0], self.context_dim), device=x.device, dtype=x.dtype)
            t_emb = t_emb + self.context_proj(c.to(dtype=x.dtype))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.final[0](h)
        h = self.final[1](h)
        return self.final[2](h), h


class FeatureProjector(nn.Module):
    """Project encoded numerical and categorical columns into one latent vector."""

    def __init__(
        self,
        schema_metadata: dict[str, dict[str, Any]],
        numerical_columns: list[str],
        categorical_columns: list[str],
        embedding_rule: str = "log2",
    ) -> None:
        """Build embeddings for categorical columns and compute the latent size."""

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

    def to_browser_config(self) -> dict[str, Any]:
        """Serialize the latent layout and embeddings needed by ONNX Runtime Web."""

        # Browser inference decodes categorical slices from these embeddings.
        return {
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "cat_dims": [
                {"cardinality": cardinality, "embedding_dim": embedding_dim}
                for cardinality, embedding_dim in self.cat_dims
            ],
            "latent_dim": self.total_latent_dim,
            "embeddings": [embedding.weight.detach().cpu().float().tolist() for embedding in self.embeddings],
            "augmented_columns": [
                column
                for column in self.numerical_columns
                if self.schema_metadata.get(column, {}).get("augmented", False)
            ],
        }

    def forward(self, x_num: Tensor | None, x_cat: Tensor | None) -> Tensor:
        """Project numerical and categorical tensors into one latent vector."""

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
        for embedding, (_, emb_dim) in zip(self.embeddings, self.cat_dims, strict=False):
            chunk = latent[:, current : current + emb_dim]
            current += emb_dim
            distances = torch.cdist(chunk.float(), embedding.weight.float())
            decoded.append(distances.argmin(dim=1))
        return torch.stack(decoded, dim=1)

    def split_numerical(self, latent: Tensor) -> Tensor | None:
        """Return the leading numerical slice of a latent vector."""

        if not self.num_dim:
            return None
        return latent[:, : self.num_dim]


class CategoricalHeadedDenoiser(nn.Module):
    """Categorical logit heads for the TabDDPM composite objective.

    Wraps a base denoiser and returns ``{"noise": ..., "cat_logits": [...]}``
    without changing the noise path.
    """

    def __init__(self, base: nn.Module, cat_cardinalities: list[int], hidden_dim: int) -> None:
        """Wrap a base denoiser and attach one logit head per categorical column."""

        super().__init__()
        if not cat_cardinalities:
            raise ValueError("cat_cardinalities must not be empty.")
        self.base = base
        self.hidden_dim = hidden_dim
        self.cat_heads = nn.ModuleList(
            nn.Linear(hidden_dim, cardinality) for cardinality in cat_cardinalities
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor | None = None) -> dict[str, Any]:
        """Predict noise and per-column categorical logits."""

        noise, hidden = self.base.forward_with_hidden(x, t, c)
        return {
            "noise": noise,
            "cat_logits": [head(hidden) for head in self.cat_heads],
        }


def build_denoiser(
    d_in: int,
    num_dim: int,
    cat_dims: list[tuple[int, int]],
    config: dict[str, Any],
    context_dim: int | None = None,
) -> nn.Module:
    """Build a denoiser from a serialized model configuration.

    Mirrors ``TrainingConfig.model_dump()`` so trainer and bundle rebuild
    identical architectures. MLP is the default; the base denoiser is wrapped
    with logit heads when ``lambda_cat > 0`` and categorical columns exist.
    """

    hidden_dims = tuple(int(dim) for dim in config.get("hidden_dims", (512, 1024, 1024, 512)))
    denoiser_type = config.get("denoiser_type", "mlp")
    lambda_cat = float(config.get("lambda_cat", 0.0))
    if denoiser_type == "transformer":
        from datalus.models.transformer import TabularTransformerDenoiser

        d_model = int(config.get("transformer_d_model", 256))
        num_cls = int(config.get("transformer_num_cls", 2))
        base = TabularTransformerDenoiser(
            d_in=d_in,
            num_dim=num_dim,
            cat_dims=cat_dims,
            context_dim=context_dim,
            d_model=d_model,
            num_blocks=int(config.get("transformer_blocks", 4)),
            nhead=int(config.get("transformer_heads", 8)),
            dim_ff=d_model * int(config.get("transformer_ff_factor", 2)),
            num_inds=config.get("transformer_num_inds", 16),
            num_cls=num_cls,
            rope_base=config.get("transformer_rope_base"),
            ffn_chunk_size=config.get("transformer_ffn_chunk_size"),
            row_chunk_size=config.get("transformer_row_chunk_size"),
        )
        hidden_dim = (num_cls + 1) * d_model
    elif denoiser_type == "mlp":
        base = TabularDenoiserMLP(d_in=d_in, hidden_dims=hidden_dims, context_dim=context_dim)
        hidden_dim = hidden_dims[-1]
    else:
        raise ValueError(f"Unsupported denoiser type: {denoiser_type}")
    cat_cardinalities = [cardinality for cardinality, _ in cat_dims]
    if lambda_cat > 0 and cat_cardinalities:
        return CategoricalHeadedDenoiser(base, cat_cardinalities, hidden_dim)
    return base


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        """Snapshot trainable parameters as the initial shadow weights."""

        self.decay = decay
        self.shadow = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model parameters into the exponential moving average."""

        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Return the decay and shadow weights for persistence."""

        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the decay and shadow weights from a state dict."""

        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the shadow weights into the model parameters."""

        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
