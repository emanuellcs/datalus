"""Pydantic configuration and contract models shared across DATALUS."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseConfig(BaseModel):
    """Base contract that forbids extra attributes and exposes JSON serialization."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for artifact manifests."""

        return self.model_dump(mode="json")


class ColumnProfile(BaseConfig):
    """Domain description of one column after topology inference."""

    column_name: str
    original_dtype: str
    inferred_topology: str
    encoding_strategy: str
    cardinality: int | None = None
    null_ratio: float | None = None
    category_frequencies: dict[str, int] | None = None
    rare_category_count: int | None = None
    rare_category_threshold: int | None = None
    rare_categories_preserved: bool = True
    is_target: bool = False
    retained: bool = True
    reason: str | None = None
    augmented: bool = False


class RePaintConfig(BaseConfig):
    """Inference policy for RePaint-style tabular inpainting."""

    num_inference_steps: int = Field(default=250, ge=1)
    jump_length: int = Field(default=10, ge=1)
    jump_n_sample: int = Field(default=10, ge=1)
    eta: float = Field(default=0.0, ge=0.0)


class PrivacyThresholds(BaseConfig):
    """Approval thresholds for empirical privacy evidence."""

    memorization_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    mia_roc_auc: float = Field(default=0.55, ge=0.0, le=1.0)
    dcr_percentile: float = Field(default=1.0, ge=0.0, le=100.0)


class ShadowMIAConfig(BaseConfig):
    """Domain parameters for the black-box shadow-model MIA protocol."""

    mode: Literal["release", "ci_lite"] = "release"
    n_shadow_models: int = Field(default=4, ge=1)
    shadow_train_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    synthetic_multiplier: float = Field(default=1.0, gt=0.0)
    n_neighbors: int = Field(default=5, ge=1)
    max_rows: int | None = Field(default=None, ge=8)
    attack_estimators: int = Field(default=200, ge=10)
    attack_max_depth: int | None = Field(default=8, ge=1)
    attack_min_samples_leaf: int = Field(default=3, ge=1)
    random_state: int = 42


class GenerationConfig(BaseConfig):
    """User-facing generation policy shared by CLI and HTTP interfaces."""

    mode: Literal[
        "ab_initio",
        "augmentation",
        "balancing",
        "inpainting",
        "counterfactual",
    ] = "ab_initio"
    n_records: int = Field(default=100, ge=1)
    ddim_steps: int = Field(default=50, ge=1)
    seed: int = 42
    cfg_scale: float = Field(default=1.0, ge=0.0)
    conditions: dict[str, Any] = Field(default_factory=dict)


class BalancingConfig(GenerationConfig):
    """Policy for selective synthetic generation of underrepresented classes."""

    mode: Literal["balancing"] = "balancing"
    target_column: str
    target_distribution: dict[str, int] = Field(default_factory=dict)
    max_attempts: int = Field(default=10, ge=1)
    strict: bool = False


class TrainingConfig(BaseConfig):
    """Use-case configuration for Colab-safe TabDDPM training."""

    schema_path: str
    data_path: str
    output_dir: str
    batch_size: int = Field(default=2_048, ge=1)
    epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    checkpoint_every_steps: int = Field(
        default=500,
        ge=0,
        description="Save a checkpoint every N training steps (0 disables step cadence)",
    )
    seed: int = 42
    num_timesteps: int = Field(default=1_000, ge=1)
    hidden_dims: tuple[int, ...] = (512, 1024, 1024, 512)
    amp: bool = True
    condition_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    ema_decay: float = Field(default=0.9999, gt=0.0, lt=1.0)
    warmup_steps: int = Field(default=500, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    max_encoder_fit_rows: int = Field(default=100_000, ge=1)
    gpu: str | None = Field(default=None, description="CUDA device indices, e.g., '0' or '0,1'")
    keep_last: int = Field(
        default=3,
        ge=0,
        description="Maximum number of recent step checkpoints to keep (0 = keep all)",
    )
    save_every: int = Field(
        default=1,
        ge=1,
        description="Save a checkpoint every N epochs, at epoch boundaries",
    )
    min_free_space_gb: float = Field(
        default=2.0,
        ge=0.0,
        description="Refuse checkpoint writes below this free-space threshold in GiB (0 disables)",
    )
    save_strategy: Literal["all", "latest", "best"] = Field(
        default="latest",
        description="Checkpoint strategy: 'all' (keep all), 'latest' (rolling), 'best' (track lowest loss)",
    )
    # TabFM-style encoding policy.
    quantile_noise: float = Field(
        default=0.0,
        ge=0.0,
        description="RTDL-style relative Gaussian noise added to numeric fit samples",
    )
    rtdl_quantile_dynamic: bool = Field(
        default=False,
        description="Dynamic quantile count min(n_samples // 30, n_quantiles) with floor 10",
    )
    outlier_threshold: float | None = Field(
        default=None,
        gt=0.0,
        description="Two-stage robust z-score clipping threshold for numeric columns (None = disabled)",
    )
    numeric_standardize: bool = Field(
        default=False,
        description="Standardize quantile-encoded numeric values with clipping",
    )
    cat_encoder_mode: Literal["alphabetical", "appearance", "frequency"] = "alphabetical"
    min_cat_frequency: int = Field(
        default=1,
        ge=1,
        description="Collapse categories below this frequency to __UNKNOWN__ "
        "(1 = preserve all observed categories)",
    )
    # Denoiser selection.
    denoiser_type: Literal["mlp", "transformer"] = Field(
        default="mlp",
        description="Denoiser architecture: 'mlp' (residual MLP) or 'transformer' (attention)",
    )
    transformer_d_model: int = Field(default=256, ge=16)
    transformer_blocks: int = Field(default=4, ge=1)
    transformer_heads: int = Field(default=8, ge=1)
    transformer_ff_factor: int = Field(default=2, ge=1)
    transformer_num_inds: int | None = Field(
        default=16,
        ge=1,
        description="Induced set-attention memory size (None = skip the induced stage)",
    )
    transformer_num_cls: int = Field(default=2, ge=1)
    transformer_rope_base: float | None = Field(
        default=None,
        ge=0.0,
        description="RoPE base frequency; None disables RoPE for permutation-invariant columns",
    )
    transformer_ffn_chunk_size: int | None = Field(default=8192, ge=1)
    transformer_row_chunk_size: int | None = Field(default=4096, ge=1)
    # Composite loss and conditioning.
    target_column: str | None = Field(
        default=None,
        description="Optional target column for class-conditional (CFG) training",
    )
    lambda_num: float = Field(default=1.0, ge=0.0, description="Weight of the numerical MSE loss")
    lambda_cat: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the categorical cross-entropy loss (0 = pure MSE)",
    )
    # Training-time feature augmentation.
    feature_augmentation: Literal["none", "crosses", "svd"] = Field(
        default="none",
        description="Opt-in training-time feature augmentation: pairwise crosses or SVD structural features",
    )
