"""Tests for diffusion sampling, inpainting, and RePaint schedules."""

from itertools import pairwise

import torch

from datalus.config import RePaintConfig
from datalus.models.diffusion import TabularDiffusion
from datalus.models.nn import (
    CategoricalHeadedDenoiser,
    TabularDenoiserMLP,
    build_denoiser,
)
from datalus.models.schedules import make_repaint_schedule
from datalus.models.transformer import TabularTransformerDenoiser


def test_ddim_and_repaint_shapes_are_stable():
    """Sampling and inpainting preserve the latent shape and known values."""

    torch.manual_seed(0)
    denoiser = TabularDenoiserMLP(d_in=4, hidden_dims=(8, 8), dim_t=16)
    diffusion = TabularDiffusion(denoiser, num_timesteps=20)
    sample = diffusion.sample_ddim((3, 4), torch.device("cpu"), ddim_steps=5, seed=123)
    assert sample.shape == (3, 4)

    original = torch.randn(3, 4)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32).repeat(3, 1)
    inpainted = diffusion.inpaint_repaint(
        original,
        mask,
        RePaintConfig(num_inference_steps=8, jump_length=2, jump_n_sample=2),
        seed=123,
    )
    assert inpainted.shape == original.shape
    assert torch.allclose(inpainted[:, :2], original[:, :2], atol=1e-5)


def test_repaint_schedule_contains_forward_jumps():
    """The RePaint schedule includes forward jumps and ends at -1."""

    schedule = make_repaint_schedule(100, 20, jump_length=5, jump_n_sample=2)
    assert any(next_step > step for step, next_step in pairwise(schedule))
    assert schedule[-1] == -1


def _small_transformer_denoiser(d_in=4, num_dim=2, cat_dims=None, context_dim=None):
    return TabularTransformerDenoiser(
        d_in=d_in,
        num_dim=num_dim,
        cat_dims=cat_dims or [],
        dim_t=16,
        d_model=16,
        num_blocks=1,
        nhead=4,
        num_inds=4,
        num_cls=2,
        rope_base=None,
    )


def test_transformer_denoiser_predicts_noise_with_stable_shape():
    """The attention denoiser preserves the (x, t) -> noise contract."""

    torch.manual_seed(0)
    denoiser = _small_transformer_denoiser(d_in=4, num_dim=2, cat_dims=[(5, 2)])
    x = torch.randn(3, 4)
    t = torch.tensor([5, 10, 500])
    noise = denoiser(x, t)
    assert noise.shape == (3, 4)
    repeated = denoiser(x, t)
    assert torch.allclose(noise, repeated, atol=1e-6)


def test_transformer_denoiser_ddim_sampling_is_stable():
    """DDIM sampling through the transformer denoiser keeps the latent shape."""

    torch.manual_seed(1)
    denoiser = _small_transformer_denoiser(d_in=4, num_dim=2)
    diffusion = TabularDiffusion(denoiser, num_timesteps=20)
    sample = diffusion.sample_ddim((3, 4), torch.device("cpu"), ddim_steps=5, seed=9)
    assert sample.shape == (3, 4)
    assert torch.isfinite(sample).all()


def test_transformer_denoiser_without_rope_is_permutation_invariant():
    """Without RoPE, permuting feature tokens changes nothing when re-tokenized."""

    torch.manual_seed(2)
    denoiser = _small_transformer_denoiser(d_in=3, num_dim=3, context_dim=None)
    x = torch.randn(2, 3)
    t = torch.tensor([3, 7])
    base = denoiser(x, t)
    permuted_x = x[:, [2, 0, 1]]
    permuted = denoiser(permuted_x, t)
    assert torch.allclose(base[:, [2, 0, 1]], permuted, atol=1e-5)


def test_mlp_denoiser_forward_with_hidden_matches_noise_head():
    """forward_with_hidden returns the same noise as forward plus the hidden."""

    torch.manual_seed(3)
    denoiser = TabularDenoiserMLP(d_in=4, hidden_dims=(8, 8), dim_t=16)
    x = torch.randn(2, 4)
    t = torch.tensor([1, 2])
    noise, hidden = denoiser.forward_with_hidden(x, t)
    assert torch.allclose(noise, denoiser(x, t), atol=1e-6)
    assert hidden.shape == (2, 8)


def test_categorical_headed_denoiser_composite_loss():
    """The composite loss blends MSE and categorical cross-entropy."""

    torch.manual_seed(4)
    base = TabularDenoiserMLP(d_in=4, hidden_dims=(8, 8), dim_t=16)
    headed = CategoricalHeadedDenoiser(base, [5, 3], 8)
    diffusion = TabularDiffusion(
        headed,
        num_timesteps=20,
        num_dim=2,
        cat_dims=[(5, 1), (3, 1)],
        lambda_num=1.0,
        lambda_cat=0.5,
    )
    x_start = torch.randn(4, 4)
    x_cat = torch.tensor([[1, 0], [4, 2], [0, 1], [3, 2]], dtype=torch.long)
    metrics = diffusion.compute_loss(x_start, x_cat)
    assert "loss" in metrics and "mse" in metrics and "cat_ce" in metrics
    assert metrics["loss"].ndim == 0
    assert metrics["cat_ce"].item() >= 0


def test_categorical_heads_do_not_change_plain_mse_path():
    """With lambda_cat=0 the diffusion reports only the MSE objective."""

    torch.manual_seed(5)
    headed = CategoricalHeadedDenoiser(
        TabularDenoiserMLP(d_in=4, hidden_dims=(8, 8), dim_t=16),
        [5],
        8,
    )
    diffusion = TabularDiffusion(headed, num_timesteps=20, lambda_cat=0.0)
    x_start = torch.randn(4, 4)
    metrics = diffusion.compute_loss(x_start)
    assert set(metrics) == {"loss", "mse"}


def test_predict_noise_cfg_extracts_dict_output():
    """CFG noise prediction works with dict-output denoisers."""

    torch.manual_seed(6)
    base = TabularDenoiserMLP(d_in=4, hidden_dims=(8, 8), dim_t=16, context_dim=3)
    headed = CategoricalHeadedDenoiser(base, [5], 8)
    diffusion = TabularDiffusion(headed, num_timesteps=20)
    x = torch.randn(2, 4)
    t = torch.tensor([4, 9])
    context = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    guided = diffusion.predict_noise_cfg(x, t, context, cfg_scale=2.0)
    assert guided.shape == (2, 4)
    unguided = diffusion.predict_noise_cfg(x, t, None, cfg_scale=2.0)
    assert unguided.shape == (2, 4)


def test_build_denoiser_roundtrips_state_dict():
    """build_denoiser reconstructs the same architecture from a config dict."""

    torch.manual_seed(7)
    config = {
        "denoiser_type": "transformer",
        "hidden_dims": (16, 16),
        "lambda_cat": 1.0,
        "transformer_d_model": 16,
        "transformer_blocks": 1,
        "transformer_heads": 4,
        "transformer_ff_factor": 2,
        "transformer_num_inds": 4,
        "transformer_num_cls": 2,
        "transformer_rope_base": None,
        "transformer_ffn_chunk_size": None,
        "transformer_row_chunk_size": None,
    }
    first = build_denoiser(4, 2, [(5, 1), (3, 1)], config, context_dim=2)
    second = build_denoiser(4, 2, [(5, 1), (3, 1)], config, context_dim=2)
    state = first.state_dict()
    second.load_state_dict(state)
    x = torch.randn(2, 4)
    t = torch.tensor([1, 2])
    assert torch.allclose(first(x, t)["noise"], second(x, t)["noise"], atol=1e-6)


def test_build_denoiser_defaults_to_mlp_for_legacy_configs():
    """Checkpoints without denoiser_type reconstruct the MLP architecture."""

    denoiser = build_denoiser(4, 0, [], {"hidden_dims": (8, 8)})
    assert isinstance(denoiser, TabularDenoiserMLP)
