import torch

from datalus.diffusion import RePaintConfig, TabularDiffusion, make_repaint_schedule
from datalus.nn import TabularDenoiserMLP


def test_ddim_and_repaint_shapes_are_stable():
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
    schedule = make_repaint_schedule(100, 20, jump_length=5, jump_n_sample=2)
    assert any(next_step > step for step, next_step in zip(schedule, schedule[1:]))
    assert schedule[-1] == -1
