"""Diffusion schedules, DDIM sampling, RePaint inpainting, and CFG utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VarianceSchedule(nn.Module):
    """Precompute stable diffusion schedule tensors."""

    def __init__(
        self, num_timesteps: int = 1_000, schedule_type: str = "cosine"
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        if schedule_type == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp(min=1e-8, max=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt((1.0 - alphas_cumprod).clamp(min=1e-8)),
        )

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(1e-5, 0.999)

    def extract(self, values: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        out = values.gather(0, t.detach().cpu()).to(device=t.device)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def alpha_bar_at(self, timestep: int | Tensor, device: torch.device) -> Tensor:
        if isinstance(timestep, Tensor):
            return self.alphas_cumprod[timestep].to(device)
        if timestep < 0:
            return torch.tensor(1.0, device=device, dtype=self.alphas_cumprod.dtype)
        return self.alphas_cumprod[timestep].to(device)


@dataclass(slots=True)
class RePaintConfig:
    """Parameters for tabular RePaint resampling."""

    num_inference_steps: int = 250
    jump_length: int = 10
    jump_n_sample: int = 10
    eta: float = 0.0


class TabularDiffusion(nn.Module):
    """Continuous latent-space diffusion for heterogeneous tabular records."""

    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 1_000,
        schedule_type: str = "cosine",
        condition_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.condition_dropout = condition_dropout
        self.schedule = VarianceSchedule(num_timesteps, schedule_type)

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.schedule.extract(
            self.schedule.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus = self.schedule.extract(
            self.schedule.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def compute_loss(
        self, x_start: Tensor, context: Tensor | None = None
    ) -> dict[str, Tensor]:
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        if context is not None and self.condition_dropout > 0:
            keep = torch.rand((batch_size, 1), device=device) >= self.condition_dropout
            context = torch.where(keep, context, torch.zeros_like(context))
        predicted_noise = self.denoiser(x_noisy, t, context)
        loss = F.mse_loss(predicted_noise, noise)
        return {"loss": loss, "mse": loss.detach()}

    def forward(
        self, x_start: Tensor, context: Tensor | None = None
    ) -> dict[str, Tensor]:
        return self.compute_loss(x_start, context)

    @torch.no_grad()
    def predict_noise_cfg(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor | None,
        cfg_scale: float = 1.0,
        group_guidance: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        if context is None or cfg_scale == 1.0:
            return self.denoiser(x, t, context)
        uncond = self.denoiser(x, t, torch.zeros_like(context))
        if group_guidance:
            guided = uncond
            for group in group_guidance:
                mask = group["mask"].to(device=context.device, dtype=context.dtype)
                scale = float(group["scale"])
                group_context = context * mask
                cond_group = self.denoiser(x, t, group_context)
                guided = guided + scale * (cond_group - uncond)
            return guided
        cond = self.denoiser(x, t, context)
        return uncond + cfg_scale * (cond - uncond)

    @torch.no_grad()
    def ddim_step(
        self,
        x: Tensor,
        t_value: int,
        prev_t_value: int,
        context: Tensor | None = None,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        group_guidance: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        batch_size = x.shape[0]
        device = x.device
        t = torch.full((batch_size,), t_value, device=device, dtype=torch.long)
        pred_noise = self.predict_noise_cfg(x, t, context, cfg_scale, group_guidance)
        alpha_t = self.schedule.alpha_bar_at(t_value, device).clamp(min=1e-8)
        alpha_prev = self.schedule.alpha_bar_at(prev_t_value, device).clamp(min=1e-8)
        pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        if eta > 0 and prev_t_value >= 0:
            variance = (
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            ).clamp(min=0)
            sigma = eta * torch.sqrt(variance)
            noise = torch.randn_like(x)
        else:
            sigma = torch.tensor(0.0, device=device, dtype=x.dtype)
            noise = torch.zeros_like(x)
        direction = (
            torch.sqrt((1.0 - alpha_prev - sigma**2).clamp(min=0.0)) * pred_noise
        )
        return torch.sqrt(alpha_prev) * pred_x0 + direction + sigma * noise

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        context: Tensor | None = None,
        cfg_scale: float = 1.0,
        seed: int | None = None,
        eta: float = 0.0,
        group_guidance: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, device=device, generator=generator)
        timesteps = make_ddim_timesteps(self.num_timesteps, ddim_steps)
        for idx, t_value in enumerate(timesteps):
            prev_t = timesteps[idx + 1] if idx < len(timesteps) - 1 else -1
            x = self.ddim_step(
                x, int(t_value), int(prev_t), context, cfg_scale, eta, group_guidance
            )
        return x

    @torch.no_grad()
    def inpaint_repaint(
        self,
        original_latent: Tensor,
        known_mask: Tensor,
        config: RePaintConfig | None = None,
        context: Tensor | None = None,
        cfg_scale: float = 1.0,
        seed: int | None = None,
    ) -> Tensor:
        """Apply RePaint-style inference-time conditioning to tabular masks."""

        repaint = config or RePaintConfig()
        device = original_latent.device
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        schedule = make_repaint_schedule(
            self.num_timesteps,
            repaint.num_inference_steps,
            repaint.jump_length,
            repaint.jump_n_sample,
        )
        x = torch.randn_like(original_latent)
        mask = known_mask.to(device=device, dtype=original_latent.dtype)
        for idx, t_value in enumerate(schedule[:-1]):
            next_t = int(schedule[idx + 1])
            t_value = int(t_value)
            if next_t > t_value:
                x = self.undo_step(x, t_value, next_t)
                continue
            unknown = self.ddim_step(
                x, t_value, next_t, context, cfg_scale, repaint.eta
            )
            if next_t >= 0:
                next_batch = torch.full(
                    (original_latent.shape[0],),
                    next_t,
                    device=device,
                    dtype=torch.long,
                )
                known = self.q_sample(original_latent, next_batch)
            else:
                known = original_latent
            x = mask * known + (1.0 - mask) * unknown
        return x

    @torch.no_grad()
    def counterfactual_sample(
        self,
        original_latent: Tensor,
        t_star: int,
        counterfactual_context: Tensor,
        cfg_scale: float = 2.0,
        ddim_steps: int = 50,
        seed: int | None = None,
        group_guidance: list[dict[str, Any]] | None = None,
    ) -> Tensor:
        """Partially corrupt a record and denoise under an intervention context."""

        device = original_latent.device
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                original_latent.shape, device=device, generator=generator
            )
        else:
            noise = torch.randn_like(original_latent)
        t = torch.full(
            (original_latent.shape[0],), t_star, device=device, dtype=torch.long
        )
        x = self.q_sample(original_latent, t, noise)
        timesteps = [
            step
            for step in make_ddim_timesteps(self.num_timesteps, ddim_steps)
            if step <= t_star
        ]
        if not timesteps or timesteps[0] != t_star:
            timesteps.insert(0, t_star)
        for idx, t_value in enumerate(timesteps):
            prev_t = timesteps[idx + 1] if idx < len(timesteps) - 1 else -1
            x = self.ddim_step(
                x,
                int(t_value),
                int(prev_t),
                counterfactual_context,
                cfg_scale,
                group_guidance=group_guidance,
            )
        return x

    def undo_step(self, sample: Tensor, from_t: int, to_t: int) -> Tensor:
        """Jump forward by reintroducing noise between two timesteps."""

        if to_t <= from_t:
            return sample
        device = sample.device
        alpha_from = self.schedule.alpha_bar_at(from_t, device).clamp(min=1e-8)
        alpha_to = self.schedule.alpha_bar_at(to_t, device).clamp(min=1e-8)
        ratio = (alpha_to / alpha_from).clamp(min=1e-8, max=1.0)
        return torch.sqrt(ratio) * sample + torch.sqrt(1.0 - ratio) * torch.randn_like(
            sample
        )


def make_ddim_timesteps(num_train_timesteps: int, ddim_steps: int) -> list[int]:
    """Return descending timestep subsequence for DDIM."""

    steps = max(1, min(num_train_timesteps, int(ddim_steps)))
    values = torch.linspace(0, num_train_timesteps - 1, steps).round().long().flip(0)
    return [int(value) for value in values.unique_consecutive()]


def make_repaint_schedule(
    num_train_timesteps: int,
    num_inference_steps: int,
    jump_length: int,
    jump_n_sample: int,
) -> list[int]:
    """Create the RePaint jump-back schedule scaled to training timesteps."""

    num_inference_steps = max(1, min(num_train_timesteps, num_inference_steps))
    jump_length = max(1, int(jump_length))
    jump_n_sample = max(1, int(jump_n_sample))
    jumps = {
        j: jump_n_sample - 1
        for j in range(0, max(num_inference_steps - jump_length, 0), jump_length)
    }
    sequence: list[int] = []
    t = num_inference_steps
    while t >= 1:
        t -= 1
        sequence.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                sequence.append(t)
    scale = max(1, num_train_timesteps // num_inference_steps)
    scaled = [min(num_train_timesteps - 1, value * scale) for value in sequence]
    if scaled[-1] != -1:
        scaled.append(-1)
    return scaled
