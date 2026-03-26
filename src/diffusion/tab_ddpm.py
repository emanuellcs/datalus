"""
DATALUS Diffusion Module
Layer 3: Markov Chain Orchestrator and DDIM Sampler for Tabular Data
"""

import math
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class VarianceSchedule(nn.Module):
    """
    Computes and stores the noise scheduling parameters (beta, alpha, alpha_bar)
    for the forward and reverse Markov processes.
    """
    def __init__(self, num_timesteps: int = 1000, schedule_type: str = "cosine"):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers so they are moved to the correct device automatically
        # but are not considered model parameters to be updated by the optimizer.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> Tensor:
        """
        Cosine schedule proposed by Nichol and Dhariwal (2021).
        Prevents the noise level from destroying information too quickly in early steps.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def extract(self, a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
        """
        Extracts the appropriate scaling factor for a batch of timesteps 't'.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class TabularDiffusion(nn.Module):
    """
    The orchestrator class that manages the diffusion process for continuous latent tensors.
    Integrates the Denoising MLP and the Variance Schedule.
    """
    def __init__(
        self,
        denoiser: nn.Module,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine"
    ):
        super().__init__()
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.schedule = VarianceSchedule(num_timesteps, schedule_type)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        Forward Process: Diffuses data by injecting Gaussian noise for a specific timestep t.
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.schedule.extract(self.schedule.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.schedule.extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start: Tensor, context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Computes the Mean Squared Error (MSE) loss for the Simple Objective.
        Handles dynamic Classifier-Free Guidance dropping if context is provided.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample uniform timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # Classifier-Free Guidance: Randomly drop context during training (10% of the time)
        if context is not None:
            drop_mask = torch.rand((batch_size, 1), device=device) < 0.1
            context = torch.where(drop_mask, torch.zeros_like(context), context)

        predicted_noise = self.denoiser(x_noisy, t, context)
        
        # Simple MSE Objective
        loss = F.mse_loss(predicted_noise, noise)
        
        return {"loss": loss}

    @torch.no_grad()
    def p_sample_ddim(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        context: Optional[Tensor] = None,
        cfg_scale: float = 3.0
    ) -> Tensor:
        """
        Reverse Process: Samples synthetic data using the accelerated DDIM algorithm.
        Allows for conditional generation using Classifier-Free Guidance (CFG).
        """
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        
        # Create subsequences for accelerated sampling
        step_ratio = self.num_timesteps // ddim_steps
        timesteps = (torch.arange(0, ddim_steps) * step_ratio).round()[::-1].long().to(device)

        for i, step in enumerate(timesteps):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # Predict noise (with or without CFG)
            if context is not None and cfg_scale > 1.0:
                # Double batch for CFG: [conditional, unconditional]
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t, t], dim=0)
                c_double = torch.cat([context, torch.zeros_like(context)], dim=0)
                
                pred_noise_double = self.denoiser(x_double, t_double, c_double)
                pred_noise_cond, pred_noise_uncond = pred_noise_double.chunk(2)
                
                # Extrapolate towards the condition
                pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = self.denoiser(x, t, context)

            # DDIM update step
            alpha_bar = self.schedule.alphas_cumprod[step]
            alpha_bar_prev = self.schedule.alphas_cumprod[timesteps[i + 1]] if i < ddim_steps - 1 else torch.tensor(1.0)
            
            # Predict original data x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            
            # Deterministic progression to the next step
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return x

    def forward(self, x_start: Tensor, context: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Standard PyTorch forward interface mapped to loss computation for elegant training loops.
        """
        return self.compute_loss(x_start, context)


if __name__ == "__main__":
    # Smoke test for the Diffusion Engine
    from mlp import TabularDenoiserMLP
    
    batch_dim = 16
    feat_dim = 100
    
    dummy_model = TabularDenoiserMLP(d_in=feat_dim)
    diffusion = TabularDiffusion(denoiser=dummy_model, num_timesteps=1000)
    
    dummy_data = torch.randn(batch_dim, feat_dim)
    
    # Test loss computation
    loss_dict = diffusion(dummy_data)
    print(f"Training Loss: {loss_dict['loss'].item():.4f}")
    
    # Test DDIM Sampling (Fast generation in 50 steps)
    synthetic_latent = diffusion.p_sample_ddim(
        shape=(batch_dim, feat_dim),
        device=torch.device("cpu"),
        ddim_steps=50
    )
    print(f"Synthetic Data Shape: {synthetic_latent.shape} | Expected: [{batch_dim}, {feat_dim}]")