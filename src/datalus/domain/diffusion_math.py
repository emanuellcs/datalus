"""Framework-independent diffusion schedule mathematics.

This module avoids NumPy, PyTorch, and ONNX dependencies by returning plain
Python floats and integers. Infrastructure adapters convert these values into
framework tensors for training, export, or browser inference.
"""

from __future__ import annotations

import math


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> list[float]:
    """Return Nichol-Dhariwal cosine betas with numerical clipping."""

    if timesteps < 1:
        raise ValueError("timesteps must be positive.")
    alpha_bar: list[float] = []
    for step in range(timesteps + 1):
        x = step / timesteps
        value = math.cos(((x + s) / (1.0 + s)) * math.pi / 2.0) ** 2
        alpha_bar.append(value)
    first = alpha_bar[0]
    alpha_bar = [value / first for value in alpha_bar]
    betas = [1.0 - alpha_bar[idx + 1] / alpha_bar[idx] for idx in range(timesteps)]
    return [min(0.999, max(1e-5, beta)) for beta in betas]


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> list[float]:
    """Return a simple linear beta schedule for ablation and tests."""

    if timesteps < 1:
        raise ValueError("timesteps must be positive.")
    if timesteps == 1:
        return [beta_start]
    delta = (beta_end - beta_start) / (timesteps - 1)
    return [beta_start + idx * delta for idx in range(timesteps)]


def make_ddim_timesteps(num_train_timesteps: int, ddim_steps: int) -> list[int]:
    """Return a descending deterministic DDIM subsequence."""

    steps = max(1, min(num_train_timesteps, int(ddim_steps)))
    if steps == 1:
        return [num_train_timesteps - 1]
    values = [
        round(idx * (num_train_timesteps - 1) / (steps - 1)) for idx in range(steps)
    ]
    descending = list(reversed(values))
    deduped: list[int] = []
    for value in descending:
        if not deduped or deduped[-1] != value:
            deduped.append(int(value))
    return deduped


def make_repaint_schedule(
    num_train_timesteps: int,
    num_inference_steps: int,
    jump_length: int,
    jump_n_sample: int,
) -> list[int]:
    """Create the RePaint jump-back schedule scaled to training timesteps."""

    num_inference_steps = max(1, min(num_train_timesteps, int(num_inference_steps)))
    jump_length = max(1, int(jump_length))
    jump_n_sample = max(1, int(jump_n_sample))
    jumps = {
        step: jump_n_sample - 1
        for step in range(0, max(num_inference_steps - jump_length, 0), jump_length)
    }
    sequence: list[int] = []
    step = num_inference_steps
    while step >= 1:
        step -= 1
        sequence.append(step)
        if jumps.get(step, 0) > 0:
            jumps[step] -= 1
            for _ in range(jump_length):
                step += 1
                sequence.append(step)
    scale = max(1, num_train_timesteps // num_inference_steps)
    scaled = [min(num_train_timesteps - 1, value * scale) for value in sequence]
    if scaled[-1] != -1:
        scaled.append(-1)
    return scaled
