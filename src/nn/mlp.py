"""
DATALUS Neural Network Module
Layer 3: Robust Multi-Layer Perceptron (MLP) for Tabular Denoising
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes the diffusion timestep 't' into a dense continuous vector 
    using sinusoidal positional embeddings, similar to Transformer architectures.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Tensor of shape (batch_size,) containing timesteps.
        Returns:
            Tensor of shape (batch_size, dim) containing time embeddings.
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Compute the exponential scaling factor
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Calculate sinusoidal embeddings
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 != 0:
            embeddings = torch.cat((embeddings, torch.zeros_like(embeddings[:, :1])), dim=-1)
            
        return embeddings


class ResidualMLPBlock(nn.Module):
    """
    A robust residual block integrating time embeddings and optional context
    conditioning for Classifier-Free Guidance (CFG).
    """
    def __init__(self, d_in: int, d_out: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_in, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_out, d_out)
        self.norm2 = nn.LayerNorm(d_out)
        
        # Projection for time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, d_out)
        )
        
        # Residual connection projection (if dimensions change)
        self.residual_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass with time embedding injection.
        """
        h = self.linear1(x)
        h = self.norm1(h)
        
        # Inject time embedding
        time_context = self.time_mlp(t_emb)
        h = h + time_context
        
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.norm2(h)
        
        return h + self.residual_proj(x)


class TabularDenoiserMLP(nn.Module):
    """
    The core eps-theta model. Learns to predict and reverse the added 
    Gaussian and Multinomial noise from the continuous latent space.
    """
    def __init__(
        self,
        d_in: int,
        dim_t: int = 128,
        hidden_dims: tuple = (256, 512, 512, 256),
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.d_in = d_in
        
        # Time embedding pipeline
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim_t),
            nn.Linear(dim_t, dim_t * 4),
            nn.SiLU(),
            nn.Linear(dim_t * 4, dim_t * 4)
        )
        time_emb_dim = dim_t * 4
        
        # Context embedding pipeline for CFG (optional)
        self.has_context = context_dim is not None
        if self.has_context:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
            
        # Network layers
        self.input_proj = nn.Linear(d_in, hidden_dims[0])
        
        self.blocks = nn.ModuleList()
        in_dims = hidden_dims[:-1]
        out_dims = hidden_dims[1:]
        
        for d_in_block, d_out_block in zip(in_dims, out_dims):
            self.blocks.append(
                ResidualMLPBlock(d_in_block, d_out_block, time_emb_dim, dropout)
            )
            
        # Final projection back to data dimension
        self.final_norm = nn.LayerNorm(hidden_dims[-1])
        self.act = nn.SiLU()
        self.final_proj = nn.Linear(hidden_dims[-1], d_in)
        
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the final layer to zero to ensure the model outputs 
        an identity mapping at the very beginning of training.
        """
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Noisy latent representations of shape (batch_size, d_in)
            t: Diffusion timesteps of shape (batch_size,)
            c: Optional conditioning vectors of shape (batch_size, context_dim)
        Returns:
            Predicted noise epsilon of shape (batch_size, d_in)
        """
        # Embed time
        t_emb = self.time_embed(t)
        
        # Add conditioning context if provided (for Classifier-Free Guidance)
        if self.has_context and c is not None:
            c_emb = self.context_proj(c)
            t_emb = t_emb + c_emb
            
        # Pass through the network
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = block(h, t_emb)
            
        h = self.final_norm(h)
        h = self.act(h)
        
        return self.final_proj(h)


if __name__ == "__main__":
    # Smoke test for the architecture
    batch_size = 32
    feature_dim = 150
    
    model = TabularDenoiserMLP(d_in=feature_dim)
    
    dummy_x = torch.randn(batch_size, feature_dim)
    dummy_t = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    output = model(dummy_x, dummy_t)
    print(f"Output shape: {output.shape} | Expected: [{batch_size}, {feature_dim}]")