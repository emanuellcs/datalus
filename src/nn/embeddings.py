"""
DATALUS Neural Network Module
Layer 2: Heterogeneous Embeddings and Latent Space Projections
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor


class FeatureProjector(nn.Module):
    """
    Dynamically maps mixed tabular data (continuous and categorical) into a unified, 
    continuous latent space suitable for Gaussian Diffusion.
    """
    def __init__(
        self,
        schema_metadata: Dict[str, dict],
        embedding_rule: str = "fastai"
    ):
        super().__init__()
        self.schema_metadata = schema_metadata
        
        # Isolate numerical and categorical features based on the Auto-Prep schema
        self.cat_features = [
            col for col, meta in schema_metadata.items() 
            if "CATEGORICAL" in meta["inferred_topology"] or meta["inferred_topology"] == "BOOLEAN"
        ]
        self.num_features = [
            col for col, meta in schema_metadata.items() 
            if "NUMERICAL" in meta["inferred_topology"]
        ]
        
        self.num_dim = len(self.num_features)
        self.cat_dims: List[Tuple[int, int]] = []
        
        # Register PyTorch Embeddings dynamically using an elite ModuleList approach
        self.embeddings = nn.ModuleList()
        
        for col in self.cat_features:
            cardinality = schema_metadata[col]["cardinality"]
            # To handle unobserved categories during inference (e.g., NaNs or new tokens), we add +1
            vocab_size = cardinality + 1 
            
            # Determine embedding dimension
            if embedding_rule == "fastai":
                # Fast.ai heuristic: min(50, 1.6 * cardinality^0.56)
                emb_dim = min(50, int(1.6 * (vocab_size ** 0.56)))
            elif embedding_rule == "log2":
                emb_dim = max(2, math.ceil(math.log2(vocab_size)))
            else:
                emb_dim = 16 # Default fixed fallback
                
            self.cat_dims.append((vocab_size, emb_dim))
            
            emb_layer = nn.Embedding(vocab_size, emb_dim)
            # Xavier/Glorot initialization for smooth latent manifold start
            nn.init.xavier_uniform_(emb_layer.weight)
            self.embeddings.append(emb_layer)
            
        # Total continuous dimension that will be passed to the MLP Denoiser
        self.total_latent_dim = self.num_dim + sum(emb_dim for _, emb_dim in self.cat_dims)
        
    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        """
        Args:
            x_num: Tensor of shape (batch_size, num_numerical_features)
            x_cat: Tensor of shape (batch_size, num_categorical_features) containing integer indices
        Returns:
            x_continuous: Tensor of shape (batch_size, total_latent_dim)
        """
        tensors_to_concat = []
        
        if x_num is not None and self.num_dim > 0:
            tensors_to_concat.append(x_num)
            
        if x_cat is not None and len(self.cat_features) > 0:
            # Iterate through categorical columns and apply their respective embedding layers
            for i, emb_layer in enumerate(self.embeddings):
                # x_cat[:, i] extracts the discrete indices for the i-th categorical feature
                emb_out = emb_layer(x_cat[:, i].long())
                tensors_to_concat.append(emb_out)
                
        # Concatenate numerical and embedded categorical features along the feature dimension
        x_continuous = torch.cat(tensors_to_concat, dim=-1)
        return x_continuous


class LatentDecoder(nn.Module):
    """
    Reverses the projection process. Takes the denoised continuous latent tensor 
    and maps it back to numerical values and categorical logits for evaluation/sampling.
    """
    def __init__(self, projector: FeatureProjector):
        super().__init__()
        self.num_dim = projector.num_dim
        self.cat_dims = projector.cat_dims
        
        # For categorical features, we need linear layers to project the embedding 
        # dimension back to the vocabulary size (to compute softmax/logits).
        self.decoders = nn.ModuleList()
        for vocab_size, emb_dim in self.cat_dims:
            self.decoders.append(nn.Linear(emb_dim, vocab_size))

    def forward(self, x_continuous: Tensor) -> Tuple[Optional[Tensor], List[Tensor]]:
        """
        Args:
            x_continuous: The generated synthetic tensor of shape (batch_size, total_latent_dim)
        Returns:
            Tuple containing:
            - x_num_reconstructed: Tensor of numerical features
            - cat_logits: List of Tensors, each containing the logits for a categorical feature
        """
        batch_size = x_continuous.shape[0]
        
        # 1. Extract numerical features (they are at the beginning of the concatenated tensor)
        x_num_reconstructed = None
        current_idx = 0
        
        if self.num_dim > 0:
            x_num_reconstructed = x_continuous[:, :self.num_dim]
            current_idx += self.num_dim
            
        # 2. Extract and decode categorical features
        cat_logits = []
        for i, (vocab_size, emb_dim) in enumerate(self.cat_dims):
            # Slice the embedding corresponding to the i-th categorical feature
            emb_slice = x_continuous[:, current_idx : current_idx + emb_dim]
            current_idx += emb_dim
            
            # Project back to logits (for Multinomial loss or Argmax sampling)
            logits = self.decoders[i](emb_slice)
            cat_logits.append(logits)
            
        return x_num_reconstructed, cat_logits


if __name__ == "__main__":
    # Smoke Test: Mocking Auto-Prep Schema
    mock_schema = {
        "IDADE": {"inferred_topology": "NUMERICAL_CONTINUOUS", "cardinality": None},
        "TEMPO_UTI": {"inferred_topology": "NUMERICAL_DISCRETE", "cardinality": None},
        "SEXO": {"inferred_topology": "CATEGORICAL_LOW_CARDINALITY", "cardinality": 2},
        "CID_10": {"inferred_topology": "CATEGORICAL_HIGH_CARDINALITY", "cardinality": 150}
    }
    
    batch_size = 64
    
    # Instantiate Projector and Decoder
    projector = FeatureProjector(schema_metadata=mock_schema, embedding_rule="fastai")
    decoder = LatentDecoder(projector)
    
    # Mock Data
    dummy_x_num = torch.randn(batch_size, 2) # IDADE, TEMPO_UTI
    dummy_x_cat = torch.randint(0, 2, (batch_size, 2)) # Dummy indices for SEXO and CID_10
    
    print(f"Projector expecting Latent Dim: {projector.total_latent_dim}")
    
    # Forward Pass: Tabular -> Latent Space
    latent_tensor = projector(dummy_x_num, dummy_x_cat)
    print(f"Latent Tensor Shape: {latent_tensor.shape}")
    
    # Reverse Pass: Latent Space -> Tabular
    rec_num, rec_cat_logits = decoder(latent_tensor)
    print(f"Reconstructed Numerical Shape: {rec_num.shape}")
    print(f"Reconstructed SEXO Logits Shape: {rec_cat_logits[0].shape}")
    print(f"Reconstructed CID_10 Logits Shape: {rec_cat_logits[1].shape}")