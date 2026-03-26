# DATALUS: Diffusion Augmented Tabular Architecture for Local Utility and Security

> **English** | [Avaliadores do Prêmio Jovem Cientista: Leia a documentação em Português aqui](./README_pt-BR.md)

DATALUS is a general-purpose, open-source Generative AI framework designed to synthesize high-dimensional, heterogeneous tabular data. By leveraging Denoising Diffusion Probabilistic Models (TabDDPM), DATALUS learns the joint probability distribution of sensitive datasets and samples entirely new, statistically identical records from pure noise.

Built with a focus on MLOps and Edge Computing, this framework enables organizations to democratize access to their data for machine learning research without compromising cryptographic privacy.

## Core Architecture

DATALUS is built upon four technical pillars:

* **Zero-Shot Preprocessing:** Utilizing `Polars` for lazy evaluation, the engine automatically infers column topology (continuous, discrete, high-cardinality categorical) and applies latent embedding mappings without loading massive datasets into RAM.
* **Heterogeneous Diffusion Engine:** Applies continuous Gaussian diffusion for numerical features and Multinomial diffusion for categorical embeddings, parameterized by a robust Multi-Layer Perceptron (MLP) with time-step embeddings.
* **Autonomous Audit Orchestrator (OAA):** Automatically evaluates every generated artifact across two dimensions:
  * *Privacy:* Distance to Closest Record (DCR) and Membership Inference Attack (MIA) resistance.
  * *Utility:* Machine Learning Efficacy via Train-on-Synthetic, Test-on-Real (TSTR) using gradient boosting models (CatBoost/LightGBM).
* **Edge-Optimized Deploy:** The master weights (FP32) are exported to the ONNX Runtime and subjected to Post-Training Quantization (INT8). This allows lightning-fast, offline inference on commodity CPUs (via AVX-2 instructions).

## Advanced Capabilities

* **Stochastic Data Augmentation:** Expand imbalanced datasets with high-fidelity synthetic samples.
* **Classifier-Free Guidance (CFG):** Condition the generation process on specific latent attributes.
* **Tabular Inpainting:** Probabilistic imputation for missing data (NaNs).
* **Counterfactual Simulation:** Intervene in latent variables for policy impact modeling.

## Ecosystem & Reproducibility

This GitHub monorepo contains the agnostic engine. Pre-trained weights and synthetic datasets are hosted on our decentralized infrastructure:
* **Hugging Face:** Serves as our Model Registry, hosting the `.onnx` artifacts and metadata schemas.
* **Kaggle:** Hosts our Proof-of-Concept datasets, Model Cards, and didactic notebooks demonstrating the mathematical efficacy of the framework.

## License
This project is licensed under the Apache License 2.0.