# Activation Sparsity & Regularization in Convolutional Networks
**CSCI 567 — University of Southern California**

## Authors
Dominic Woetzel, Kaushika Degekar Gulbarga, Sankara Bharadwaj Rangavajjala, Sarghi Kaur Gahra

## Overview
This project systematically compares four activation functions — ReLU, GELU, SiLU, and SwiGLU — combined with three regularization strategies — dropout, weight decay, and stochastic depth — on a ResNet-20 architecture trained on CIFAR-100. We analyze how activation sparsity interacts with explicit regularization and evaluate models on both clean accuracy and corruption robustness (CIFAR-100-C).

## Repository Structure
```
.
├── template_RELU_run.ipynb   # ReLU experiments (4 regularizers × 2 seeds, 200 epochs)
├── template_GELU_run.ipynb   # GELU experiments (4 regularizers × 2 seeds, 200 epochs)
├── template_Swiglu_run.ipynb # SwiGLU experiments (4 regularizers × 2 seeds, 200 epochs)
└── template_SILU_run.ipynb   # SiLU experiments (4 regularizers, 100 epochs)
```

## Dataset
- **CIFAR-100**: downloaded automatically via `torchvision.datasets.CIFAR100`
- **CIFAR-100-C**: downloaded automatically from [Zenodo](https://zenodo.org/record/3555552) during evaluation cells

## How to Reproduce Results
1. Open the desired notebook in Google Colab
2. Run all cells sequentially — training, evaluation, and CIFAR-100-C robustness are all included
3. Per-run results are printed after each training run; a summary table and plots are generated in the final cells

Each notebook covers the following configurations:

| Regularizer | Dropout | Weight Decay | Stochastic Depth | Baseline |
|---|---|---|---|---|
| Rate / Coefficient | 0.1 (0.3 for SiLU) | 5e-4 | 0.1 | — |

## Key Results

| Activation | Best Top-1 | Best Config | Baseline Top-1 |
|---|---|---|---|
| SiLU | **68.54%** | Weight Decay | 63.82% |
| ReLU | 67.94% | Weight Decay | 62.25% |
| GELU | 67.73% | Weight Decay | 62.64% |
| SwiGLU | 65.78% | Weight Decay | 59.45% |

## Dependencies
All notebooks run on Google Colab with the following libraries:
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

No additional installation is required when running on Colab.

## Notes
- ReLU, GELU, and SwiGLU models are trained for 200 epochs; SiLU for 100 epochs
- All 200-epoch results are averaged over 2 random seeds
- ReLU² was attempted but training diverged due to gradient explosion in all configurations
