# Energy Based Models
EBMs are a family of models currently under research. Their remarkable advantage with respect to VAEs is that they do not make any assumption on the form of the probability density they fit. These models are also a potential competitor of GANs. In this work (project at EURECOM University) I implement them as generative models with *Maximum Likelihood* estimation, aimed at generating MNIST images.  
Here you can also find the final report of my project: `Final_Report__EBM___ML.pdf`  
For more info: https://arxiv.org/abs/2101.03288

# EBM PyTorch training packages
These packages offer key utilities to train an Energy Based Model with ML estimation. MCMC sampling from the model can be carried out with Langevin dynamics (SGLD) or Stochastic Gradient Hamiltonian Monte Carlo (SGHMC).   

Two python packages:
- `ebm`: train on MNIST dataset
- `ebm_toy` in `toy_examples` folder: train on **gmm** (gaussian mixture model) or **circles** 2D datasets, where the ground truth distribution is known and supervised metrics (e.g. Kolmogorov-Smirnov distance) can be computed.

