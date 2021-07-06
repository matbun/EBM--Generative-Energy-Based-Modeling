This folder is a package for EBM training on toy datasets taken from https://github.com/point0bar1/ebm-anatomy.

# train.py
Trainer classes for different approaches in EBM training (MCMC sampling from model):
- `EBMLangVanilla` to use first order Langevin dynamics (SGLD).  
- `EBMLang2Ord` to use second order Langevin dynamics (SGHMC).
- 
# models.py
CNN model for 2D examples of shape: `(C x H X W) = (2 x 1 x 1)`.

# utils.py
- ToyDataset class
- `ksDist` and `ks2d2s`: 2D Kolmogorov-Sminorv test to compute the distance among two samples' distributions.
- Discrete KL divergence on a 2D grid

# config.py
Is a configuration module that contains global variables that are useful in all modules and in the training notebook.
In the training notebook import it as `from ebm_toy.config import *`, before everything else concerning EBM.
