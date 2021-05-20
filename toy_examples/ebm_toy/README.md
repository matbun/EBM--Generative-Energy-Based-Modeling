This folder is a package for EBM training on toy datasets taken from https://github.com/point0bar1/ebm-anatomy.

# train.py
Trainer classes for different approaches in EBM training (MCMC sampling from model):
- `EBMLangVanilla` to use first order Langevin dynamics.  
- `EBMLang2Ord` to use second order Langevin dynamics.
- ...

# models.py
CNN model for 2D examples of shape: `(C x W X H) = (2 x 1 x 1)`.

# utils.py
- ToyDataset class
- `ksDist` and `ks2d2s`: 2D Kolmogorov-Sminorv test to compute the distance among two samples' distributions.
- ...

# config.py
Is a configuration module that contains variables that useful in all modules and in the training notebook.
In the training notebook import it as `from ebm.config import *`, before everything else concerning EBM.
