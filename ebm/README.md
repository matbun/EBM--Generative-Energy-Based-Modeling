This folder is a package for EBM training.

# train.py
Trainer classes for different approaches in EBM training (MCMC sampling from model):
- `EBMLangVanilla` to use first order Langevin dynamics.  
- `EBMLang2Ord` to use second order Langevin dynamics.
- ...

# models.py
Two possible CNN models for MNIST dataset.

# config.py
Is a configuration module that contains variables that useful in all modules and in the training notebook.
In the training notebook import it as `from ebm.config import *`, before everything else concerning EBM.
