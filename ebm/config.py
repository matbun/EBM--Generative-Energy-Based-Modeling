import os
import numpy as np
import torch

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import shutil
except:
    install("pytest-shutil")
    import shutil


HOME_DIR = os.path.expanduser("~")

PROJECT_ROOT = os.path.join(HOME_DIR, "Projects/EBM_proj")
os.chdir(PROJECT_ROOT)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = PROJECT_ROOT + "/data"

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = PROJECT_ROOT + "/saved_models/MNIST/"


# Set rdn seed
def set_seed(seed: int=0, deterministic: bool=True, benchmark: bool=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = deterministic
    torch.backends.cudnn.benchmark = benchmark