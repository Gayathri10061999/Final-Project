"""
Seed Utility

Ensures reproducible training.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed=42):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    print("=" * 50)
    print(f"Seed Set Successfully : {seed}")
    print("=" * 50)
