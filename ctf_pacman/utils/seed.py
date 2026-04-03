"""Global random seed utility for full reproducibility."""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set the random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value. Use the same seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
