"""Image-quality metrics (numpy)."""

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-10))
