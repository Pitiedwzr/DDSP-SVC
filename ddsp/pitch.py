import numpy as np


def interpolate_unvoiced_f0(f0, f0_min):
    """Interpolate an F0 contour without discarding its voicing decisions."""
    f0 = np.asarray(f0).copy()
    voiced = f0 > 0
    if voiced.any():
        uv = ~voiced
        f0[uv] = np.interp(np.where(uv)[0], np.where(voiced)[0], f0[voiced])
        f0[f0 < f0_min] = f0_min
    return f0, voiced.astype(np.float32)
