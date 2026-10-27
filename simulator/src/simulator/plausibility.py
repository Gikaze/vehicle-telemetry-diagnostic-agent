"""Pure plausibility functions over disc temperature series, shared by the invariant tests and the
stream checks (definitions and bounds in config/plausibility.yaml)."""

import numpy as np


def disc_rise_ratio(
    temps: np.ndarray, ambient: float, window: int, min_rear_rise: float
) -> np.ndarray:
    """Front/rear ratio of the axle mean disc temperature rises, one value per window.

    `temps` holds one row per sample, four columns in WHEELS order (fl, fr, rl, rr), degC. The
    samples are cut into consecutive windows of `window` samples (a trailing partial window is
    dropped). Per window: the mean over the window of the front axle mean rise above `ambient`,
    over the same for the rear axle; NaN where the rear value is not above `min_rear_rise` (K).
    """
    if window < 1:
        raise ValueError("the window must hold at least one sample")
    count = len(temps) // window
    rise = np.asarray(temps)[: count * window] - ambient
    front = rise[:, :2].mean(axis=1).reshape(count, window).mean(axis=1)
    rear = rise[:, 2:].mean(axis=1).reshape(count, window).mean(axis=1)
    evaluated = rear > min_rear_rise
    return np.where(evaluated, front / np.where(evaluated, rear, 1.0), np.nan)
