"""Plausibility functions over disc temperature series (plausibility.yaml definitions)."""

import numpy as np
import pytest

from simulator.plausibility import disc_rise_ratio


def test_disc_rise_ratio_by_hand() -> None:
    """Ambient 10 degC, windows of 2 samples, ratio evaluated above 20 K of rear rise.
    Window 1: front axle rises 30 and 40 K (mean 35), rear 20 and 22 K (mean 21): 35 / 21.
    Window 2: rear axle mean rise exactly 20 K: not evaluated (the threshold is strict).
    The fifth sample is a partial window and is dropped."""
    temps = np.array(
        [
            [38.0, 42.0, 29.0, 31.0],  # front 30, rear 20
            [48.0, 52.0, 31.0, 33.0],  # front 40, rear 22
            [60.0, 60.0, 30.0, 30.0],  # front 50, rear 20
            [60.0, 60.0, 30.0, 30.0],
            [99.0, 99.0, 99.0, 99.0],
        ]
    )
    ratio = disc_rise_ratio(temps, 10.0, 2, 20.0)
    assert len(ratio) == 2
    assert ratio[0] == pytest.approx(35 / 21, rel=1e-12)
    assert np.isnan(ratio[1])


def test_disc_rise_ratio_uses_the_axle_means() -> None:
    """A left/right difference on one axle moves the ratio only through the axle mean."""
    temps = np.array([[50.0, 70.0, 40.0, 40.0]])  # front rises 40 and 60, rear 30 and 30
    assert disc_rise_ratio(temps, 10.0, 1, 20.0)[0] == pytest.approx(50 / 30, rel=1e-12)


def test_disc_rise_ratio_needs_a_window() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        disc_rise_ratio(np.zeros((4, 4)), 0.0, 0, 20.0)
