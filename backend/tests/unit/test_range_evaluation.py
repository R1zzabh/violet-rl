"""Statistical intervals reported by real-range evaluation."""

from __future__ import annotations

import pytest

from violet.range.evaluate_real import wilson_interval


def test_wilson_interval_has_nonzero_uncertainty_at_binary_extremes() -> None:
    interval = wilson_interval(0, 20)
    assert interval is not None
    lower, upper = interval
    assert lower == pytest.approx(0.0)
    assert 0.15 < upper < 0.17
    interval = wilson_interval(20, 20)
    assert interval is not None
    lower, upper = interval
    assert 0.83 < lower < 0.85
    assert upper == pytest.approx(1.0)


def test_wilson_interval_is_undefined_for_no_eligible_episodes() -> None:
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="between"):
        wilson_interval(1, 0)
