"""Hand-worked checks of the asset-weighted composite aggregate return."""
import pytest

from cra import registry


def test_composite_return_is_beginning_value_weighted_average():
    composite_return = registry.get_metric("composite_return")
    # Account A: 600,000 beginning value, +1% return.
    # Account B: 400,000 beginning value, -2% return.
    # Weighted: (600,000*0.01 + 400,000*-0.02) / 1,000,000 = -0.002.
    value = composite_return([600_000, 400_000], [0.01, -0.02])
    assert value == pytest.approx(-0.002)


def test_composite_return_equal_weights_is_plain_average():
    composite_return = registry.get_metric("composite_return")
    value = composite_return([500_000, 500_000], [0.05, -0.01])
    assert value == pytest.approx((0.05 + -0.01) / 2)


def test_composite_return_rejects_nonpositive_total():
    composite_return = registry.get_metric("composite_return")
    with pytest.raises(ValueError):
        composite_return([0.0, 0.0], [0.01, 0.02])


def test_excess_return_is_plain_subtraction():
    excess_return = registry.get_metric("excess_return")
    assert excess_return(0.08, 0.05) == pytest.approx(0.03)
    assert excess_return(0.02, 0.05) == pytest.approx(-0.03)
