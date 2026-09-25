"""Properties of the Leinert (1998) Table 17 zodiacal light map.

Zodiacal light at fixed helioecliptic longitude must fall, or stay flat, toward
the ecliptic pole.  Cells the published table leaves blank must be NaN in the
tabulated array and filled by a rule that preserves that property, so the
interpolator never sees a brightness that rises with |beta|.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.interpolate import RegularGridInterpolator

from argus_sim.data import zodiacal_leinert1998 as zl
from argus_sim.data.zodiacal_leinert1998 import ABS_BETA_DEG, ELONGATION_DEG, ZODIACAL_S10


@pytest.fixture
def filled():
    if not hasattr(zl, "ZODIACAL_S10_FILLED"):
        pytest.fail("zodiacal_leinert1998 exposes no ZODIACAL_S10_FILLED array")
    return zl.ZODIACAL_S10_FILLED


def _assert_non_increasing(row, label):
    finite = row[np.isfinite(row)]
    assert np.all(np.diff(finite) <= 0), f"{label}: brightness rises toward the pole: {finite}"


@pytest.mark.parametrize("i", range(len(ELONGATION_DEG)))
def test_tabulated_row_non_increasing_in_abs_beta(i):
    _assert_non_increasing(ZODIACAL_S10[i], f"elong={ELONGATION_DEG[i]}")


def test_table_shape_matches_axes(filled):
    assert ZODIACAL_S10.shape == (len(ELONGATION_DEG), len(ABS_BETA_DEG))
    assert filled.shape == ZODIACAL_S10.shape


def test_axes_strictly_increasing():
    assert np.all(np.diff(ELONGATION_DEG) > 0)
    assert np.all(np.diff(ABS_BETA_DEG) > 0)


def test_tabulated_values_positive():
    assert np.all(ZODIACAL_S10[np.isfinite(ZODIACAL_S10)] > 0)


def test_filled_table_is_finite(filled):
    assert np.all(np.isfinite(filled))


def test_filled_table_preserves_tabulated_cells(filled):
    tabulated = np.isfinite(ZODIACAL_S10)
    np.testing.assert_array_equal(filled[tabulated], ZODIACAL_S10[tabulated])


@pytest.mark.parametrize("i", range(len(ELONGATION_DEG)))
def test_filled_row_non_increasing_in_abs_beta(i, filled):
    _assert_non_increasing(filled[i], f"filled elong={ELONGATION_DEG[i]}")


def test_every_row_has_an_in_plane_anchor():
    assert np.all(np.isfinite(ZODIACAL_S10[:, 0]))


@settings(max_examples=200, deadline=None)
@given(
    lon=st.floats(min_value=30.0, max_value=180.0),
    b1=st.floats(min_value=0.0, max_value=90.0),
    b2=st.floats(min_value=0.0, max_value=90.0),
)
def test_interpolated_brightness_non_increasing_in_abs_beta(lon, b1, b2):
    interp = RegularGridInterpolator(
        (ELONGATION_DEG, ABS_BETA_DEG),
        getattr(zl, "ZODIACAL_S10_FILLED", ZODIACAL_S10),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    lo, hi = sorted((b1, b2))
    s_lo, s_hi = interp([[lon, lo], [lon, hi]])
    assert np.isfinite(s_lo) and np.isfinite(s_hi)
    assert s_hi <= s_lo * (1 + 1e-12)
