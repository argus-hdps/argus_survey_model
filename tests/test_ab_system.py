"""Round-trip property tests for ABPhot magnitude ↔ photon conversions."""

import numpy as np
import pytest


from argus_sim import ABPhot, SystemThroughput, c


@pytest.fixture(scope="module")
def ab():
    tp = SystemThroughput(throughput_loss=1.0)
    collecting_area = c.telescope.collecting_area
    return ABPhot(collecting_area=collecting_area, throughput_loss=1.0, throughput=tp)


BANDS = ["V", "g", "r", "i", "A"]
MAGNITUDES = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]


@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("mag", MAGNITUDES)
def test_mag_photons_roundtrip(ab, band, mag):
    """mag_from_photons(photons_from_mag(m)) must recover m within 1e-6."""
    photons = ab.photons_from_mag(band, mag)
    recovered = ab.mag_from_photons(band, photons)
    np.testing.assert_allclose(float(recovered), mag, atol=1e-6)
