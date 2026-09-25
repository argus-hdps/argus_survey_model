"""Property tests for SystemThroughput component bounds."""

import numpy as np
import pytest

from argus_sim import SystemThroughput


@pytest.fixture(scope="module")
def tp():
    return SystemThroughput(throughput_loss=1.0)


class TestComponentBounds:
    """Every throughput component must be in [0, 1] across the full wavelength grid."""

    def test_qe_bounded(self, tp):
        qe = np.asarray(tp.qe_wav)
        assert np.all(qe >= 0), f"QE min = {qe.min()}"
        assert np.all(qe <= 1), f"QE max = {qe.max()}"

    def test_atmosphere_bounded(self, tp):
        atm = np.asarray(tp.filt_wav.atmosphere)
        assert np.all(atm >= 0), f"atmosphere min = {atm.min()}"
        assert np.all(atm <= 1), f"atmosphere max = {atm.max()}"

    def test_optics_efficiency_bounded(self, tp):
        opt = np.array(list(tp.optics.values()))
        assert np.all(opt > 0), f"optics min = {opt.min()}"
        assert np.all(opt <= 1), f"optics max = {opt.max()}"

    @pytest.mark.parametrize("band", ["V", "g", "r", "i", "A", "sqm"])
    def test_filter_response_bounded(self, tp, band):
        filt = np.asarray(tp.filt_wav[band])
        assert np.all(filt >= 0), f"{band} filter min = {filt.min()}"
        assert np.all(filt <= 1), f"{band} filter max = {filt.max()}"
