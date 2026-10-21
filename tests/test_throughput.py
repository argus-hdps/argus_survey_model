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

    @pytest.mark.parametrize("band", ["V", "beta", "rho", "sqm"])
    def test_filter_response_bounded(self, tp, band):
        filt = np.asarray(tp.filt_wav[band])
        assert np.all(filt >= 0), f"{band} filter min = {filt.min()}"
        assert np.all(filt <= 1), f"{band} filter max = {filt.max()}"


def test_survey_bands_only():
    """Only beta and rho are survey bands; V and SQM are references for the sky model."""
    from argus_sim.throughput import OPTICS_EFFICIENCY, SURVEY_BANDS

    assert SURVEY_BANDS == ("beta", "rho")
    assert set(OPTICS_EFFICIENCY) == {"beta", "rho", "V", "sqm"}


@pytest.mark.parametrize("name", ["g", "r", "i", "A", "r+i"])
def test_unknown_filter_raises(name):
    """A removed or misspelt filter name fails loudly instead of being ignored."""
    with pytest.raises(ValueError, match="Unknown filters"):
        SystemThroughput(filters={name: True})


def test_canonical_band():
    """Old runs and stores call beta "g"; readers translate it and leave current names alone."""
    from argus_sim.throughput import canonical_band

    assert canonical_band("g") == "beta"
    assert canonical_band(np.str_("g")) == "beta"
    assert canonical_band("beta") == "beta" and canonical_band("rho") == "rho"
