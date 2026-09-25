"""Light-curve simulation over an ArgusSim exact-coverage survey run.

The visit store (``python -m argus_sim.lightcurve build RUN_DIR --out STORE``) holds everything a query needs::

    run = SurveyRun(store)                        # a store directory or s3://bucket/prefix (storage_options=...)
    vis = run.visits(ra, dec, band="g")           # one row per 60-s slot and OTA that covered the position
    err = run.sigma_mag(vis, 18.0)                # magnitude errors per slot
    lc = run.inject(vis, model, rng=np.random.default_rng(1))
    nightly = bin_lightcurve(lc, by="night")
"""

from .photometry import bin_lightcurve, inject, limit_mag, sigma_mag, snr, source_electrons
from .run import SurveyRun

__all__ = ["SurveyRun", "bin_lightcurve", "inject", "limit_mag", "sigma_mag", "snr", "source_electrons"]
