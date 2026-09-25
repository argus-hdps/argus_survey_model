"""On-sky (full-system) coadd depth from per-ratchet survey rows.

For a source of flux ``F`` seen through signal throughput ``T_i`` in exposure ``i`` with background noise
``sigma_i``, the inverse-variance coadd has ``SNR^2 = F^2 * sum(T_i^2 / sigma_i^2)``, so the flux reaching SNR
``snr`` is ``snr / sqrt(sum(T_i^2 / sigma_i^2))``.

``inv_above`` accumulates with ``T_i = 1``: the above-atmosphere, above-optics quantity, as in a cadence map
ingested without ``signal_tp``.  ``inv_onsky`` accumulates with the transmission the caller passes.  Reported
depths are through the full system: pass :func:`system_transmission`, optics x atmosphere at each row's airmass,
which is what the Survey worker's ``signal_tp`` and the exact engine use.  :func:`atmospheric_transmission` alone
gives an atmosphere-only depth (above the optics).
"""

from __future__ import annotations

import numpy as np

from .observatory import pickering_airmass


def atmospheric_transmission(alt_deg: np.ndarray, atm_tp_am1: float) -> np.ndarray:
    """Band-mean transmission at the given altitudes, Beer-Lambert in airmass.

    Uses the survey's own convention (``survey.py``): the band-averaged
    transmission at airmass 1 raised to the Pickering airmass.
    """
    return np.asarray(atm_tp_am1, dtype=np.float64) ** pickering_airmass(np.asarray(alt_deg, dtype=np.float64))


def system_transmission(alt_deg: np.ndarray, band_params: dict) -> np.ndarray:
    """Full-system signal throughput at each row.

    This is the exact efficiency eta(X) / band_qe at the row's Pickering airmass, for one band's
    ``build_band_params`` entry, as the Survey worker uses it.
    """
    from .spectral_sky import signal_throughput

    return signal_throughput(band_params, pickering_airmass(np.asarray(alt_deg, dtype=np.float64)))


def extinction_coefficient(atm_tp_am1: float) -> float:
    """Magnitudes per airmass implied by a band-mean transmission at airmass 1."""
    return float(-2.5 * np.log10(atm_tp_am1))


def accumulate_onsky(
    healpix: np.ndarray,
    noise: np.ndarray,
    source_shot: np.ndarray,
    transmission: np.ndarray,
    n_epochs: int,
    exptime_s: float,
    airmass_per_row: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Accumulate one ratchet's rows into per-pixel sums without and with the given transmission.

    ``inv_above`` has no transmission and ``inv_onsky`` has ``transmission``; pass :func:`system_transmission`
    for full-system depth.

    Mirrors ``CadenceAccumulator.ingest``: rows whose background variance is
    not positive are dropped, and every row stands for ``n_epochs`` identical
    exposures.

    Returns
    -------
    dict
        ``pixel``, ``inv_above`` (sum of n / sigma^2), ``inv_onsky``
        (sum of n T^2 / sigma^2), ``n_obs``, ``exptime_s``, and
        ``airmass_weight`` (sum of n * X, for a coverage-weighted mean airmass
        once divided by ``n_obs``).

    """
    noise = np.asarray(noise, dtype=np.float64)
    source_shot = np.asarray(source_shot, dtype=np.float64)
    bkg_sq = np.maximum(noise**2 - source_shot**2, 0.0)
    valid = bkg_sq > 0
    hpx = np.asarray(healpix)[valid]
    inv = 1.0 / bkg_sq[valid]
    t2 = np.asarray(transmission, dtype=np.float64)[valid] ** 2
    if airmass_per_row is None:
        airmass_per_row = np.zeros(len(noise))
    airmass = np.asarray(airmass_per_row, dtype=np.float64)[valid]
    pixel, inverse = np.unique(hpx, return_inverse=True)
    out = {
        "pixel": pixel.astype(np.int64),
        "inv_above": np.zeros(len(pixel)),
        "inv_onsky": np.zeros(len(pixel)),
        "n_obs": np.zeros(len(pixel), dtype=np.int64),
        "exptime_s": np.zeros(len(pixel)),
        "airmass_weight": np.zeros(len(pixel)),
    }
    np.add.at(out["inv_above"], inverse, inv)
    np.add.at(out["inv_onsky"], inverse, inv * t2)
    np.add.at(out["n_obs"], inverse, 1)
    np.add.at(out["airmass_weight"], inverse, airmass)
    for key in ("inv_above", "inv_onsky", "n_obs", "exptime_s", "airmass_weight"):
        out[key] = out[key] * n_epochs
    out["exptime_s"] = out["n_obs"] * float(exptime_s)
    return out


def limmag_from_inv_sum(
    inv_sum: np.ndarray,
    n_obs: np.ndarray,
    exptime_s: np.ndarray,
    snr: float,
    ab,
    band: str,
    band_qe: float,
) -> np.ndarray:
    """Limiting magnitude from an accumulated inverse-variance sum.

    The same conversion as ``cadence_accumulator.limmag_from_map`` applied to
    either the above-atmosphere or the on-sky sum.
    """
    import astropy.units as u

    from .cadence_accumulator import _qe_as_float

    signal_electrons = snr / np.sqrt(np.asarray(inv_sum, dtype=np.float64))
    exptime_per_obs = np.asarray(exptime_s, dtype=np.float64) / np.asarray(n_obs, dtype=np.float64)
    photons_per_sec = signal_electrons / exptime_per_obs / _qe_as_float(band_qe)
    mags = ab.mag_from_photons(band, photons_per_sec * (u.photon / u.second))
    return np.asarray(getattr(mags, "value", mags), dtype=np.float64)
