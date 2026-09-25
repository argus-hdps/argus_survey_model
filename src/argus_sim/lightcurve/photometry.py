"""The light-curve photometric model.

These are pure functions of a visit table and the band zero points. For a source of AB magnitude m in band b, one
frame of exposure t, throughput T and background noise sigma_bkg:

    S   = zp_b 10^(-0.4 m) QE_b t T           electrons
    var = sigma_bkg^2 + S                      (the source term can be switched off)
    snr = S / sqrt(var),   sigma_mag = sqrt((2.5 / ln 10 / snr)^2 + floor^2)

A slot is one 60-s exposure, or in bright time the stack of sixty 1-s frames. ``per="slot"`` (the default) gives
the numbers for the slot: signal n S and variance n var for its n_frames frames. ``per="frame"`` gives them for
one frame. The two differ only for fast slots.
"""

from __future__ import annotations

import inspect

import astropy.table as tbl
import numpy as np

MAG_PER_FRAC = 2.5 / np.log(10.0)
CADENCES = ("slot", "native")
PER = ("slot", "frame")
AB_MAG_1_UJY = 23.9


def _band_constants(band, zeropoints: dict) -> tuple[np.ndarray, np.ndarray]:
    band = np.asarray(band).astype(str)
    zp, qe = np.empty(len(band)), np.empty(len(band))
    for b in np.unique(band):
        if b not in zeropoints:
            raise KeyError(f"no zero point for band {b!r}; the zero points cover {sorted(zeropoints)}")
        sel = band == b
        zp[sel] = zeropoints[b]["zp_photons_per_sec"]
        qe[sel] = zeropoints[b]["band_qe"]
    return zp, qe


def _f(vis, name: str) -> np.ndarray:
    return np.asarray(vis[name], dtype=np.float64)


def _electrons_per_mag0(vis, zeropoints: dict) -> np.ndarray:
    """Electrons per frame from a source of magnitude 0: zp QE t T."""
    zp, qe = _band_constants(vis["band"], zeropoints)
    return zp * qe * _f(vis, "exptime_s") * _f(vis, "throughput")


def _n(vis, per: str) -> np.ndarray | float:
    if per not in PER:
        raise ValueError(f"per must be one of {PER}, not {per!r}")
    return _f(vis, "n_frames") if per == "slot" else 1.0


def source_electrons(vis, mag, zeropoints: dict) -> np.ndarray:
    """Source electrons in one frame of each row."""
    return _electrons_per_mag0(vis, zeropoints) * 10.0 ** (-0.4 * np.asarray(mag, dtype=np.float64))


def snr(vis, mag, zeropoints: dict, *, source_poisson: bool = True, per: str = "slot") -> np.ndarray:
    """Return the SNR of a source of magnitude ``mag`` (scalar or one per row) in each row's slot or frame."""
    n = _n(vis, per)
    s = source_electrons(vis, mag, zeropoints)
    var = _f(vis, "sigma_bkg_e") ** 2 + (s if source_poisson else 0.0)
    return s / np.sqrt(var) * np.sqrt(n)


def sigma_mag(
    vis, mag, zeropoints: dict, *, floor: float = 0.005, source_poisson: bool = True, per: str = "slot"
) -> np.ndarray:
    """Return the magnitude error per row: the SNR term and the systematic floor added in quadrature."""
    return np.hypot(MAG_PER_FRAC / snr(vis, mag, zeropoints, source_poisson=source_poisson, per=per), floor)


def limit_mag(
    vis, zeropoints: dict, *, nsigma: float = 5.0, source_poisson: bool = True, per: str = "slot"
) -> np.ndarray:
    """Return the magnitude at which each row's slot or frame reaches SNR ``nsigma``.

    Solves n S^2 = k^2 (sigma^2 + S) for the signal S per frame, with n frames (1 for ``per="frame"``).
    """
    k2 = float(nsigma) ** 2
    n = _n(vis, per)
    var_b = _f(vis, "sigma_bkg_e") ** 2
    if source_poisson:
        s = (k2 + np.sqrt(k2 * k2 + 4.0 * n * k2 * var_b)) / (2.0 * n)
    else:
        s = np.sqrt(k2 * var_b / n)
    return -2.5 * np.log10(s / _electrons_per_mag0(vis, zeropoints))


def _frames(vis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slot index, frame index within the slot and frame start mjd, for every frame of every slot."""
    n = np.asarray(vis["n_frames"], dtype=np.int64)
    slot_of = np.repeat(np.arange(len(vis)), n)
    k = np.arange(len(slot_of)) - np.repeat(np.cumsum(n) - n, n)
    mjd = _f(vis, "mjd")[slot_of] + k * _f(vis, "exptime_s")[slot_of] / 86400.0
    return slot_of, k, mjd


def _frame_table(vis, slot_of, k, mjd) -> tbl.Table:
    out = tbl.Table(vis[slot_of], copy=False)
    out["mjd"] = mjd
    out["n_frames"] = np.ones(len(out), dtype=np.asarray(vis["n_frames"]).dtype)
    out["frame"] = k
    return out


def _wants_vis(model) -> bool:
    try:
        return "vis" in inspect.signature(model).parameters
    except (TypeError, ValueError):
        return False


def _evaluate(model, mjd, band, rows) -> np.ndarray:
    if _wants_vis(model):
        m = model(mjd, band, vis=rows() if callable(rows) else rows)
    else:
        m = model(mjd, band)
    return np.asarray(m, dtype=np.float64) * np.ones(len(mjd))


INJECT_PASSTHROUGH = ("source", "tanseg_id", "subarray", "ratchet", "night", "slot", "frame", "n_frames")


def inject(
    vis,
    model,
    zeropoints: dict,
    *,
    rng: np.random.Generator,
    floor: float = 0.005,
    cadence: str = "slot",
    nsigma: float = 5.0,
    source_poisson: bool = True,
) -> tbl.Table:
    """Simulate photometry of a model light curve at the visits.

    ``model(mjd, band)`` returns the AB magnitude for arrays of frame start times and band names. If the model has
    a parameter named ``vis``, it is called as ``model(mjd, band, vis=rows)``, where ``rows`` holds the visit row
    of each evaluation (fast slots already expanded into frames, so ``len(rows) == len(mjd)``). A model can then
    depend on the exposure, for example on the seeing:

        def blended(mjd, band, vis):
            # a neighbour 3 arcsec away leaks more light into the aperture in poor seeing
            seeing = vis["seeing_zenith"] * vis["airmass"] ** 0.6
            return -2.5 * np.log10(10 ** (-0.4 * 18.0) + 0.05 * seeing * 10 ** (-0.4 * 17.0))

    ``cadence="slot"`` (the default) gives one row per slot, in the input order. A fast slot is the mean of its
    sixty 1-s frames, with the model evaluated at every frame. ``cadence="native"`` gives one row per frame: each
    fast slot becomes sixty rows at the slot start + k s, drawn independently from the slot's parameters.

    Fluxes are drawn in electrons per frame: flux_obs = flux_true (1 + f g1) + sqrt(var) g2, with g1, g2 unit
    normal and f = floor / (2.5 / ln 10). ``flux_err_e`` = sqrt(var + (f flux_true)^2) is the total error, so the
    pulls (flux_obs - flux_true) / flux_err are unit normal. Each output row gets the floor once; a fast slot's
    floor does not average down over its frames. The ``_ujy`` columns give the same fluxes in microjansky
    (AB 23.9 = 1 uJy); they do not depend on exposure time or throughput, so rows can be averaged.
    ``limit_mag`` is the row's nsigma limit, the same as ``limit_mag(vis)`` for slots.
    """
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, not {cadence!r}")
    frac_floor = float(floor) / MAG_PER_FRAC
    slot_of, k, frame_mjd = _frames(vis)
    frame_band = np.asarray(vis["band"]).astype(str)[slot_of]
    mag_frame = _evaluate(model, frame_mjd, frame_band, lambda: _frame_table(vis, slot_of, k, frame_mjd))
    if cadence == "native":
        rows = _frame_table(vis, slot_of, k, frame_mjd)
        per_mag0 = _electrons_per_mag0(rows, zeropoints)
        mag_true = mag_frame
        flux_true = per_mag0 * 10.0 ** (-0.4 * mag_true)
        var_stat = _f(rows, "sigma_bkg_e") ** 2 + (flux_true if source_poisson else 0.0)
        lim = limit_mag(rows, zeropoints, nsigma=nsigma, source_poisson=source_poisson, per="frame")
    else:
        rows = vis
        n = np.asarray(vis["n_frames"], dtype=np.int64)
        per_mag0 = _electrons_per_mag0(rows, zeropoints)
        s_frame = per_mag0[slot_of] * 10.0 ** (-0.4 * mag_frame)
        s_sum = np.bincount(slot_of, s_frame, minlength=len(rows))
        flux_true = s_sum / n
        single = n == 1
        with np.errstate(divide="ignore"):
            mag_true = -2.5 * np.log10(flux_true / per_mag0)
        mag_true[single] = mag_frame[single[slot_of]]
        var_stat = (n * _f(rows, "sigma_bkg_e") ** 2 + (s_sum if source_poisson else 0.0)) / n**2
        lim = limit_mag(rows, zeropoints, nsigma=nsigma, source_poisson=source_poisson, per="slot")
    band = np.asarray(rows["band"]).astype(str)
    g = rng.standard_normal((2, len(rows)))
    flux_obs = flux_true * (1.0 + frac_floor * g[0]) + np.sqrt(var_stat) * g[1]
    flux_err = np.sqrt(var_stat + (frac_floor * flux_true) ** 2)
    pos = flux_obs > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_obs = np.where(pos, -2.5 * np.log10(np.where(pos, flux_obs, 1.0) / per_mag0), np.nan)
        mag_err = np.where(pos, MAG_PER_FRAC * flux_err / np.where(pos, flux_obs, 1.0), np.nan)
    snr_obs = flux_obs / flux_err
    e_per_ujy = per_mag0 * 10.0 ** (-0.4 * AB_MAG_1_UJY)
    out = tbl.Table(
        {
            "mjd": _f(rows, "mjd"),
            "band": band,
            "tel_id": np.asarray(rows["tel_id"]),
            "exptime_s": _f(rows, "exptime_s"),
            "mag_true": mag_true,
            "flux_true_e": flux_true,
            "flux_obs_e": flux_obs,
            "flux_err_e": flux_err,
            "flux_true_ujy": flux_true / e_per_ujy,
            "flux_obs_ujy": flux_obs / e_per_ujy,
            "flux_err_ujy": flux_err / e_per_ujy,
            "mag_obs": mag_obs,
            "mag_err": mag_err,
            "snr_obs": snr_obs,
            "detected": snr_obs >= nsigma,
            "limit_mag": lim,
        }
    )
    for name in INJECT_PASSTHROUGH:
        if name in rows.colnames:
            out[name] = rows[name]
    out.meta.update(cadence=cadence, floor=float(floor), nsigma=float(nsigma), source_poisson=bool(source_poisson))
    return out


BIN_BY = ("night", "ratchet")


def bin_lightcurve(lc, by="night", *, nsigma: float = 5.0, floor: float = 0.0) -> tbl.Table:
    """Bin an ``inject`` light curve per band, as inverse-variance means of ``flux_obs_ujy``.

    ``by`` is ``"night"``, ``"ratchet"`` or a bin width in days (bins of ``floor((mjd - mjd.min()) / by)``). Every
    measurement in a bin is used, detected or not, and bands are never mixed. The binned error is that of the
    inverse-variance mean. The per-measurement errors already include each measurement's floor, and averaging
    reduces that floor as if it were random noise; real systematics may not average down. ``floor`` (mag, default
    0) adds a systematic to each binned flux in quadrature for users who want a floor per bin.

    Columns: ``mjd`` (mean of the bin's measurements), ``band``, ``n``, ``flux_ujy``, ``flux_err_ujy``, ``mag``
    (NaN when the flux is not positive), ``mag_err``, ``snr``, ``detected`` (snr >= nsigma), ``limit_mag`` (the
    magnitude of nsigma times the binned error), and ``night`` or ``ratchet`` for those bins.
    """
    mjd = _f(lc, "mjd")
    if isinstance(by, str):
        if by not in BIN_BY:
            raise ValueError(f"by must be one of {BIN_BY} or a bin width in days, not {by!r}")
        key = np.asarray(lc[by], dtype=np.int64)
    else:
        width = float(by)
        if not width > 0:
            raise ValueError(f"the bin width must be positive, not {by!r}")
        key = np.floor((mjd - mjd.min()) / width).astype(np.int64) if len(mjd) else np.zeros(0, np.int64)
    band = np.asarray(lc["band"]).astype(str)
    names, band_idx = np.unique(band, return_inverse=True)
    groups, inv = np.unique(np.stack([band_idx, key]), axis=1, return_inverse=True)
    inv = inv.ravel()
    ng = groups.shape[1]
    w = 1.0 / _f(lc, "flux_err_ujy") ** 2
    sw = np.bincount(inv, w, minlength=ng)
    flux = np.bincount(inv, w * _f(lc, "flux_obs_ujy"), minlength=ng) / sw
    err = np.sqrt(1.0 / sw + (float(floor) / MAG_PER_FRAC * flux) ** 2)
    n = np.bincount(inv, minlength=ng)
    pos = flux > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(pos, AB_MAG_1_UJY - 2.5 * np.log10(np.where(pos, flux, 1.0)), np.nan)
        mag_err = np.where(pos, MAG_PER_FRAC * err / np.where(pos, flux, 1.0), np.nan)
    out = tbl.Table(
        {
            "mjd": np.bincount(inv, mjd, minlength=ng) / n,
            "band": names[groups[0]] if ng else np.zeros(0, str),
            "n": n,
            "flux_ujy": flux,
            "flux_err_ujy": err,
            "mag": mag,
            "mag_err": mag_err,
            "snr": flux / err,
            "detected": flux / err >= nsigma,
            "limit_mag": AB_MAG_1_UJY - 2.5 * np.log10(nsigma * err),
        }
    )
    if isinstance(by, str):
        out[by] = groups[1]
    out = out[np.lexsort((out["band"], out["mjd"]))]
    out.meta.update(by=by, nsigma=float(nsigma), floor=float(floor))
    return out
