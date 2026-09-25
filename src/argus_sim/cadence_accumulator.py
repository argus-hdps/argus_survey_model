"""Cadence accumulation utilities.

Maps post-culled ratchet numbers to window IDs at each configured
cadence level (ratchet, hour, night, season).  Window IDs are contiguous
integers starting from 0, assigned in chronological order.

The CadenceAccumulator class ingests per-ratchet DataFrames and
accumulates background-only inverse-variance depth into HealSparse
recarray maps, one per cadence window.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Set, Union

import astropy.units as u
import healsparse as hsp
import numpy as np
import numpy.typing as npt

from .provenance import git_state, write_provenance

if TYPE_CHECKING:
    from .ab_system import ABPhot


DEPTH_DTYPE = np.dtype(
    [
        ("inv_bkg_noise_sq_sum", np.float64),
        ("n_obs", np.int32),
        ("exptime_s", np.float64),
    ]
)

# Width of each cadence bucket in fractional days
_CADENCE_WIDTH_DAYS = {
    "ratchet": None,  # 1:1 mapping, each ratchet is its own window
    "15min": 15.0 / 1440.0,
    "hour": 1.0 / 24.0,
    "night": None,  # handled via the nights array directly
    "week": 7.0,
    "season": 90.0,
}


def build_window_registry(
    ratchet_nums: Union[npt.ArrayLike, Sequence[int]],
    mjds: Union[npt.ArrayLike, Sequence[float]],
    nights: Union[npt.ArrayLike, Sequence[int]],
    cadence_levels: List[str],
) -> Dict[str, Dict[int, int]]:
    """Map post-culled ratchet numbers to window IDs at each cadence level.

    Parameters
    ----------
    ratchet_nums : array-like of int
        Post-downsampled ratchet indices (after weather and
        fraction_of_ratchets_to_run culling).
    mjds : array-like of float
        MJD timestamp for each ratchet, same length as ratchet_nums.
    nights : array-like of int
        Night index for each ratchet (from the sun-altitude grouping
        in Survey.run), same length as ratchet_nums.
    cadence_levels : list of str
        Cadence levels to compute, e.g. ["hour", "night", "season"].

    Returns
    -------
    dict[str, dict[int, int]]
        Outer key is the cadence level string.  Inner dict maps each
        ratchet_num to its zero-indexed, contiguous window_id.

    """
    ratchet_nums = np.asarray(ratchet_nums)
    mjds = np.asarray(mjds, dtype=float)
    nights = np.asarray(nights)

    registry: Dict[str, Dict[int, int]] = {}

    for level in cadence_levels:
        if len(ratchet_nums) == 0:
            registry[level] = {}
            continue

        if level == "ratchet":
            raw_keys = ratchet_nums
        elif level == "night":
            raw_keys = nights
        else:
            width = _CADENCE_WIDTH_DAYS[level]
            raw_keys = np.floor(mjds / width).astype(np.int64)

        unique_keys, inverse = np.unique(raw_keys, return_inverse=True)
        # inverse already maps each element to a contiguous 0..N-1 range
        # because np.unique returns sorted unique values
        registry[level] = {int(r): int(inverse[i]) for i, r in enumerate(ratchet_nums)}

    return registry


def _extract_signal_rate(
    depth_map: hsp.HealSparseMap,
    snr: float,
    band_qe: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pixels and detection-threshold photon rates from a depth map.

    Shared computation for both limmag conversion functions: reads valid
    pixels, inverts accumulated inverse-variance to coadd noise, and
    converts the electron-level signal to a per-second photon rate.

    Returns
    -------
    pixels : np.ndarray
        Valid HEALPix pixel indices from the depth map.
    signal_photons_per_sec : np.ndarray
        Photon rate (photons/s) corresponding to the given SNR threshold
        at each pixel, as a dimensionless float array.

    """
    pixels = depth_map.valid_pixels
    vals = depth_map.get_values_pix(pixels)

    coadd_noise = 1.0 / np.sqrt(vals["inv_bkg_noise_sq_sum"])
    signal_electrons = snr * coadd_noise
    exptime_per_obs = vals["exptime_s"] / vals["n_obs"]
    signal_photons_per_sec = signal_electrons / exptime_per_obs / _qe_as_float(band_qe)

    return pixels, signal_photons_per_sec


def _qe_as_float(band_qe: float | u.Quantity) -> float:
    """Band QE as a plain electrons-per-photon float.

    Callers that average ``SystemThroughput.qe_freq`` get a Quantity in
    electron / photon; left in place it turns the photon rate into a
    non-dimensionless Quantity and ``mag_from_photons`` cannot take its log.
    """
    if isinstance(band_qe, u.Quantity):
        try:
            return float(band_qe.to_value(u.electron / u.photon))
        except u.UnitConversionError:
            return float(band_qe.to_value(u.dimensionless_unscaled))
    return float(band_qe)


def limmag_from_map(
    depth_map: hsp.HealSparseMap,
    snr: float,
    ab: ABPhot,
    band: str,
    band_qe: float,
) -> np.ndarray:
    """Convert an accumulated depth map to per-pixel limiting magnitudes.

    Parameters
    ----------
    depth_map : healsparse.HealSparseMap
        Recarray map with DEPTH_DTYPE fields (inv_bkg_noise_sq_sum,
        n_obs, exptime_s).
    snr : float
        Signal-to-noise ratio threshold.  Higher SNR yields brighter
        (numerically smaller) limiting magnitudes.
    ab : ABPhot
        AB photometry converter for the instrument.
    band : str
        Filter band name, passed to ``ab.mag_from_photons``.
    band_qe : float
        Band-averaged quantum efficiency (electrons per photon).

    Returns
    -------
    np.ndarray
        Structured array with fields ``pixel`` (int64) and ``limmag``
        (float64), one row per valid pixel in the depth map.

    """
    pixels, signal_photons_per_sec = _extract_signal_rate(depth_map, snr, band_qe)

    mags = ab.mag_from_photons(band, signal_photons_per_sec * (u.photon / u.second))
    if hasattr(mags, "value"):
        mags = mags.value

    out_dtype = np.dtype([("pixel", np.int64), ("limmag", np.float64)])
    out = np.empty(len(pixels), dtype=out_dtype)
    out["pixel"] = pixels
    out["limmag"] = mags
    return out


def limmag_from_map_zp(
    depth_map: hsp.HealSparseMap,
    snr: float,
    zp_photons_per_sec: float,
    band_qe: float,
) -> np.ndarray:
    """Convert a depth map to limiting magnitudes using stored zero-points.

    Unlike ``limmag_from_map``, this function does not require an ABPhot
    instance, only the scalar zero-point and QE values stored in the
    cadence manifest.

    Parameters
    ----------
    depth_map : healsparse.HealSparseMap
        Recarray map with DEPTH_DTYPE fields.
    snr : float
        Detection threshold (e.g. 5.0 for 5σ).
    zp_photons_per_sec : float
        Above-atmosphere photon rate for a 0th-mag AB source (photons/s).
    band_qe : float
        Band-averaged quantum efficiency (electrons per photon).

    Returns
    -------
    np.ndarray
        Structured array with ``pixel`` (int64) and ``limmag`` (float64).

    """
    pixels, signal_photons_per_sec = _extract_signal_rate(depth_map, snr, band_qe)

    mags = -2.5 * np.log10(signal_photons_per_sec / zp_photons_per_sec)

    out_dtype = np.dtype([("pixel", np.int64), ("limmag", np.float64)])
    out = np.empty(len(pixels), dtype=out_dtype)
    out["pixel"] = pixels
    out["limmag"] = mags
    return out


def _make_empty_depth_map(nside_sparse: int) -> hsp.HealSparseMap:
    """Create an empty HealSparse recarray map with DEPTH_DTYPE."""
    return hsp.HealSparseMap.make_empty(
        nside_coverage=32,
        nside_sparse=nside_sparse,
        dtype=DEPTH_DTYPE,
        primary="inv_bkg_noise_sq_sum",
    )


def _accumulate_into_map(
    hsmap: hsp.HealSparseMap,
    pixels: np.ndarray,
    inv_bkg_noise_sq: np.ndarray,
    exptime_s: np.ndarray,
    counts: np.ndarray | None = None,
) -> None:
    """Add inverse-variance contributions to an existing depth map in place.

    For pixels already in the map the values are summed; for new pixels
    the contributions are inserted directly.
    """
    if len(pixels) == 0:
        return

    if counts is None:
        counts = np.ones(len(pixels), dtype=np.int32)

    existing = hsmap.get_values_pix(pixels)
    existing_valid = existing["inv_bkg_noise_sq_sum"] > 0

    vals = np.zeros(len(pixels), dtype=DEPTH_DTYPE)
    vals["inv_bkg_noise_sq_sum"] = np.where(
        existing_valid,
        existing["inv_bkg_noise_sq_sum"] + inv_bkg_noise_sq,
        inv_bkg_noise_sq,
    )
    vals["n_obs"] = np.where(
        existing_valid,
        existing["n_obs"] + counts,
        counts,
    )
    vals["exptime_s"] = np.where(
        existing_valid,
        existing["exptime_s"] + exptime_s,
        exptime_s,
    )
    hsmap.update_values_pix(pixels, vals)


class CadenceAccumulator:
    """Accumulates background-only inverse-variance depth per cadence window.

    Parameters
    ----------
    window_registry : dict[str, dict[int, int]]
        From :func:`build_window_registry`.  Maps cadence level -> ratchet_num -> window_id.
    cadence_levels : list of str
        Cadence levels to accumulate (must be keys in *window_registry*).
    nside_sparse : int
        HEALPix NSIDE for the output HealSparse maps.
    output_dir : str
        Directory where .hsp files and cadence_manifest.json are written.
    mjds : dict[int, float] | None
        Optional mapping of ratchet_num -> MJD, used for manifest metadata.

    """

    def __init__(
        self,
        window_registry: Dict[str, Dict[int, int]],
        cadence_levels: List[str],
        nside_sparse: int,
        output_dir: str,
        mjds: Dict[int, float] | None = None,
        band_zeropoints: Dict[str, Dict[str, float]] | None = None,
        config_fingerprint: str | None = None,
    ) -> None:
        """Initialize accumulator from a pre-built window registry.

        Parameters
        ----------
        window_registry : dict[str, dict[int, int]]
            From :func:`build_window_registry`.
        cadence_levels : list of str
            Cadence levels to accumulate (keys of ``window_registry``).
        nside_sparse : int
            HEALPix NSIDE for the output HealSparse maps.
        output_dir : str
            Directory for the .hsp files and cadence_manifest.json.
        mjds : dict[int, float], optional
            Ratchet number to MJD, for the manifest.
        band_zeropoints : dict, optional
            Per-band photometric zero-points for self-describing outputs.
            Keys are band names, values are dicts with ``zp_photons_per_sec``
            (photons/s at AB mag 0) and ``band_qe`` (electrons per photon).
        config_fingerprint : str, optional
            Opaque hash identifying the simulation configuration, stored
            in the manifest provenance block for reproducibility.

        """
        self._registry = window_registry
        self._levels = cadence_levels
        self._nside = nside_sparse
        self._output_dir = output_dir
        self._mjds = mjds or {}
        self._band_zeropoints = band_zeropoints or {}
        self._config_fingerprint = config_fingerprint

        # {level: {window_id: {band: HealSparseMap}}}
        self._maps: Dict[str, Dict[int, Dict[str, hsp.HealSparseMap]]] = {level: {} for level in cadence_levels}

        # Track which ratchets have been ingested or skipped per level,
        # grouped by window_id, so finalize knows when a window is complete.
        self._ingested: Dict[str, Set[int]] = {level: set() for level in cadence_levels}

        # Per-window MJD tracking for the manifest
        self._window_mjds: Dict[str, Dict[int, List[float]]] = {level: defaultdict(list) for level in cadence_levels}

        # Invert registry: level -> window_id -> {ratchet_nums}
        self._ratchets_per_window: Dict[str, Dict[int, Set[int]]] = {}
        for level in cadence_levels:
            inverted: Dict[int, Set[int]] = defaultdict(set)
            for ratchet_num, window_id in window_registry[level].items():
                inverted[window_id].add(ratchet_num)
            self._ratchets_per_window[level] = dict(inverted)

        # Tracking for already-flushed windows
        self._flushed: Dict[str, Set[int]] = {level: set() for level in cadence_levels}
        self._flushed_meta: Dict[str, List[Dict[str, Any]]] = {level: [] for level in cadence_levels}

    def _get_or_create_map(self, level: str, window_id: int, band: str) -> hsp.HealSparseMap:
        if window_id not in self._maps[level]:
            self._maps[level][window_id] = {}
        if band not in self._maps[level][window_id]:
            self._maps[level][window_id][band] = _make_empty_depth_map(self._nside)
        return self._maps[level][window_id][band]

    def _check_and_flush(self, ratchet_num: int) -> None:
        """Flush completed windows to disk and free their memory."""
        for level in self._levels:
            window_id = self._registry[level].get(ratchet_num)
            if window_id is None or window_id in self._flushed[level]:
                continue
            expected = self._ratchets_per_window[level][window_id]
            if expected.issubset(self._ingested[level]):
                self._flush_window(level, window_id)

    def _flush_window(self, level: str, window_id: int) -> None:
        """Write maps for a completed window to disk and free memory."""
        band_maps = self._maps[level].pop(window_id, {})
        self._flushed[level].add(window_id)
        if not band_maps:
            return

        mjd_list = sorted(self._window_mjds[level].pop(window_id, []))

        if level == "ratchet" and mjd_list:
            from astropy.time import Time

            date_str = Time(mjd_list[0], format="mjd").datetime.strftime("%Y%m%d")
            level_dir = os.path.join(self._output_dir, "ratchet", date_str)
        else:
            level_dir = os.path.join(self._output_dir, level)
        os.makedirs(level_dir, exist_ok=True)

        band_files: List[Dict[str, Any]] = []
        total_pixels = 0
        for band in sorted(band_maps.keys()):
            hsmap = band_maps[band]
            fname = f"depth_{level}_w{window_id:04d}_{band}.hsp"
            fpath = os.path.join(level_dir, fname)
            hsmap.write(fpath, clobber=True)
            n_pix = int(hsmap.valid_pixels.size)
            total_pixels += n_pix
            band_files.append(
                {
                    "band": band,
                    "file": os.path.relpath(fpath, self._output_dir),
                    "n_pixels": n_pix,
                }
            )

        meta: Dict[str, Any] = {
            "window_id": window_id,
            "bands": band_files,
            "n_pixels": total_pixels,
        }
        if mjd_list:
            meta["mjd_min"] = mjd_list[0]
            meta["mjd_max"] = mjd_list[-1]

        self._flushed_meta[level].append(meta)

    def ingest(self, ratchet_num: int, data: dict, n_epochs: int = 1) -> None:
        """Ingest a ratchet result and accumulate depth.

        Parameters
        ----------
        ratchet_num : int
            The ratchet index (must be a key in the window registry).
        data : dict
            Ratchet result with keys ``healpix``, ``noise``,
            ``source_shot``, ``exptime_s``, and ``band``, and optionally
            ``signal_tp`` (full-system signal throughput per row; when
            present the accumulated sum is ``T^2 / sigma_bkg^2``).  Arrays
            must all share the same leading dimension.
        n_epochs : int
            Multiplier for this ratchet.  All epochs in a ratchet share
            the same noise properties, so ingesting once with n_epochs=N
            is equivalent to ingesting the same data N times.

        """
        hpx = np.asarray(data["healpix"])
        noise = np.asarray(data["noise"], dtype=np.float64)
        source_shot = np.asarray(data["source_shot"], dtype=np.float64)

        bkg_noise_sq = noise**2 - source_shot**2
        bkg_noise_sq = np.maximum(bkg_noise_sq, 0.0)
        valid = bkg_noise_sq > 0
        if not np.any(valid):
            self.skip(ratchet_num)
            return

        hpx = hpx[valid]
        bkg_noise_sq = bkg_noise_sq[valid]
        inv_bkg_noise_sq = 1.0 / bkg_noise_sq
        # Full-system depth: weight each exposure by the square of its signal throughput
        # (optics x atmosphere at the row's airmass), when the ratchet result carries it.  Without it the sum is
        # the above-atmosphere, above-optics quantity.
        if data.get("signal_tp") is not None:
            inv_bkg_noise_sq = inv_bkg_noise_sq * np.asarray(data["signal_tp"], dtype=np.float64)[valid] ** 2

        n_valid = int(np.count_nonzero(valid))
        exptime_s = np.full(n_valid, float(data["exptime_s"]))

        bands = np.asarray(data["band"])[valid]

        mjd = self._mjds.get(ratchet_num)

        for band in np.unique(bands):
            bmask = bands == band
            bhpx = hpx[bmask]
            binv = inv_bkg_noise_sq[bmask]
            bexp = exptime_s[bmask]

            unique_pix, inverse = np.unique(bhpx, return_inverse=True)
            agg_inv = np.zeros(len(unique_pix), dtype=np.float64)
            agg_exp = np.zeros(len(unique_pix), dtype=np.float64)
            agg_counts = np.zeros(len(unique_pix), dtype=np.int32)
            np.add.at(agg_inv, inverse, binv)
            np.add.at(agg_exp, inverse, bexp)
            np.add.at(agg_counts, inverse, 1)

            if n_epochs != 1:
                agg_inv *= n_epochs
                agg_exp *= n_epochs
                agg_counts *= n_epochs

            for level in self._levels:
                window_id = self._registry[level].get(ratchet_num)
                if window_id is None:
                    continue
                hsmap = self._get_or_create_map(level, window_id, band)
                _accumulate_into_map(hsmap, unique_pix.astype(np.int64), agg_inv, agg_exp, agg_counts)
                self._ingested[level].add(ratchet_num)
                if mjd is not None:
                    self._window_mjds[level][window_id].append(mjd)

        self._check_and_flush(ratchet_num)

    def skip(self, ratchet_num: int) -> None:
        """Mark a ratchet as processed without contributing data.

        This is used for ratchets that returned None (e.g., bright time
        or sun too high) so that finalize can correctly determine when
        all ratchets in a window have been processed.
        """
        for level in self._levels:
            self._ingested[level].add(ratchet_num)
        self._check_and_flush(ratchet_num)

    def finalize(self) -> Dict[str, Any]:
        """Flush remaining windows to .hsp files and write manifest.

        Most windows will already have been flushed incrementally during
        ingestion.  This handles any stragglers and assembles the manifest.

        Returns
        -------
        dict
            The cadence manifest that was written to disk.

        """
        os.makedirs(self._output_dir, exist_ok=True)

        for level in self._levels:
            for window_id in list(self._maps[level].keys()):
                self._flush_window(level, window_id)

        provenance: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "zeropoint_assumptions": (
                "Above-atmosphere AB photon rates; no atmospheric extinction applied (airmass=0 equivalent)"
            ),
        }
        try:
            from . import __version__

            provenance["argus_sim_version"] = __version__
        except Exception:
            pass
        git = git_state()
        if git["git_hash"] is not None:
            provenance["git_hash"] = git["git_hash_short"]
            provenance["git_hash_full"] = git["git_hash"]
            provenance["git_dirty"] = git["git_dirty"]
        if self._config_fingerprint is not None:
            provenance["config_fingerprint"] = self._config_fingerprint

        write_provenance(self._output_dir, "survey", {"config_fingerprint": self._config_fingerprint})

        manifest: Dict[str, Any] = {
            "provenance": provenance,
            "nside_sparse": self._nside,
            "band_zeropoints": self._band_zeropoints,
            "cadence_levels": {},
        }

        for level in self._levels:
            windows_meta = sorted(self._flushed_meta[level], key=lambda m: m["window_id"])
            manifest["cadence_levels"][level] = {
                "n_windows": len(windows_meta),
                "windows": windows_meta,
            }

        manifest_path = os.path.join(self._output_dir, "cadence_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
