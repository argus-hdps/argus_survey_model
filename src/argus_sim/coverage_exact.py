"""Exact per-point OTA coverage for the Argus ring layout.

Given sky positions and a time span, report which OTA (and so which subarray and band) covers each position
in each exposure, using the gnomonic membership test that array_arrangement defines for the layout: a unit
vector p is inside the OTA centred on c when p.c > 0, |p.e_t / p.c| < tan(long/2) and |p.e_r / p.c| < tan(short/2),
with e_t = norm(up x c) (along-ring) and e_r = c x e_t (radial).

Tracking model: every subarray is rotated rigidly about the true celestial polar axis at the site latitude.  A
rotation about the polar axis is a shift in hour angle, so a position at hour angle H is seen at array-frame hour
angle H + theta(t).  The ratchet runs theta down from ``theta_start_deg`` at the sidereal rate (the Earth rotation
angle's rate, so a tracked position holds exactly still in the array frame) for ``track_min``, about 3.51 deg, then
resets during ``reset_min`` with no exposure.  All eight subarrays are synchronized, so they share theta.

Input RA/Dec are CIRS, as in the rest of ArgusSim, and hour angle is formed from the Earth rotation angle.
With ``apparent=True`` they are taken as ICRS and moved to CIRS (precession, nutation, aberration) at each
night's midpoint.  With ``refraction=True`` the apparent altitude is raised by Saemundsson's formula scaled to
the site pressure and temperature (790 mb, median night 10 C).  Both flags default off.

Arrays are processed with CuPy when a working GPU is present and NumPy otherwise (``backend="auto"``); both
paths run the same float64 arithmetic on the same candidate table, and ``tests/test_coverage_exact.py``
checks that they agree.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import astropy.coordinates as crds
import astropy.time as atime
import astropy.units as u
import numpy as np
from scipy.spatial import cKDTree

from argus_sim import c, get_logger

log = get_logger(__name__)

SIDEREAL_DEG_PER_MIN = 360.98564736629 / 1440.0  # Earth rotation angle rate
_GRID_STEP = 0.25  # deg, candidate-table cell in zenith distance and azimuth
_ZD_MAX = 56.0  # deg, beyond the footprint edge (52.48) with margin


def get_backend(backend: str = "auto"):
    """Return the array module: CuPy if requested or available with a working device, else NumPy."""
    if backend == "numpy":
        return np
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1).sum()
        return cp
    except Exception as err:  # no cupy, no device, or a driver/runtime mismatch
        if backend == "cupy":
            raise RuntimeError(
                f"The CuPy backend needs CuPy (the gpu extra) and an NVIDIA GPU with a working CUDA driver: {err}"
            ) from err
        return np


@dataclasses.dataclass(frozen=True)
class Schedule:
    """Ratchet cadence.  Blocks of ``track_min + reset_min`` start at ``phase_ref`` + k * period."""

    exptime_s: float = 60.0
    track_min: float = 15.0  # 15 exposures per ratchet; the block is 16 min with the reset
    reset_min: float = 1.0
    theta_start_deg: float = 3.75
    sun_alt_max_deg: float = -18.0
    phase_ref: str = "2000-01-01T00:00:00"

    @property
    def period_min(self) -> float:
        """Length of one ratchet block (tracking plus reset), minutes."""
        return self.track_min + self.reset_min

    def exposures(self, t0: atime.Time, t1: atime.Time, location: crds.EarthLocation) -> dict:
        """Mid-exposure times in [t0, t1) with the Sun below ``sun_alt_max_deg``, and their ratchet index and theta."""
        ref = atime.Time(self.phase_ref, scale="utc")
        per = self.period_min / 1440.0
        k0 = int(np.floor((t0.mjd - ref.mjd) / per))
        k1 = int(np.ceil((t1.mjd - ref.mjd) / per))
        n_exp = int(np.floor(self.track_min * 60.0 / self.exptime_s + 1e-9))
        k = np.repeat(np.arange(k0, k1 + 1), n_exp)
        i = np.tile(np.arange(n_exp), k1 - k0 + 1)
        off_min = (i + 0.5) * self.exptime_s / 60.0
        mjd = ref.mjd + k * per + off_min / 1440.0
        keep = (mjd >= t0.mjd) & (mjd < t1.mjd)
        mjd, k, off_min = mjd[keep], k[keep], off_min[keep]
        t = atime.Time(mjd, format="mjd", scale="utc")
        if len(t):
            sun = crds.get_body("sun", t).transform_to(crds.AltAz(obstime=t, location=location)).alt.deg
            dark = sun < self.sun_alt_max_deg
            t, k, off_min = t[dark], k[dark], off_min[dark]
        theta = self.theta_start_deg - SIDEREAL_DEG_PER_MIN * off_min
        return {"time": t, "ratchet": k, "theta_deg": theta}


@dataclasses.dataclass
class Layout:
    """OTA centres (alt/az, compass azimuth) with tel_id, subarray and band, plus the camera field."""

    tel_id: np.ndarray
    subarray: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    band: np.ndarray
    fov_deg: tuple[float, float]
    name: str = ""

    @classmethod
    def from_config(cls, layout: str | None = None, options: list[str] | None = None) -> "Layout":
        """Build from a registered ring layout and filter cycle, assigning bands as CradlePointing does."""
        from .ring_layout import assign_filters, assign_filters_from_table, load_ring_layout, table_applies

        name = layout if layout is not None else c.packing_strategy.layout_file
        tab = load_ring_layout(name)
        options = list(options if options is not None else c.filter_strategy.options)
        cam = tab.copy()
        cam["alt_center"] = 90.0 - np.asarray(tab["zenith_dist"])
        cam["az_center"] = np.asarray(tab["az"])
        if tab.meta.get("band_table") and table_applies(options):
            cam = assign_filters_from_table(cam, options, tab.meta["band_table"])
        else:
            cell = (
                c.filter_strategy.cell_size
                if c.filter_strategy.cell_size is not None
                else c.telescope.short_axis_extent
            )
            cam = assign_filters(cam, options, cell)
        fov = tab.meta.get("fov_deg") or (c.telescope.long_axis_extent, c.telescope.short_axis_extent)
        return cls(
            tel_id=np.asarray(tab["tel_id"]),
            subarray=np.asarray(tab["subarray_n"]),
            alt_deg=np.asarray(cam["alt_center"], float),
            az_deg=np.asarray(cam["az_center"], float),
            band=np.asarray(cam["filter"]).astype(str),
            fov_deg=(float(fov[0]), float(fov[1])),
            name=str(tab.meta.get("layout", "")),
        )

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return unit vectors (north, east, up) of each OTA's centre and its along-ring and radial axes."""
        C = _unit(self.alt_deg, self.az_deg, np)
        et = np.cross([0.0, 0.0, 1.0], C)
        et /= np.linalg.norm(et, axis=1, keepdims=True)
        er = np.cross(C, et)
        return C, et, er


def _unit(alt_deg, az_deg, xp):
    a, z = xp.radians(alt_deg), xp.radians(az_deg)
    return xp.stack([xp.cos(a) * xp.cos(z), xp.cos(a) * xp.sin(z), xp.sin(a)], axis=-1)


@lru_cache(maxsize=8)
def _candidate_table(key: tuple) -> tuple[np.ndarray, int]:
    """For each (zd, az) cell of _GRID_STEP, every OTA whose centre is within reach of any point of the cell.

    ``key`` is (alt tuple, az tuple, fov[, extra_deg]); ``extra_deg`` widens the reach for objects larger than
    a point centred in the cell (a TAN segment's half-diagonal, in tanseg_coverage).
    """
    alt, az, fov = np.array(key[0]), np.array(key[1]), key[2]
    extra = key[3] if len(key) > 3 else 0.0
    C = _unit(alt, az, np)
    half_diag = np.degrees(np.arctan(np.hypot(np.tan(np.radians(fov[0] / 2)), np.tan(np.radians(fov[1] / 2)))))
    nz, na = int(round(_ZD_MAX / _GRID_STEP)), int(round(360.0 / _GRID_STEP))
    zc = (np.arange(nz) + 0.5) * _GRID_STEP
    ac = (np.arange(na) + 0.5) * _GRID_STEP
    Z, A = np.meshgrid(zc, ac, indexing="ij")
    P = _unit(90.0 - Z.ravel(), A.ravel(), np)
    # the cell's circumradius on the sphere is at most the half-diagonal of a flat cell, plus margin
    rad = half_diag + np.hypot(_GRID_STEP / 2, _GRID_STEP / 2) + 0.05 + extra
    chord = 2 * np.sin(np.radians(rad) / 2)
    lists = cKDTree(C).query_ball_point(P, chord)
    K = max(len(x) for x in lists)
    table = np.full((nz * na, K), -1, dtype=np.int32)
    for i, x in enumerate(lists):
        table[i, : len(x)] = sorted(x)
    return table.reshape(nz, na, K), K


def _members(layout: Layout, alt_deg, az_deg, xp, chunk: int = 200_000):
    """Exact membership for positions given in the array frame (alt/az).  Returns (row index, OTA row index) pairs."""
    key = (tuple(layout.alt_deg), tuple(layout.az_deg), layout.fov_deg)
    table_np, K = _candidate_table(key)
    table = xp.asarray(table_np)
    C_np, et_np, er_np = layout.basis()
    C, et, er = xp.asarray(C_np), xp.asarray(et_np), xp.asarray(er_np)
    tl, ts = np.tan(np.radians(layout.fov_deg[0] / 2)), np.tan(np.radians(layout.fov_deg[1] / 2))
    rows, otas = [], []
    n = len(alt_deg)
    for s in range(0, n, chunk):
        alt, az = alt_deg[s : s + chunk], az_deg[s : s + chunk]
        zd = 90.0 - alt
        ok = zd < _ZD_MAX
        idx = xp.nonzero(ok)[0]
        if idx.size == 0:
            continue
        iz = xp.floor(zd[idx] / _GRID_STEP).astype(xp.int32)
        ia = (xp.floor((az[idx] % 360.0) / _GRID_STEP).astype(xp.int32)) % table.shape[1]
        cand = table[iz, ia]  # (m, K)
        P = _unit(alt[idx], az[idx], xp)[:, None, :]  # (m, 1, 3)
        cc = xp.clip(cand, 0, None)
        d = (P * C[cc]).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inside = (
                (cand >= 0)
                & (d > 0)
                & (xp.abs((P * et[cc]).sum(-1) / d) < tl)
                & (xp.abs((P * er[cc]).sum(-1) / d) < ts)
            )
        r, k = xp.nonzero(inside)
        rows.append(idx[r] + s)
        otas.append(cand[r, k])
    if not rows:
        return xp.zeros(0, xp.int64), xp.zeros(0, xp.int32)
    return xp.concatenate(rows), xp.concatenate(otas)


def _hadec_to_altaz(ha_deg, dec_deg, lat_deg, xp):
    h, d, p = xp.radians(ha_deg), xp.radians(dec_deg), np.radians(lat_deg)
    sa = np.sin(p) * xp.sin(d) + np.cos(p) * xp.cos(d) * xp.cos(h)
    alt = xp.arcsin(xp.clip(sa, -1.0, 1.0))
    az = xp.arctan2(-xp.cos(d) * xp.sin(h), xp.sin(d) * np.cos(p) - xp.cos(d) * np.sin(p) * xp.cos(h))
    return xp.degrees(alt), xp.degrees(az) % 360.0


def refraction_deg(alt_deg, pressure_mb: float = 790.0, temp_c: float = 10.0, xp=np):
    """Saemundsson (1986) refraction for true altitude, scaled to pressure and temperature; degrees."""
    h = xp.maximum(alt_deg, -1.0)
    r_arcmin = 1.02 / xp.tan(xp.radians(h + 10.3 / (h + 5.11)))
    return r_arcmin / 60.0 * (pressure_mb / 1010.0) * (283.0 / (273.0 + temp_c))


@dataclasses.dataclass
class Coverage:
    """Covering records: one row per (exposure, position, OTA), ordered by exposure, then position, then OTA row.

    ``ota`` indexes the layout's rows; ``tel_id``, ``subarray`` and ``band`` are looked up from it on access,
    so a full-sky night (tens of millions of records) stays at 16 bytes per record.
    """

    point: np.ndarray
    exposure: np.ndarray
    ota: np.ndarray
    layout: Layout
    times: atime.Time
    ratchet: np.ndarray
    theta_deg: np.ndarray
    backend: str

    @property
    def tel_id(self) -> np.ndarray:
        """Telescope id of each record."""
        return self.layout.tel_id[self.ota]

    @property
    def subarray(self) -> np.ndarray:
        """Subarray of each record."""
        return self.layout.subarray[self.ota]

    @property
    def band(self) -> np.ndarray:
        """Band of each record."""
        return self.layout.band[self.ota]


def covering(
    ra_deg,
    dec_deg,
    t0: atime.Time,
    t1: atime.Time,
    *,
    layout: Layout | None = None,
    schedule: Schedule | None = None,
    apparent: bool = False,
    refraction: bool = False,
    pressure_mb: float = 790.0,
    temp_c: float = 10.0,
    backend: str = "auto",
    batch_pairs: int | None = None,
) -> Coverage:
    """Which OTA, subarray and band covers each position in each exposure of the schedule between t0 and t1."""
    xp = get_backend(backend)
    layout = layout or Layout.from_config()
    schedule = schedule or Schedule()
    site = crds.EarthLocation(
        lat=c.observatory.latitude * u.deg, lon=c.observatory.longitude * u.deg, height=c.observatory.altitude * u.m
    )
    ex = schedule.exposures(t0, t1, site)
    t = ex["time"]
    ra = np.atleast_1d(np.asarray(ra_deg, float))
    dec = np.atleast_1d(np.asarray(dec_deg, float))
    n_pt, n_ex = len(ra), len(t)
    # per-exposure source positions: one set, or one per night when moving ICRS -> CIRS
    if apparent and n_ex:
        nights = np.floor(t.mjd + c.observatory.longitude / 360.0 - 0.5)
        uniq, ex_night = np.unique(nights, return_inverse=True)
        ra_set, dec_set = np.empty((len(uniq), n_pt)), np.empty((len(uniq), n_pt))
        for i, n in enumerate(uniq):
            mid = atime.Time(np.mean(t.mjd[nights == n]), format="mjd", scale="utc")
            cirs = crds.SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").transform_to(crds.CIRS(obstime=mid))
            ra_set[i], dec_set[i] = cirs.ra.deg, cirs.dec.deg
    else:
        ra_set, dec_set, ex_night = ra[None, :], dec[None, :], np.zeros(n_ex, int)
    era = t.earth_rotation_angle(longitude=c.observatory.longitude * u.deg).deg if n_ex else np.zeros(0)
    ra_x, dec_x = xp.asarray(ra_set), xp.asarray(dec_set)
    shift = xp.asarray(era + ex["theta_deg"])  # array-frame HA = ERA + lon - RA + theta; lon is inside ERA here
    night_x = xp.asarray(ex_night)
    pts, exps, otas = [], [], []
    if batch_pairs is None:  # (pairs x candidates x 3) float64 temporaries: ~1 GB per 1M pairs on the host
        batch_pairs = 1_000_000 if xp is np else 4_000_000
    per = max(1, int(batch_pairs // max(n_pt, 1)))  # exposures per batch
    for j0 in range(0, n_ex, per):
        j = xp.arange(j0, min(j0 + per, n_ex))
        ha = (shift[j][:, None] - ra_x[night_x[j]] + 180.0) % 360.0 - 180.0
        dd = dec_x[night_x[j]]
        alt, az = _hadec_to_altaz(ha.ravel(), dd.ravel(), c.observatory.latitude, xp)
        if refraction:
            alt = alt + refraction_deg(alt, pressure_mb, temp_c, xp)
        rows, o = _members(layout, alt, az, xp)
        pts.append(rows % n_pt)
        exps.append((rows // n_pt + j0).astype(xp.int32))
        otas.append(o)
    to_np = (lambda a: a.get()) if xp is not np else (lambda a: a)
    if pts:
        point = to_np(xp.concatenate(pts)).astype(np.int32)
        exposure = to_np(xp.concatenate(exps)).astype(np.int32)
        ota = to_np(xp.concatenate(otas))
    else:
        point = exposure = ota = np.zeros(0, np.int32)
    # candidates come out in (exposure, position, table) order on both backends; OTA rows are sorted per cell
    return Coverage(
        point=point,
        exposure=exposure,
        ota=ota.astype(np.int16),
        layout=layout,
        times=t,
        ratchet=ex["ratchet"],
        theta_deg=ex["theta_deg"],
        backend=xp.__name__,
    )


def coverage_intervals(ra_deg, dec_deg, t0: atime.Time, t1: atime.Time, **kwargs) -> list[dict]:
    """Per position, OTA and band: contiguous intervals of coverage, [t_in, t_out] in MJD (exposure mid-times).

    Consecutive exposures of one OTA merge into one interval; a gap of more than one exposure (a reset, the
    position leaving the OTA, daylight) starts a new one.  This is the form a coverage tool reports.
    """
    cov = covering(ra_deg, dec_deg, t0, t1, **kwargs)
    mjd = cov.times.mjd if len(cov.times) else np.zeros(0)
    step = (kwargs.get("schedule") or Schedule()).exptime_s / 86400.0
    out = []
    if len(cov.point) == 0:
        return out
    key = np.lexsort((cov.exposure, cov.tel_id, cov.point))
    p, tid, e, b = cov.point[key], cov.tel_id[key], cov.exposure[key], cov.band[key]
    tm = mjd[e]
    brk = np.r_[True, (p[1:] != p[:-1]) | (tid[1:] != tid[:-1]) | (tm[1:] - tm[:-1] > 1.5 * step)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(p)] - 1
    sub = dict(zip(cov.tel_id, cov.subarray))
    for s, e_ in zip(starts, ends):
        out.append(
            dict(
                point=int(p[s]),
                tel_id=int(tid[s]),
                subarray=int(sub[tid[s]]),
                band=str(b[s]),
                t_in=float(tm[s]),
                t_out=float(tm[e_]),
                n_exposures=int(e_ - s + 1),
            )
        )
    out.sort(key=lambda r: (r["point"], r["t_in"], r["tel_id"]))
    return out
