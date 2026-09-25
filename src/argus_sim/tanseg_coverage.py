"""Exact OTA coverage of the HDPS TAN-segment / minipix grid, per ratchet, with streaming per-minipix sums.

Geometry: the gnomonic OTA test of coverage_exact applied to the tiles and minipixes of the HDPS all-sky
tessellation (skymap_shim).  Tiles are ICRS; each ratchet maps them into the array frame with one rotation:
ICRS -> CIRS (IAU 2006/2000A bias-precession-nutation, erfa.c2i06a; aberration not applied), then hour angle
ERA + lon - RA + theta (rigid rotation about the true celestial pole, all subarrays in phase), then the site
latitude.  Tracking runs at the sidereal rate, so the whole ratchet has one
geometry.

Per ratchet and (tile, OTA) pair the outcome is FULL (all four tile corners inside the OTA: every minipix is
covered, since both shapes are convex great-circle polygons) or PARTIAL (a 4096-bit minipix-centre mask).  A
pair with no tile corner inside the OTA is PARTIAL only if an OTA corner falls inside the tile; otherwise the
two cannot intersect (the OTA is wider than a tile on both axes, so an OTA edge crossing a tile with no corner
of either inside the other would need two parallel OTA edges inside one tile).

Per-minipix sums, per band: full-system weight n x T^2 / sigma_bkg^2 (float32), exposure count (uint16).  They
live on the device for the tiles reachable from the site, one night at a time.
"""

from __future__ import annotations

import dataclasses

import astropy.time as atime
import astropy.units as u
import erfa
import numpy as np

from argus_sim import c, get_logger

from . import skymap_shim as sh
from .coverage_exact import _ZD_MAX, Layout, _candidate_table, _hadec_to_altaz, _unit, get_backend

log = get_logger(__name__)

TILE_HALF_DIAG_DEG = float(np.degrees(np.arctan(np.sqrt(2) * np.tan(np.radians(sh.TILE_SIZE_DEG / 2)))))


def icrs_to_array_rotation(t: atime.Time, theta_deg: float) -> np.ndarray:
    """3x3 rotation taking an ICRS unit vector to the array frame (north, east, up) at time t and tracking theta."""
    lat, lon = c.observatory.latitude, c.observatory.longitude
    m = erfa.c2i06a(t.tt.jd1, t.tt.jd2)  # GCRS/ICRS -> CIRS, bias-precession-nutation
    era_loc = t.earth_rotation_angle(longitude=lon * u.deg).deg
    ra = np.array([0.0, 90.0, 0.0])
    dec = np.array([0.0, 0.0, 90.0])
    alt, az = _hadec_to_altaz(era_loc + theta_deg - ra, dec, lat, np)
    r_cirs_to_neu = _unit(alt, az, np).T  # columns: images of the CIRS basis vectors
    return r_cirs_to_neu @ m


@dataclasses.dataclass
class RatchetPairs:
    """One ratchet's covering (tile row, OTA row) pairs."""

    full_row: object
    full_ota: object
    part_row: object
    part_ota: object
    part_mask: object  # bool (n_part, 4096)
    n_tiles_in_view: int


class TansegCoverage:
    """Tile/minipix coverage engine for one layout on one backend."""

    def __init__(
        self,
        layout: Layout | None = None,
        backend: str = "auto",
        dec_margin_deg: float = 1.0,
        dec_range: tuple[float, float] | None = None,
    ):
        """Set up the engine for every tile the footprint can reach (or the tiles in ``dec_range``, degrees)."""
        self.xp = xp = get_backend(backend)
        self.layout = layout or Layout.from_config()
        lat = c.observatory.latitude
        reach = _ZD_MAX
        if dec_range is None:  # every tile the footprint can reach
            dec_range = (max(lat - reach - dec_margin_deg, -90.0), min(lat + reach + dec_margin_deg, 90.0))
        tl = sh.tiles(*dec_range)
        self.tanseg_id = tl["tanseg_id"]  # increasing
        self.n_tiles = len(self.tanseg_id)
        T, E, N = sh.tangent_basis(tl["ra_center"], tl["dec_center"])
        self.T, self.E, self.N = xp.asarray(T), xp.asarray(E), xp.asarray(N)
        xi, eta = sh.minipix_offsets()
        self.mp_xi, self.mp_eta = xp.asarray(xi), xp.asarray(eta)
        cx, cy = sh.tile_corner_offsets()
        self.cn_xi, self.cn_eta = xp.asarray(cx), xp.asarray(cy)
        self.tile_half_tan = float(np.tan(np.radians(sh.SEGMENT_PIXEL_SIZE / 2 * sh.PIXEL_SCALE_DEG)))
        C, et, er = self.layout.basis()
        self.C, self.et, self.er = xp.asarray(C), xp.asarray(et), xp.asarray(er)
        self.tl = float(np.tan(np.radians(self.layout.fov_deg[0] / 2)))
        self.ts = float(np.tan(np.radians(self.layout.fov_deg[1] / 2)))
        corners = []
        for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            v = C + s1 * self.tl * et + s2 * self.ts * er
            corners.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        self.ota_corners = xp.asarray(np.stack(corners, axis=1))  # (n_ota, 4, 3)
        self.bands = [str(b) for b in sorted(set(self.layout.band))]
        self.ota_band = xp.asarray(np.array([self.bands.index(b) for b in self.layout.band], np.int8))
        key = (tuple(self.layout.alt_deg), tuple(self.layout.az_deg), self.layout.fov_deg, TILE_HALF_DIAG_DEG)
        table, self.K = _candidate_table(key)
        self.table = xp.asarray(table)

    # ------------------------------------------------------------------ geometry
    def _inside_ota(self, P, ota):
        """P (..., 3) in the array frame, ota (...) OTA rows broadcastable to P[..., 0]: inside the OTA's field."""
        xp = self.xp
        d = (P * self.C[ota]).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (d > 0)
                & (xp.abs((P * self.et[ota]).sum(-1) / d) < self.tl)
                & (xp.abs((P * self.er[ota]).sum(-1) / d) < self.ts)
            )

    def pairs(self, R, chunk_pairs: int = 1500) -> RatchetPairs:
        """Covering pairs for one ratchet whose ICRS -> array rotation is R (3x3)."""
        xp = self.xp
        Rx = xp.asarray(R)
        Tn = self.T @ Rx.T
        zd = xp.degrees(xp.arccos(xp.clip(Tn[:, 2], -1, 1)))
        rows = xp.nonzero(zd < _ZD_MAX)[0]
        az = xp.degrees(xp.arctan2(Tn[rows, 1], Tn[rows, 0])) % 360.0
        iz = xp.floor(zd[rows] / 0.25).astype(xp.int32)
        ia = xp.floor(az / 0.25).astype(xp.int32) % self.table.shape[1]
        cand = self.table[iz, ia]  # (m, K)
        En, Nn = self.E[rows] @ Rx.T, self.N[rows] @ Rx.T
        Tr = Tn[rows]
        # tile corners in the array frame: (m, 4, 3)
        Pc = Tr[:, None, :] + self.cn_xi[None, :, None] * En[:, None, :] + self.cn_eta[None, :, None] * Nn[:, None, :]
        Pc /= xp.linalg.norm(Pc, axis=-1, keepdims=True)
        valid = cand >= 0
        cc = xp.clip(cand, 0, None)
        n_in = self._inside_ota(Pc[:, None, :, :], cc[:, :, None]).sum(-1)  # (m, K)
        full = valid & (n_in == 4)
        # an OTA corner inside the tile: tile-frame standard coordinates of the OTA's four corners
        V = self.ota_corners[cc]  # (m, K, 4, 3)
        d = (V * Tr[:, None, None, :]).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (V * En[:, None, None, :]).sum(-1) / d
            et = (V * Nn[:, None, None, :]).sum(-1) / d
        corner_in_tile = ((d > 0) & (xp.abs(xi) < self.tile_half_tan) & (xp.abs(et) < self.tile_half_tan)).any(-1)
        part = valid & ~full & ((n_in > 0) | corner_in_tile)
        fr, fk = xp.nonzero(full)
        pr, pk = xp.nonzero(part)
        part_row, part_ota = rows[pr], cand[pr, pk]
        masks = []
        for s in range(0, len(part_row), chunk_pairs):
            r_, o_ = part_row[s : s + chunk_pairs], part_ota[s : s + chunk_pairs]
            Tt, Ee, Nnn = Tn[r_], self.E[r_] @ Rx.T, self.N[r_] @ Rx.T
            P = (
                Tt[:, None, :]
                + self.mp_xi[None, :, None] * Ee[:, None, :]
                + self.mp_eta[None, :, None] * Nnn[:, None, :]
            )
            P /= xp.linalg.norm(P, axis=-1, keepdims=True)
            masks.append(self._inside_ota(P, o_[:, None]))
        mask = xp.concatenate(masks) if masks else xp.zeros((0, sh.N_MINIPIX), bool)
        keep = mask.any(1)
        return RatchetPairs(
            full_row=rows[fr],
            full_ota=cand[fr, fk],
            part_row=part_row[keep],
            part_ota=part_ota[keep],
            part_mask=mask[keep],
            n_tiles_in_view=int(len(rows)),
        )

    def pair_field_angles(self, p: RatchetPairs, R) -> tuple[object, object]:
        """Field angle (deg from the OTA's optical axis) of each covering pair.

        The point is the tile centre for FULL pairs and the centroid of the covered minipixes for PARTIAL pairs.
        R is the ratchet's ICRS -> array rotation.
        """
        xp = self.xp
        Rx = xp.asarray(R)

        # The covered centroid of a partial pair is mask @ (xi, eta) / mask.sum(): a matrix-vector product on
        # the bool mask, in row chunks.  Materialising the mask as float64 (pairs x 4096) and multiplying it by
        # the minipix offsets would cost three temporaries of ~1 GB at a heavy-overlap ratchet, enough to push a
        # 5-year run over the CuPy pool cap on a 16 GB GPU.
        xi_eta = xp.stack([self.mp_xi, self.mp_eta], axis=1).astype(xp.float32)  # (4096, 2)

        def centroid(mask, chunk=4096):
            out = xp.empty((len(mask), 2), xp.float64)
            for s in range(0, len(mask), chunk):
                m = mask[s : s + chunk].astype(xp.float32)
                nw = xp.maximum(m.sum(1), 1.0)
                out[s : s + chunk] = (m @ xi_eta) / nw[:, None]
            return out[:, 0], out[:, 1]

        def angle(rows, ota, mask=None):
            if len(rows) == 0:
                return xp.zeros(0, xp.float64)
            P = self.T[rows] @ Rx.T
            if mask is not None:
                mx, my = centroid(mask)
                P = P + mx[:, None] * (self.E[rows] @ Rx.T) + my[:, None] * (self.N[rows] @ Rx.T)
                P = P / xp.linalg.norm(P, axis=-1, keepdims=True)
            return xp.degrees(xp.arccos(xp.clip((P * self.C[ota]).sum(-1), -1.0, 1.0)))

        return angle(p.full_row, p.full_ota), angle(p.part_row, p.part_ota, p.part_mask)


# ---------------------------------------------------------------------- accumulation
MINIPIX_AREA_DEG2 = (sh.MINIPIX_PIXEL_SIZE * sh.PIXEL_SCALE_DEG) ** 2


def owned_mask(tanseg_id: np.ndarray, xp=np, chunk: int = 2000) -> object:
    """(n_tiles, 4096) bool: minipixes whose centre the forward lookup assigns to their own tile.

    Tiles overlap by 32 px; counting owned minipixes only gives each sky position once (area, medians).
    """
    xi, eta = sh.minipix_offsets(xp)
    ra_c, dec_c = sh.tile_center(xp.asarray(tanseg_id), xp)
    out = []
    for s in range(0, len(tanseg_id), chunk):
        T, E, N = sh.tangent_basis(ra_c[s : s + chunk], dec_c[s : s + chunk], xp)
        P = T[:, None, :] + xi[None, :, None] * E[:, None, :] + eta[None, :, None] * N[:, None, :]
        P /= xp.linalg.norm(P, axis=-1, keepdims=True)
        ra = xp.degrees(xp.arctan2(P[..., 1], P[..., 0])) % 360.0
        dec = xp.degrees(xp.arcsin(xp.clip(P[..., 2], -1, 1)))
        tid, loc = sh.radec_to_composite_id(ra.ravel(), dec.ravel(), xp)
        want = xp.asarray(tanseg_id[s : s + chunk])[:, None]
        out.append((tid.reshape(P.shape[:2]) == want) & (loc.reshape(P.shape[:2]) == xp.arange(sh.N_MINIPIX)[None, :]))
    return xp.concatenate(out)


FAST_UNIT = 60  # fast-cadence exposures are counted in units of 60 (a fast ratchet has n_dark x 60 of them)


class MinipixAccumulator:
    """Per-minipix, per-band sums for the current night on the device; run totals in host memmaps.

    ``ivar`` (float32) is the summed full-system weight, n x T^2 / sigma_bkg^2 with T the signal throughput
    (optics x atmosphere at the exposure's airmass); ``nobs`` (uint16) counts base-cadence exposures.  Fast
    (bright-time, 1-s) exposures are counted in ``fast_units`` of ``FAST_UNIT`` exposures: a fast ratchet is
    wholly fast and has n_dark x 60 exposures, so the unit is exact, keeps the night's count inside uint16
    (a winter night can exceed 65,535 one-second looks per minipix), and the array is only allocated once
    bright time occurs.  Contributions are scattered over row chunks into uint32 / float32 temporaries (CuPy
    has no uint16 scatter-add), which bounds the temporaries' size.
    """

    def __init__(self, engine: TansegCoverage, outdir: str | None, save_nightly: bool = False, chunk_rows: int = 4096):
        """Allocate the night's device sums and the run totals under ``outdir`` for ``engine``'s tiles."""
        self.eng, self.xp = engine, engine.xp
        self.nb = len(engine.bands)
        xp = self.xp
        n = engine.n_tiles
        self.ivar = xp.zeros((self.nb, n, sh.N_MINIPIX), xp.float32)
        self.nobs = xp.zeros((self.nb, n, sh.N_MINIPIX), xp.uint16)
        self.fast_units = None
        self.touched = xp.zeros(n, bool)
        self.outdir, self.save_nightly, self.chunk_rows = outdir, save_nightly, chunk_rows
        self.totals = None
        self.night_summaries = []
        if outdir is not None:
            import os

            d = os.path.join(outdir, "minipix")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "tanseg_id.npy"), engine.tanseg_id)
            self.totals = {
                "ivar": np.lib.format.open_memmap(
                    os.path.join(d, "total_ivar.npy"), "w+", np.float32, (self.nb, n, sh.N_MINIPIX)
                ),
                "nobs": np.lib.format.open_memmap(
                    os.path.join(d, "total_nobs.npy"), "w+", np.uint32, (self.nb, n, sh.N_MINIPIX)
                ),
                "fast_units": None,
            }
            self.mdir = d

    def add(self, p: RatchetPairs, weight: dict[int, object], n_epochs: int, fast: bool = False) -> None:
        """Add one ratchet.

        ``weight[b]`` is band b's per-exposure weight T^2 / sigma_bkg^2: either one value per tile row (length
        n_tiles), or a tuple (full, partial) with one value per covering pair of band b, in the order of
        ``p.full_row[band == b]`` and ``p.part_row[band == b]`` (per-pair PSF).
        """
        xp, eng = self.xp, self.eng
        if fast:
            if n_epochs % FAST_UNIT:
                raise ValueError(f"fast ratchet with {n_epochs} exposures, not a multiple of {FAST_UNIT}")
            if self.fast_units is None:
                self.fast_units = xp.zeros_like(self.nobs)
            target, count = self.fast_units, n_epochs // FAST_UNIT
        else:
            target, count = self.nobs, n_epochs
        fb, pb = eng.ota_band[p.full_ota], eng.ota_band[p.part_ota]
        for b in range(self.nb):
            fr_all, pr_all, pm_all = p.full_row[fb == b], p.part_row[pb == b], p.part_mask[pb == b]
            if len(fr_all) == 0 and len(pr_all) == 0:
                continue
            rows_all = xp.unique(xp.concatenate([fr_all, pr_all]))
            w = weight[b]
            if isinstance(w, tuple):
                wf_all, wp_all = xp.asarray(w[0]), xp.asarray(w[1])
            else:
                wf_all, wp_all = w[fr_all], w[pr_all]
            for c0 in range(0, len(rows_all), self.chunk_rows):
                rows = rows_all[c0 : c0 + self.chunk_rows]
                lo, hi = rows[0], rows[-1]
                fsel = (fr_all >= lo) & (fr_all <= hi)
                psel = (pr_all >= lo) & (pr_all <= hi)
                fr, pr, pm = fr_all[fsel], pr_all[psel], pm_all[psel]
                wf, wp = wf_all[fsel], wp_all[psel]
                cnt = xp.zeros((len(rows), sh.N_MINIPIX), xp.uint32)
                iv = xp.zeros((len(rows), sh.N_MINIPIX), xp.float32)
                fi, pi = xp.searchsorted(rows, fr), xp.searchsorted(rows, pr)
                if len(fr):
                    xp.add.at(cnt, fi, xp.uint32(count))
                    xp.add.at(
                        iv, fi, xp.broadcast_to((wf * n_epochs).astype(xp.float32)[:, None], (len(fr), sh.N_MINIPIX))
                    )
                if len(pr):
                    xp.add.at(cnt, pi, pm.astype(xp.uint32) * count)
                    xp.add.at(iv, pi, pm.astype(xp.float32) * (wp * n_epochs).astype(xp.float32)[:, None])
                target[b, rows] += cnt.astype(xp.uint16)
                self.ivar[b, rows] += iv
                self.touched[rows] = True

    def end_night(self, night: int, owned=None) -> dict:
        """Flush the night: nightly file (optional), fold into the host totals, reset. Returns a summary."""
        xp = self.xp
        rows = xp.nonzero(self.touched)[0]
        rows_h = rows.get() if xp is not np else rows
        summary = {"night": int(night), "tiles": int(len(rows_h))}
        for b, band in enumerate(self.eng.bands):
            seen_n = 0
            owned_n = 0
            for c0 in range(0, len(rows), self.chunk_rows):
                r = rows[c0 : c0 + self.chunk_rows]
                n = self.nobs[b, r] > 0
                if self.fast_units is not None:
                    n = n | (self.fast_units[b, r] > 0)
                seen_n += int(n.sum())
                if owned is not None:
                    owned_n += int((n & owned[r]).sum())
            if owned is not None:
                summary[f"{band}_owned_minipix"] = owned_n
                summary[f"{band}_area_deg2"] = float(owned_n) * MINIPIX_AREA_DEG2
            summary[f"{band}_minipix"] = seen_n
        if self.totals is not None:
            import os

            if self.fast_units is not None and self.totals["fast_units"] is None:
                self.totals["fast_units"] = np.lib.format.open_memmap(
                    os.path.join(self.mdir, "total_fast_units.npy"), "w+", np.uint32, self.totals["nobs"].shape
                )
            # touched rows run in long contiguous stretches (tanseg ids follow dec bands): flush slices, not
            # fancy indices, so the memmap sees sequential I/O
            breaks = np.flatnonzero(np.diff(rows_h) != 1) + 1
            for run in np.split(rows_h, breaks):
                for a in range(0, len(run), 4096):
                    lo, hi = int(run[a]), int(run[min(a + 4096, len(run)) - 1]) + 1
                    for key, arr in (("ivar", self.ivar), ("nobs", self.nobs), ("fast_units", self.fast_units)):
                        if arr is None:
                            continue
                        blk = arr[:, lo:hi]
                        self.totals[key][:, lo:hi] += blk.get() if xp is not np else blk
            if self.save_nightly:
                sel = lambda a: a[:, rows].get() if xp is not np else a[:, rows]  # noqa: E731
                np.savez_compressed(
                    os.path.join(self.mdir, f"night_{int(night):04d}.npz"),
                    rows=rows_h,
                    tanseg_id=self.eng.tanseg_id[rows_h],
                    ivar=sel(self.ivar),
                    nobs=sel(self.nobs),
                    fast_unit=np.int32(FAST_UNIT),
                    **({"fast_units": sel(self.fast_units)} if self.fast_units is not None else {}),
                )
        self.night_summaries.append(summary)
        self.ivar[:] = 0
        self.nobs[:] = 0
        if self.fast_units is not None:
            self.fast_units[:] = 0
        self.touched[:] = False
        return summary


def write_epoch_record(
    path: str, eng: TansegCoverage, p: RatchetPairs, noise: dict[int, object], signal_tp: dict[int, object], meta: dict
) -> None:
    """Write one ratchet's per-epoch record.

    Per pair: tanseg, OTA, band, covered-minipix count, background noise and signal throughput; packed minipix
    masks for PARTIAL pairs only.  ``noise`` and ``signal_tp`` are the per-exposure background noise
    sqrt(B / sharpness) and the full-system signal throughput, either per tile row (dicts by band, length n_tiles)
    or per pair (tuples (full, partial) in the order of p's pairs).
    """
    h = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)  # noqa: E731
    rows = np.concatenate([h(p.full_row), h(p.part_row)])
    ota = np.concatenate([h(p.full_ota), h(p.part_ota)]).astype(np.int16)
    band = h(eng.ota_band)[ota]
    n_full = len(p.full_row)
    n_cov = np.concatenate([np.full(n_full, sh.N_MINIPIX), h(p.part_mask.sum(1))]).astype(np.int16)
    if isinstance(noise, tuple):
        nz = np.concatenate([h(noise[0]), h(noise[1])]).astype(np.float32)
        tp = np.concatenate([h(signal_tp[0]), h(signal_tp[1])]).astype(np.float32)
    else:
        nz = np.full(len(rows), np.nan, np.float32)
        tp = np.full(len(rows), np.nan, np.float32)
        for b, v in noise.items():
            sel = band == b
            nz[sel] = h(v)[rows[sel]]
            tp[sel] = h(signal_tp[b])[rows[sel]]
    np.savez_compressed(
        path,
        tanseg_id=eng.tanseg_id[rows],
        tel_id=eng.layout.tel_id[ota].astype(np.int16),
        band=band.astype(np.int8),
        n_covered=n_cov,
        noise=nz,
        signal_tp=tp,
        partial_from=np.int32(n_full),
        partial_mask=np.packbits(h(p.part_mask), axis=1),
        bands=np.array(eng.bands),
        **{k: np.asarray(v) for k, v in meta.items()},
    )


def minipix_limmag(
    ivar, nobs, fast_units, ab, band: str, band_qe: float, snr: float, exptime_s: float, fast_exptime_s: float
):
    """Full-system limiting magnitude per minipix from the sums.

    snr / sqrt(sum n T^2/sigma^2) is the above-atmosphere flux (electron-equivalent) at the threshold; it is divided
    by the exposure time per observation and the QE and converted against the AB zero point (as
    cadence_accumulator.limmag_from_map, with the throughput in the weights).
    """
    from .cadence_accumulator import _qe_as_float

    n1 = fast_units.astype(np.float64) * FAST_UNIT if fast_units is not None else 0.0
    n = nobs.astype(np.float64) + n1
    exp = nobs * exptime_s + n1 * fast_exptime_s
    out = np.full(ivar.shape, np.nan)
    ok = (ivar > 0) & (n > 0)
    signal_e = snr / np.sqrt(ivar[ok].astype(np.float64))
    rate = signal_e / (exp[ok] / n[ok]) / _qe_as_float(band_qe)
    out[ok] = _mag(ab, band, rate)
    return out


def _mag(ab, band, rate):
    m = ab.mag_from_photons(band, rate * (u.photon / u.second))
    return np.asarray(getattr(m, "value", m), float)
