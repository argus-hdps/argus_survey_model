"""Tile depth maps: per tile, band and half-year, the stacked depth and the counts of slots and nights.

The build sums, per minipix of a tile, the survey's weight n x T^2 / sigma_bkg^2, the exposure count and the
exposure time, as ``MinipixAccumulator`` does, from the full-precision pair records. A partial pair adds to one run
of columns per grid row (``geometry.row_runs``), so the sums use a difference array along each row. The tile's
depth is the median, over its owned minipixes that the period covered, of the per-minipix 5-sigma depth.
"""

from __future__ import annotations

import datetime

import numpy as np

from .geometry import GRID

NSIGMA = 5.0
DEPTH_COLUMNS = (
    "band",
    "period",
    "tanseg_id",
    "limmag",
    "covered_fraction",
    "n_nights",
    "n_slots_60",
    "n_slots_1",
    "n_frames",
)


def half_year(night: int) -> str:
    """Return the half-year label ("2028H1": January to June) of the morning that ends night ``night``.

    Night labels are the MJD of the local noon that starts the night.
    """
    d = datetime.date(1858, 11, 17) + datetime.timedelta(days=int(night) + 1)
    return f"{d.year}H{1 if d.month <= 6 else 2}"


class TileDepth:
    """Per-minipix sums of one tile for every (band, period), filled pair by pair."""

    def __init__(self, n_bands: int, n_periods: int):
        """Allocate the sums for ``n_bands`` x ``n_periods``."""
        self.nb, self.np_ = n_bands, n_periods
        self.nbp = n_bands * n_periods
        # ivar, base exposures, fast frames, exposure seconds
        self.full = np.zeros((4, self.nbp))
        self.diff = np.zeros((4, self.nbp * GRID * (GRID + 1)))
        self.rows = []

    def add(self, bp, weights, lo=None, hi=None) -> None:
        """Add pairs with (band, period) index ``bp`` and ``weights`` (4, n).

        Full pairs have ``lo``/``hi`` None; partial pairs give their row runs (n, 64).
        """
        if len(bp) == 0:
            return
        if lo is None:
            for k in range(4):
                self.full[k] += np.bincount(bp, weights[k], minlength=self.nbp)
            return
        ok = hi >= lo
        p, j = np.nonzero(ok)
        base = (bp[p] * GRID + j) * (GRID + 1)
        start, end = base + lo[p, j], base + hi[p, j] + 1
        n = self.diff.shape[1]
        for k in range(4):
            w = weights[k][p]
            self.diff[k] += np.bincount(start, w, minlength=n) - np.bincount(end, w, minlength=n)

    def sums(self) -> np.ndarray:
        """Return the per-minipix sums (4, n_bands, n_periods, 64, 64)."""
        d = self.diff.reshape(4, self.nbp, GRID, GRID + 1).cumsum(-1)[..., :GRID]
        d += self.full[:, :, None, None]
        return d.reshape(4, self.nb, self.np_, GRID, GRID)


def tile_rows(
    tanseg_id: int,
    sums: np.ndarray,
    owned: np.ndarray,
    counts: dict,
    zeropoints: list[tuple[float, float]],
) -> list[tuple]:
    """Return the depth-map rows of one tile, one per (band, period) with records plus one per band for "all".

    ``sums`` from ``TileDepth.sums``; ``owned`` (64, 64) bool; ``counts[(b, p)]`` = (n_nights, n_slots_60,
    n_slots_1, n_frames) for the pairs with records; ``zeropoints[b]`` = (zp_photons_per_sec, band_qe). Period
    index ``n_periods`` stands for "all".
    """
    iv, n60, n1, esec = sums
    nb, nper = iv.shape[:2]
    n_own = max(int(owned.sum()), 1)
    out = []
    for b in range(nb):
        zp, qe = zeropoints[b]
        present = [p for p in range(nper) if (b, p) in counts]
        if not present:
            continue
        for p in [*present, nper]:
            if p == nper:
                v, a60, a1, es = (x[b].sum(0) for x in (iv, n60, n1, esec))
                cnt = tuple(sum(counts[(b, q)][k] for q in present) for k in range(4))
            else:
                v, a60, a1, es = iv[b, p], n60[b, p], n1[b, p], esec[b, p]
                cnt = counts[(b, p)]
            ok = owned & (v > 0)
            lim = np.nan
            if ok.any():
                mean_t = es[ok] / (a60[ok] + a1[ok])
                rate = NSIGMA / np.sqrt(v[ok]) / mean_t / qe
                lim = float(np.median(-2.5 * np.log10(rate / zp)))
            out.append((b, p, tanseg_id, lim, ok.sum() / n_own, *cnt))
    return out
