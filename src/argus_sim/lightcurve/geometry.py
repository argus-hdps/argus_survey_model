"""Pair geometry recomputed from the frozen OTA frames and ratchet rotations.

The survey records, per (tile, OTA) pair, the minipixes the OTA covers (partial pairs), their number, the pair's
field angle and the tile's airmass. All four follow from the ratchet's ICRS-to-array rotation and the OTA's frame,
so the visit store keeps the rotations and recomputes them.

A tile's minipix centres lie on a regular 64 x 64 grid of standard coordinates (xi, eta) on the tile's tangent
plane, and each OTA field edge is a great circle, which is a straight line on that plane. The covered minipixes of
one grid row are therefore one run of consecutive columns. ``row_runs`` finds that run per row from five linear
inequalities, which costs 64 rows per pair instead of 4096 minipixes. Grid points where an inequality is within
rounding of zero are tested one by one with the survey's own test (``TansegCoverage.pairs``).
"""

from __future__ import annotations

import astropy.time as atime
import numpy as np

from .. import skymap_shim as sh

GRID = sh.MINIPIX_GRID_SIZE
EDGE_EPS = 1e-12  # in units of the forms, which are dot products of unit vectors


class Geometry:
    """The OTA frames of the run's layout, frozen at build time, with the coverage test of TansegCoverage."""

    def __init__(self, tel_id, subarray, C, et, er, tl: float, ts: float):
        """Store per-OTA tel_id, subarray, axis C and field axes et, er; tl and ts are tangents of the half fields."""
        self.tel_id = np.asarray(tel_id, np.int64)
        self.subarray = np.asarray(subarray, np.int64)
        self.C, self.et, self.er = (np.asarray(a, np.float64) for a in (C, et, er))
        self.tl, self.ts = float(tl), float(ts)
        self.ota_of_tel = np.full(int(self.tel_id.max()) + 1, -1, np.int64)
        self.ota_of_tel[self.tel_id] = np.arange(len(self.tel_id))

    @classmethod
    def from_layout(cls, name: str) -> Geometry:
        """Freeze a registered ring layout (``coverage_exact.Layout.from_config``)."""
        from ..coverage_exact import Layout

        lay = Layout.from_config(name)
        C, et, er = lay.basis()
        tl = np.tan(np.radians(lay.fov_deg[0] / 2))
        ts = np.tan(np.radians(lay.fov_deg[1] / 2))
        return cls(lay.tel_id, lay.subarray, C, et, er, tl, ts)

    def arrays(self) -> dict:
        """Return the constructor arguments, for saving and for worker processes."""
        return dict(
            tel_id=self.tel_id, subarray=self.subarray, C=self.C, et=self.et, er=self.er, tl=self.tl, ts=self.ts
        )

    def ota(self, tel_id) -> np.ndarray:
        """Return the layout row of each tel_id."""
        o = self.ota_of_tel[np.asarray(tel_id, np.int64)]
        if np.any(o < 0):
            raise ValueError(f"tel_id {np.unique(np.asarray(tel_id)[o < 0])[:10]} not in the layout.")
        return o

    def inside(self, P, ota) -> np.ndarray:
        """Return whether P (..., 3, array frame) is inside the field of OTA row ``ota`` (broadcast to P[..., 0])."""
        d = (P * self.C[ota]).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                (d > 0)
                & (np.abs((P * self.et[ota]).sum(-1) / d) < self.tl)
                & (np.abs((P * self.er[ota]).sum(-1) / d) < self.ts)
            )


def ratchet_rotation(mjd: float, theta_deg: float) -> np.ndarray:
    """Return the ICRS-to-array rotation at ``mjd`` (UTC) with tracking angle ``theta_deg``, as the survey does."""
    from ..tanseg_coverage import icrs_to_array_rotation

    return icrs_to_array_rotation(atime.Time(mjd, format="mjd", scale="utc"), float(theta_deg))


def tile_frames(tanseg_id) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the tile centres T and the local east E and north N unit vectors, each (n, 3)."""
    return sh.tangent_basis(*sh.tile_center(np.atleast_1d(np.asarray(tanseg_id))))


def _grid_axes() -> tuple[float, float, float, float]:
    xi, eta = sh.minipix_offsets()
    return float(xi[0]), float(xi[1] - xi[0]), float(eta[0]), float(eta[GRID] - eta[0])


XI0, DXI, ETA0, DETA = _grid_axes()
ETA_ROWS = ETA0 + DETA * np.arange(GRID)


def rotate(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply rotations R (n, 3, 3) to vectors v (n, 3) or (3,), as the survey's ``v @ R.T``."""
    if v.ndim == 1:
        return np.einsum("nij,j->ni", R, v)
    return np.einsum("nij,nj->ni", R, v)


def row_runs(
    geom: Geometry, R: np.ndarray, T: np.ndarray, E: np.ndarray, N: np.ndarray, ota: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the covered column run [lo, hi] of each minipix grid row, for n pairs.

    R (n, 3, 3) is each pair's ratchet rotation, T, E, N (n, 3) its tile frame (ICRS), ota (n,) its layout row.
    Returns lo, hi (n, 64) int16, with hi < lo for a row the OTA does not reach. Row j holds the minipixes
    j * 64 + i, i = lo..hi.
    """
    n = len(ota)
    Tn, En, Nn = rotate(R, T), rotate(R, E), rotate(R, N)
    C, et, er = geom.C[ota], geom.et[ota], geom.er[ota]

    def form(V):
        return (Tn * V).sum(-1), (En * V).sum(-1), (Nn * V).sum(-1)

    dc, uc, vc = form(C), form(et), form(er)
    # inside when every form a + b xi + c eta < 0: -d, u - tl d, -u - tl d, v - ts d, -v - ts d
    forms = [
        tuple(-x for x in dc),
        tuple(u - geom.tl * d for u, d in zip(uc, dc)),
        tuple(-u - geom.tl * d for u, d in zip(uc, dc)),
        tuple(v - geom.ts * d for v, d in zip(vc, dc)),
        tuple(-v - geom.ts * d for v, d in zip(vc, dc)),
    ]
    lo = np.zeros((n, GRID), np.int16)
    hi = np.full((n, GRID), GRID - 1, np.int16)
    edge_pts = []
    xs = np.array([XI0, XI0 + (GRID - 1) * DXI])
    es = np.array([ETA0, ETA0 + (GRID - 1) * DETA])
    for a, b, c in forms:
        # The form is linear, so its values at the grid's four corners bound it over the grid. Most edges of the
        # OTA field miss a tile entirely; only the one or two that cross it need the per-row runs.
        corner = a[:, None, None] + b[:, None, None] * xs[None, :, None] + c[:, None, None] * es[None, None, :]
        corner = corner.reshape(n, 4)
        hi[corner.min(1) > EDGE_EPS] = -1
        act = np.flatnonzero((corner.max(1) >= -EDGE_EPS) & (corner.min(1) <= EDGE_EPS))
        if len(act) == 0:
            continue
        a, b, c = a[act], b[act], c[act]
        # along row j the form is A_j + B * i, with i the column index
        A = (a + b * XI0)[:, None] + c[:, None] * ETA_ROWS[None, :]
        B = b * DXI
        flat = B == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.clip(-A / np.where(flat, 1.0, B)[:, None], -2.0, GRID + 2.0)
        pos, neg = np.flatnonzero(B > 0), np.flatnonzero(B < 0)
        hi[act[pos]] = np.minimum(hi[act[pos]], (np.ceil(x[pos]) - 1).astype(np.int16))
        lo[act[neg]] = np.maximum(lo[act[neg]], (np.floor(x[neg]) + 1).astype(np.int16))
        if flat.any():
            f = np.flatnonzero(flat)
            hi[act[f]] = np.where(A[f] >= 0, -1, hi[act[f]])
        # grid points whose form value is within rounding of zero: the survey's own test decides them
        k = np.rint(x)
        near = (np.abs(x - k) * np.abs(B)[:, None] < EDGE_EPS) & (k >= 0) & (k < GRID) & ~flat[:, None]
        if near.any():
            p, j = np.nonzero(near)
            edge_pts.append((act[p], j, k[p, j].astype(np.int64)))
    # lo only rises from 0 and hi only falls from 63, so every non-empty run lies inside the grid
    if edge_pts:
        _settle_edges(geom, Tn, En, Nn, ota, lo, hi, edge_pts)
    return lo, hi


def _settle_edges(geom, Tn, En, Nn, ota, lo, hi, edge_pts) -> None:
    """Decide grid points that lie on an OTA edge to rounding with the survey's own point test."""
    p = np.concatenate([e[0] for e in edge_pts])
    j = np.concatenate([e[1] for e in edge_pts])
    i = np.concatenate([e[2] for e in edge_pts])
    xi, eta = sh.minipix_offsets()
    m = j * GRID + i
    P = Tn[p] + xi[m][:, None] * En[p] + eta[m][:, None] * Nn[p]
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    inside = geom.inside(P, ota[p])
    for q, r, c, ins in zip(p, j, i, inside):
        if ins and c == lo[q, r] - 1:
            lo[q, r] = c
        elif ins and c == hi[q, r] + 1:
            hi[q, r] = c
        elif not ins and c == lo[q, r]:
            lo[q, r] = c + 1
        elif not ins and c == hi[q, r]:
            hi[q, r] = c - 1


def runs_count(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return the covered minipixes per row (n, 64)."""
    return np.maximum(hi - lo + 1, 0)


def field_angle(
    geom: Geometry,
    R: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    N: np.ndarray,
    ota: np.ndarray,
    runs: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the field angle (deg from the OTA axis) of n pairs, as ``TansegCoverage.pair_field_angles``.

    Without ``runs`` the point is the tile centre (full pairs). With the partial pairs' row runs it is the centroid
    of the covered minipixes.
    """
    P = rotate(R, T)
    if runs is not None:
        lo, hi = runs
        cnt = runs_count(lo, hi).astype(np.float64)
        nc = cnt.sum(1)
        nw = np.maximum(nc, 1.0)
        mx = np.where(nc > 0, XI0 + 0.5 * DXI * np.einsum("ij,ij->i", cnt, lo + hi.astype(np.float64)) / nw, 0.0)
        my = (cnt @ ETA_ROWS) / nw
        P = P + mx[:, None] * rotate(R, E) + my[:, None] * rotate(R, N)
        P /= np.linalg.norm(P, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((P * geom.C[ota]).sum(-1), -1.0, 1.0)))


def airmass(R_mid: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Return the Pickering (2002) airmass of the tile centre, with R_mid the rotation at mid-ratchet, theta 0.

    The survey evaluates the photometry at the tile centre's altitude at mid-ratchet (``Survey.run``).
    """
    from ..observatory import pickering_airmass

    V = rotate(R_mid, T)
    return pickering_airmass(np.degrees(np.arcsin(np.clip(V[:, 2], -1.0, 1.0))))


def throughput(table: dict, band: np.ndarray, X: np.ndarray, transparency: np.ndarray) -> np.ndarray:
    """Return the full-system signal throughput signal_throughput(band, X) x transparency.

    ``table`` holds ``airmass`` (the grid) and ``log_tp`` (one list per band index), from ``index.json``.
    """
    grid = np.asarray(table["airmass"], np.float64)
    out = np.empty(len(X), np.float64)
    for b, logt in enumerate(table["log_tp"]):
        sel = band == b
        if sel.any():
            out[sel] = np.exp(np.interp(np.asarray(X, np.float64)[sel], grid, np.asarray(logt, np.float64)))
    return out * transparency


def covered(geom: Geometry, R: np.ndarray, tanseg_id: int, minipix: int, tel_id, point=None) -> np.ndarray:
    """Return whether the minipix centre is inside each OTA under each ratchet rotation R (n, 3, 3).

    ``point`` (an ICRS unit vector) replaces the minipix centre.
    """
    if point is not None:
        P = rotate(R, np.asarray(point, np.float64))
    else:
        T, E, N = (v[0] for v in tile_frames(tanseg_id))
        xi, eta = sh.minipix_offsets()
        P = rotate(R, T) + xi[minipix] * rotate(R, E) + eta[minipix] * rotate(R, N)
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    return geom.inside(P, geom.ota(tel_id))


def cone_meets_ota(geom: Geometry, R: np.ndarray, tel_id, centre, radius_rad: float) -> np.ndarray:
    """Return whether a cone (ICRS unit vector ``centre``, radius in radians) overlaps each OTA field under each R.

    The field is a spherical quadrilateral. The cone overlaps it when its centre is inside or lies within the
    radius of one of its four edges.
    """
    ota = geom.ota(tel_id)
    c = rotate(R, np.asarray(centre, np.float64))
    C, et, er = geom.C[ota], geom.et[ota], geom.er[ota]
    corners = [C + s1 * geom.tl * et + s2 * geom.ts * er for s1, s2 in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    corners = [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in corners]
    hit = geom.inside(c, ota)
    cos_r = np.cos(radius_rad)
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        n = np.cross(a, b)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        s = (c * n).sum(-1)
        q = c - s[:, None] * n
        on_arc = ((np.cross(a, q) * n).sum(-1) >= 0) & ((np.cross(q, b) * n).sum(-1) >= 0)
        hit |= on_arc & (np.abs(s) <= np.sin(radius_rad))
        hit |= ((c * a).sum(-1) >= cos_r) | ((c * b).sum(-1) >= cos_r)
    return hit
