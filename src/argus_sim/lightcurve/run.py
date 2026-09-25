"""SurveyRun: visit tables for sky positions from a visit store, on local disk or in S3."""

from __future__ import annotations

import io
import json
import os

import astropy.table as tbl
import numpy as np

from .. import skymap_shim as sh
from . import geometry as g
from . import photometry
from .storage import Storage
from .store import FORMAT_VERSION, build_index, dequantise

SLOT_S = 60.0
DAY_S = 86400.0
POSITIONS = ("exact", "minipix")
CONE_POSITIONS = ("exact", "tile")
MAX_OPEN_SHARDS = 64


class SurveyRun:
    """A survey run, read through its visit store (``SurveyRun.build_index`` makes the store).

    ``visits`` returns one row per 60-s slot and OTA for every exposure that covered a position. The records come
    from the tile that owns the position (``skymap_shim.radec_to_composite_id``). Rows are sorted by mjd, then
    tel_id, and ``mjd`` is the slot start.
    """

    def __init__(self, path: str, *, index_dir: str | None = None, storage_options: dict | None = None):
        """Open a visit store.

        ``path`` is the store directory, a run directory whose store is in ``lightcurve/``, or an
        ``s3://bucket/prefix`` URI. ``storage_options`` go to ``pyarrow.fs.S3FileSystem`` (endpoint_override,
        access_key, secret_key, region, anonymous, ...). ``index_dir`` names the store directory of a run
        directory explicitly.
        """
        if index_dir is None and not str(path).startswith("s3://"):
            p = os.path.abspath(os.path.expanduser(path))
            if not os.path.isfile(os.path.join(p, "index.json")) and os.path.isfile(
                os.path.join(p, "lightcurve", "index.json")
            ):
                index_dir = os.path.join(p, "lightcurve")
        self.store = Storage(index_dir or str(path), storage_options)
        if not self.store.exists("index.json"):
            raise FileNotFoundError(
                f"No visit store in {self.store.uri}. Build one with: python -m argus_sim.lightcurve build RUN_DIR "
                "--out STORE_DIR"
            )
        self.meta = json.loads(self.store.read_bytes("index.json"))
        if self.meta["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"The visit store has format {self.meta['format_version']}; this code reads format {FORMAT_VERSION}. "
                "Rebuild it with: python -m argus_sim.lightcurve build RUN_DIR --out STORE_DIR"
            )
        self.bands = [str(b) for b in self.meta["bands"]]
        self.zeropoints = {b: dict(v) for b, v in self.meta["zeropoints"].items()}
        self.periods = [*self.meta["periods"], "all"]
        r = self.store.load_npz("ratchets.npz")
        self.ratchets = tbl.Table({k: r[k] for k in r if r[k].ndim == 1})
        self._rot, self._rot_mid, self._moon = r["rotation"], r["rotation_mid"], r["moon_altaz"]
        self._ratchet_ids = np.asarray(r["ratchet"], np.int64)
        self._transparency = np.asarray(r["transparency"], np.float64)
        n_slots = np.asarray(r["n_epochs"]) * np.asarray(r["exptime_s"]) / SLOT_S
        self._n_slots = np.rint(n_slots).astype(np.int64)
        self._n_frames = np.rint(SLOT_S / np.asarray(r["exptime_s"])).astype(np.int64)
        if np.any(np.abs(n_slots - self._n_slots) > 1e-9) or np.any(self._n_frames * r["exptime_s"] != SLOT_S):
            raise ValueError("A ratchet's exposures do not fill whole 60-s slots.")
        # mjd_first is the first dark exposure's mid-time (coverage_exact.Schedule.exposures)
        self._slot0 = np.asarray(r["mjd_first"], np.float64) - 0.5 * SLOT_S / DAY_S
        lay = self.store.load_npz("layout.npz")
        self.geometry = g.Geometry(**{k: lay[k] for k in ("tel_id", "subarray", "C", "et", "er", "tl", "ts")})
        self._band_of_tel = lay["band_of_tel"].astype(np.int64)
        self._tiles = self.store.load_npz("tiles.npz")
        self._quant = self.meta["quantisation"]["sigma_bkg_e"]
        self._tp_table = self.meta["throughput_table"]
        self._shards: dict = {}
        self._depth = None

    @classmethod
    def build_index(cls, run_dir: str, *, index_dir: str | None = None, workers: int | None = None, **kw) -> SurveyRun:
        """Build (or finish) the visit store of ``run_dir`` in ``index_dir`` (default ``run_dir/lightcurve/``) and open it.

        Keywords as ``argus_sim.lightcurve.store.build_index`` (``max_memory_gb``, ``check_every``, ...).
        """
        return cls(build_index(run_dir, index_dir=index_dir, workers=workers, **kw))

    def __repr__(self) -> str:
        """Name the store and its size."""
        m = self.meta
        return (
            f"SurveyRun({self.store.uri!r}: {m['n_ratchets']} ratchets, {m['n_tiles']} tiles, "
            f"{m['n_pair_rows']:,} pair rows, bands {self.bands})"
        )

    def io_stats(self) -> dict:
        """Return the store reads made so far: ``get_requests`` (one per read, a GET on S3) and ``bytes_read``."""
        return self.store.io_stats()

    # ------------------------------------------------------------------ photometry bound to the run's zero points
    def snr(self, vis, mag, **kw):
        """``photometry.snr`` with this run's zero points."""
        return photometry.snr(vis, mag, self.zeropoints, **kw)

    def sigma_mag(self, vis, mag, **kw):
        """``photometry.sigma_mag`` with this run's zero points."""
        return photometry.sigma_mag(vis, mag, self.zeropoints, **kw)

    def limit_mag(self, vis, **kw):
        """``photometry.limit_mag`` with this run's zero points."""
        return photometry.limit_mag(vis, self.zeropoints, **kw)

    def inject(self, vis, model, *, rng, **kw):
        """``photometry.inject`` with this run's zero points."""
        return photometry.inject(vis, model, self.zeropoints, rng=rng, **kw)

    # ------------------------------------------------------------------ summaries
    def nights(self) -> tbl.Table:
        """Return the survey's nightly summary: one row per night and band (depths, areas, Moon, weather)."""
        import pyarrow.parquet as pq

        return tbl.Table.from_pandas(pq.read_table(io.BytesIO(self.store.read_bytes("nights.parquet"))).to_pandas())

    def depth_map(self, band: str, period: str = "all") -> tbl.Table:
        """Return the tile depth map of one band and period (a label of ``run.periods``; "all" is the whole run).

        One row per tile with records: ``tanseg_id``, the tile centre ``ra`` and ``dec`` (deg), ``limmag`` (the
        5-sigma depth of the period's stack, AB mag, median over the tile's owned minipixes that the period
        covered; NaN when the period covered only the strip the tile shares with its neighbours),
        ``covered_fraction`` (of the owned minipixes), ``n_nights`` (nights with a record of the tile),
        and ``n_slots_60``, ``n_slots_1`` and ``n_frames`` summed over the tile's (tile, OTA) pairs, so a slot
        seen by two OTAs counts twice. Periods are half-years (January-June, July-December) by the date of the
        morning that ends each night.
        """
        key = f"{band}/{period}"
        groups = self.meta["depth"]["row_groups"]
        if key not in groups:
            raise ValueError(
                f"No depth map for band {band!r} and period {period!r}; bands {self.bands}, periods {self.periods}."
            )
        if self._depth is None:
            self._depth = self.store.parquet("depth.parquet")
        t = self._depth.read_row_group(groups[key]).drop_columns(["band", "period"])
        out = tbl.Table({k: t.column(k).to_numpy() for k in t.column_names})
        out.meta.update(band=band, period=period, limmag=self.meta["depth"]["limmag"])
        return out

    # ------------------------------------------------------------------ store access
    def _shard(self, shard: int):
        if shard not in self._shards:
            if len(self._shards) >= MAX_OPEN_SHARDS:
                self._shards.pop(next(iter(self._shards)))
            self._shards[shard] = self.store.parquet(f"pairs/shard_{shard:04d}.parquet")
        return self._shards[shard]

    def _tile_rows(self, tanseg_id: int) -> dict | None:
        tids = self._tiles["tanseg_id"]
        i = int(np.searchsorted(tids, tanseg_id))
        if i == len(tids) or tids[i] != tanseg_id:
            return None
        t = self._shard(int(self._tiles["shard"][i])).read_row_group(int(self._tiles["row_group"][i]))
        a = int(self._tiles["offset"][i])
        t = t.slice(a, int(self._tiles["n_rows"][i]))
        cols = {k: t.column(k).to_numpy() for k in ("ratchet", "tel_id", "partial")}
        cols["sigma_bkg_e"] = dequantise(t.column("sigma_bkg_q").to_numpy(), self._quant)
        cols["band"] = self._band_of_tel[cols["tel_id"]]
        cols["_rpos"] = np.searchsorted(self._ratchet_ids, cols["ratchet"])
        cols["_tanseg_id"] = int(tanseg_id)
        return cols

    def _covering(self, cols: dict, rows: np.ndarray, minipix: int, point=None) -> np.ndarray:
        """Return the pair rows, out of ``rows``, whose OTA covers the position."""
        part = cols["partial"][rows]
        keep = ~part
        if part.any():
            p = rows[part]
            R = self._rot[cols["_rpos"][p]]
            keep[part] = g.covered(self.geometry, R, cols["_tanseg_id"], minipix, cols["tel_id"][p], point)
        return rows[keep]

    def _prefilter(self, cols: dict, band, mjd_min, mjd_max) -> np.ndarray:
        """Pair rows in the requested bands with at least one slot inside [mjd_min, mjd_max]."""
        keep = np.ones(len(cols["ratchet"]), bool)
        if band is not None:
            want = [band] if isinstance(band, str) else list(band)
            unknown = sorted(set(want) - set(self.bands))
            if unknown:
                raise ValueError(f"Unknown band {unknown}; this run has bands {self.bands}.")
            keep &= np.isin(cols["band"], [self.bands.index(b) for b in want])
        rp = cols["_rpos"]
        if mjd_min is not None:
            keep &= self._slot0[rp] + (self._n_slots[rp] - 1) * (SLOT_S / DAY_S) >= mjd_min
        if mjd_max is not None:
            keep &= self._slot0[rp] <= mjd_max
        return np.flatnonzero(keep)

    @staticmethod
    def _window(d: dict, mjd_min, mjd_max) -> dict:
        if mjd_min is None and mjd_max is None:
            return d
        m = np.ones(len(d["mjd"]), bool)
        if mjd_min is not None:
            m &= d["mjd"] >= mjd_min
        if mjd_max is not None:
            m &= d["mjd"] <= mjd_max
        return {k: v[m] for k, v in d.items()}

    def _pair_values(self, cols: dict, rows: np.ndarray, n_covered: bool = False) -> dict:
        """Recompute airmass, throughput and field angle (and optionally n_covered) of the given pair rows."""
        n = len(rows)
        rp = cols["_rpos"][rows]
        T, E, N = (np.broadcast_to(v[0], (n, 3)) for v in g.tile_frames(cols["_tanseg_id"]))
        X = g.airmass(self._rot_mid[rp], T)
        out = {
            "airmass": X,
            "throughput": g.throughput(self._tp_table, cols["band"][rows], X, self._transparency[rp]),
            "field_angle": np.empty(n),
        }
        if n_covered:
            out["n_covered"] = np.full(n, sh.N_MINIPIX, np.int64)
        ota = self.geometry.ota(cols["tel_id"][rows])
        R = self._rot[rp]
        part = cols["partial"][rows]
        full = ~part
        out["field_angle"][full] = g.field_angle(self.geometry, R[full], T[full], E[full], N[full], ota[full])
        if part.any():
            p = np.flatnonzero(part)
            runs = g.row_runs(self.geometry, R[p], T[p], E[p], N[p], ota[p])
            out["field_angle"][p] = g.field_angle(self.geometry, R[p], T[p], E[p], N[p], ota[p], runs=runs)
            if n_covered:
                out["n_covered"][p] = g.runs_count(*runs).sum(1)
        return out

    def _slots(self, cols: dict, rows: np.ndarray, n_covered: bool = False) -> dict:
        pv = self._pair_values(cols, rows, n_covered) if len(rows) else None
        rp = cols["_rpos"][rows]
        ns = self._n_slots[rp]
        rep = np.repeat(np.arange(len(rows)), ns)
        k = np.arange(len(rep)) - np.repeat(np.cumsum(ns) - ns, ns)
        src, rsrc = rows[rep], rp[rep]
        exptime = np.asarray(self.ratchets["exptime_s"], np.float64)[rsrc]

        def per_pair(name, dtype=np.float64):
            return pv[name][rep] if pv is not None else np.zeros(0, dtype)

        out = {
            "mjd": self._slot0[rsrc] + k * (SLOT_S / DAY_S),
            "band": np.asarray(self.bands)[cols["band"][src]],
            "tel_id": cols["tel_id"][src].astype(np.int64),
            "subarray": self.geometry.subarray[self.geometry.ota(cols["tel_id"][src])],
            "exptime_s": exptime,
            "n_frames": self._n_frames[rsrc],
            "sigma_bkg_e": cols["sigma_bkg_e"][src],
            "throughput": per_pair("throughput"),
            "airmass": per_pair("airmass"),
            "field_angle": per_pair("field_angle"),
            "seeing_zenith": np.asarray(self.ratchets["seeing_zenith"], np.float64)[rsrc],
            "transparency": self._transparency[rsrc],
            "ratchet": cols["ratchet"][src].astype(np.int64),
            "night": np.asarray(self.ratchets["night"], np.int64)[rsrc],
            "slot": k,
            "_pair": src,
        }
        if n_covered:
            out["n_covered"] = per_pair("n_covered", np.int64)
        return out

    @staticmethod
    def _table(parts: list[dict], sort_keys: tuple[str, ...], drop=("_pair",)) -> tbl.Table:
        cols = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in drop}
        order = np.lexsort(tuple(cols[k] for k in reversed(sort_keys)))
        return tbl.Table({k: v[order] for k, v in cols.items()})

    def _empty(self, n_covered: bool = False) -> dict:
        cols = {
            "ratchet": np.zeros(0, np.int32),
            "tel_id": np.zeros(0, np.int16),
            "partial": np.zeros(0, bool),
            "sigma_bkg_e": np.zeros(0),
            "band": np.zeros(0, np.int64),
            "_rpos": np.zeros(0, np.int64),
            "_tanseg_id": 0,
        }
        return self._slots(cols, np.zeros(0, np.int64), n_covered)

    # ------------------------------------------------------------------ queries
    def visits(
        self, ra: float, dec: float, *, position: str = "exact", band=None, mjd_min=None, mjd_max=None
    ) -> tbl.Table:
        """Return every 60-s slot that covered (ra, dec) in degrees, one row per slot and OTA.

        ``position="exact"`` tests the position itself. ``"minipix"`` tests the centre of the minipix that owns it,
        which is how the survey's own per-minipix sums count coverage. The two differ only within about 20 arcsec
        of an OTA edge. ``band`` (a name or a list of names), ``mjd_min`` and ``mjd_max`` (inclusive, on the slot
        start) select rows while the store is read. Rows are sorted by mjd, then tel_id.
        """
        out = self.visits_many(
            np.atleast_1d(ra), np.atleast_1d(dec), position=position, band=band, mjd_min=mjd_min, mjd_max=mjd_max
        )
        out.remove_column("source")
        out.meta.update(ra=float(ra), dec=float(dec))
        return out

    def visits_many(self, ra, dec, *, position: str = "exact", band=None, mjd_min=None, mjd_max=None) -> tbl.Table:
        """Return the visits of many positions in one table, with a ``source`` column (the index into the inputs).

        Takes the same keywords as ``visits``. Each tile is read once. Rows are sorted by source, then mjd, then
        tel_id. For hundreds of 5-year positions, call this in batches and reduce each batch before the next.
        """
        ra, dec = np.atleast_1d(np.asarray(ra, np.float64)), np.atleast_1d(np.asarray(dec, np.float64))
        if ra.shape != dec.shape or ra.ndim != 1:
            raise ValueError("ra and dec must be 1-d arrays of the same length.")
        if position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, not {position!r}.")
        tid, mp = sh.radec_to_composite_id(ra, dec)
        vec = sh.tangent_basis(ra, dec)[0] if position == "exact" else None
        parts = [dict(self._empty(), source=np.zeros(0, np.int64))]
        for t in np.unique(tid):
            cols = self._tile_rows(int(t))
            if cols is None:
                continue
            cand = self._prefilter(cols, band, mjd_min, mjd_max)
            for s in np.flatnonzero(tid == t):
                rows = self._covering(cols, cand, int(mp[s]), None if vec is None else vec[s])
                d = self._window(self._slots(cols, rows), mjd_min, mjd_max)
                d["source"] = np.full(len(d["mjd"]), s, np.int64)
                parts.append(d)
        out = self._table(parts, ("source", "mjd", "tel_id"))
        out.meta.update(bands=self.bands, position=position, tanseg_id=tid.tolist(), minipix=mp.astype(int).tolist())
        return out

    def night_log(self, ra: float, dec: float, *, position: str = "exact") -> tbl.Table:
        """Return one row per night of the survey plan saying what happened at (ra, dec).

        The plan is every night from the run's start to its end (``index.json`` ``plan``), including nights lost to
        weather. ``status`` is one of:

        - ``observed``: at least one slot covered the position.
        - ``weather``: the survey did not run that night (the weather cull removed it).
        - ``moon``: the position was inside an OTA field in at least one ratchet, but every such ratchet had its
          tile inside the Moon mask (``min_moon_sep_deg`` from the Moon at mid-ratchet), so nothing was recorded.
        - ``not_in_footprint``: no ratchet that night put the position inside an OTA field.

        ``n_slots`` counts the night's visit rows and ``n_fast_slots`` those that are fast slots. A night whose
        missing coverage the Moon mask does not explain raises ``RuntimeError``; that would mean the store and the
        run disagree.
        """
        planned = self.planned_nights()
        vis = self.visits(ra, dec, position=position)
        vn = np.asarray(vis["night"], np.int64)
        fast = np.asarray(vis["n_frames"]) > 1
        pos = np.searchsorted(planned, vn)
        if np.any(planned[np.minimum(pos, len(planned) - 1)] != vn):
            raise RuntimeError("The run has visits on nights outside its plan.")
        n_slots = np.bincount(pos, minlength=len(planned))
        n_fast = np.bincount(pos[fast], minlength=len(planned))
        rn = np.asarray(self.ratchets["night"], np.int64)
        status = np.where(np.isin(planned, rn), "not_in_footprint", "weather").astype("<U16")
        status[n_slots > 0] = "observed"
        todo = np.isin(rn, planned[status == "not_in_footprint"])
        if todo.any():
            rpos = np.flatnonzero(todo)
            point, tile_centre = self._position_vectors(ra, dec, position)
            in_field = self._in_any_field(rpos, point)
            hit = rpos[in_field]
            if len(hit):
                masked = self._moon_masked(hit, tile_centre)
                bad = np.unique(rn[hit[~masked]])
                if len(bad):
                    raise RuntimeError(
                        f"On nights {bad[:10].tolist()} the position was in an OTA field outside the Moon mask but "
                        "has no record. The store does not match the run."
                    )
                status[np.isin(planned, np.unique(rn[hit]))] = "moon"
        out = tbl.Table({"night": planned, "status": status, "n_slots": n_slots, "n_fast_slots": n_fast})
        out.meta.update(ra=float(ra), dec=float(dec), position=position, plan="index.json plan")
        return out

    def planned_nights(self) -> np.ndarray:
        """Return the night labels of the survey plan, including nights the weather cull removed."""
        planned = np.asarray(self.meta["plan"]["nights"], np.int64)
        missing = np.setdiff1d(np.asarray(self.ratchets["night"]), planned)
        if len(missing):
            raise RuntimeError(f"Nights {missing[:10].tolist()} have ratchets but are not in the run's plan.")
        return planned

    def _position_vectors(self, ra, dec, position):
        if position not in POSITIONS:
            raise ValueError(f"position must be one of {POSITIONS}, not {position!r}.")
        tid, mp = sh.radec_to_composite_id(np.array([ra], float), np.array([dec], float))
        T, E, N = g.tile_frames(tid)
        if position == "exact":
            point = sh.tangent_basis(np.array([ra], float), np.array([dec], float))[0][0]
        else:
            xi, eta = sh.minipix_offsets()
            point = T[0] + xi[int(mp[0])] * E[0] + eta[int(mp[0])] * N[0]
            point = point / np.linalg.norm(point)
        return point, T[0]

    def _in_any_field(self, rpos: np.ndarray, point: np.ndarray, chunk: int = 2000) -> np.ndarray:
        """Whether the point is inside any OTA field in each ratchet (positions into ``self.ratchets``)."""
        out = np.zeros(len(rpos), bool)
        ota = np.arange(len(self.geometry.tel_id))[None, :]
        for s0 in range(0, len(rpos), chunk):
            P = self._rot[rpos[s0 : s0 + chunk]] @ point
            out[s0 : s0 + chunk] = self.geometry.inside(P[:, None, :], ota).any(1)
        return out

    def _moon_masked(self, rpos: np.ndarray, tile_centre: np.ndarray) -> np.ndarray:
        """Whether the survey's Moon mask removed the tile in each ratchet.

        Repeats ``Survey._proc_ratchet_inner``: the angle between the tile centre and the Moon at mid-ratchet, both
        in the local alt/az frame, against ``min_moon_sep_deg``.
        """
        v = self._rot_mid[rpos] @ tile_centre
        cos = np.clip((v * self._moon[rpos]).sum(-1), -1.0, 1.0)
        return np.degrees(np.arccos(cos)) < self.meta["moon_mask"]["min_moon_sep_deg"]

    def visits_cone(
        self,
        ra: float,
        dec: float,
        radius_deg: float,
        *,
        position: str = "exact",
        band=None,
        mjd_min=None,
        mjd_max=None,
    ) -> tbl.Table:
        """Return the tile-level visits of a cone.

        The table has the slots of the (tile, OTA) pairs of every tile whose centre lies within radius + the tile's
        half-diagonal of the cone centre. With ``position="exact"`` a pair is kept when its OTA field overlaps the
        cone. With ``position="tile"`` every pair of those tiles is kept, whether or not the OTA reaches the cone.
        ``band``, ``mjd_min`` and ``mjd_max`` work as in ``visits``. Extra columns: ``tanseg_id``, ``partial``
        (the OTA covers part of the tile) and ``n_covered`` (the tile's minipixes it covers). Rows are sorted by
        tanseg_id, mjd, tel_id.
        """
        from ..tanseg_coverage import TILE_HALF_DIAG_DEG

        if position not in CONE_POSITIONS:
            raise ValueError(f"position must be one of {CONE_POSITIONS}, not {position!r}.")
        tids = self._tiles["tanseg_id"]
        if not hasattr(self, "_tile_vec"):
            self._tile_vec = g.tile_frames(tids)[0]
        v, _, _ = sh.tangent_basis(np.array([ra], float), np.array([dec], float))
        ang = np.degrees(np.arccos(np.clip(self._tile_vec @ v[0], -1.0, 1.0)))
        parts = [dict(self._empty(n_covered=True), tanseg_id=np.zeros(0, np.int64), partial=np.zeros(0, bool))]
        for t in tids[ang <= radius_deg + TILE_HALF_DIAG_DEG]:
            cols = self._tile_rows(int(t))
            rows = self._prefilter(cols, band, mjd_min, mjd_max)
            if position == "exact" and len(rows):
                R = self._rot[cols["_rpos"][rows]]
                rows = rows[g.cone_meets_ota(self.geometry, R, cols["tel_id"][rows], v[0], np.radians(radius_deg))]
            d = self._window(self._slots(cols, rows, n_covered=True), mjd_min, mjd_max)
            d["tanseg_id"] = np.full(len(d["mjd"]), t, np.int64)
            d["partial"] = cols["partial"][d["_pair"]]
            parts.append(d)
        out = self._table(parts, ("tanseg_id", "mjd", "tel_id"))
        out.meta.update(ra=float(ra), dec=float(dec), radius_deg=float(radius_deg), bands=self.bands, position=position)
        return out
