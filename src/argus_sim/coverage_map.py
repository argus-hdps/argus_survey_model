"""Per-filter HealSparse coverage maps.

Builds sparse HEALPix maps that track how many cameras observe each sky
pixel in each filter band, using alt/az coordinates so the filter pattern
is fixed to the instrument frame regardless of observation time.
"""

from __future__ import annotations

import astropy.table as tbl
import astropy.units as u
import healsparse as hsp
import numpy as np
from mocpy import MOC


def build_filter_maps(
    cam_table: tbl.Table,
    long_ext: u.Quantity[u.deg],
    short_ext: u.Quantity[u.deg],
    moc_depth: int,
    nside_sparse: int,
) -> dict[str, hsp.HealSparseMap]:
    """Build per-filter HealSparse hit-count maps from a camera table.

    Maps are constructed in alt/az space so the filter pattern is fixed
    to the instrument frame.

    Parameters
    ----------
    cam_table : astropy.table.Table
        Camera pointings table with ``alt_center``, ``az_center``,
        ``rot_deg``, and ``filter`` columns.
    long_ext : u.Quantity[u.deg]
        Camera long-axis field of view.
    short_ext : u.Quantity[u.deg]
        Camera short-axis field of view.
    moc_depth : int
        HEALPix depth for MOC construction.
    nside_sparse : int
        NSIDE for the output HealSparse maps.

    Returns
    -------
    dict[str, HealSparseMap]
        Mapping from filter name to a HealSparse map of camera hit counts
        (dtype int16).

    """
    filters = sorted(set(cam_table["filter"]))
    filter_maps = {}

    for filt in filters:
        mask = np.array(cam_table["filter"]) == filt
        az = np.array(cam_table["az_center"][mask])
        alt = np.array(cam_table["alt_center"][mask])
        rot = np.array(cam_table["rot_deg"][mask])

        rot_vals = np.unique(rot)
        if len(rot_vals) == 1:
            boxes = MOC.from_boxes(
                lon=az * u.deg,
                lat=alt * u.deg,
                a=long_ext / 2,
                b=short_ext / 2,
                angle=(rot_vals[0] + 90.0) * u.deg,
                max_depth=moc_depth,
            )
        else:
            boxes = [
                MOC.from_boxes(
                    lon=[a] * u.deg,
                    lat=[al] * u.deg,
                    a=long_ext / 2,
                    b=short_ext / 2,
                    angle=(ro + 90.0) * u.deg,
                    max_depth=moc_depth,
                )
                for a, al, ro in zip(az, alt, rot)
            ]

        if isinstance(boxes, list):
            moc = sum(boxes)
        else:
            moc = boxes

        moc_pixels = moc.flatten().astype(int)

        moc_nside = 2**moc_depth
        if nside_sparse > moc_nside:
            ratio = (nside_sparse // moc_nside) ** 2
            sparse_pixels = np.repeat(moc_pixels * ratio, ratio) + np.tile(np.arange(ratio), len(moc_pixels))
        elif nside_sparse < moc_nside:
            ratio = (moc_nside // nside_sparse) ** 2
            sparse_pixels = np.unique(moc_pixels // ratio)
        else:
            sparse_pixels = moc_pixels

        hsmap = hsp.HealSparseMap.make_empty(nside_coverage=32, nside_sparse=nside_sparse, dtype=np.int16)
        unique_pix, counts = np.unique(sparse_pixels, return_counts=True)
        hsmap[unique_pix] = counts.astype(np.int16)

        filter_maps[filt] = hsmap

    return filter_maps


def filter_at_pixels(
    filter_maps: dict[str, hsp.HealSparseMap],
    az_deg: np.ndarray,
    alt_deg: np.ndarray,
) -> np.ndarray:
    """Return the dominant filter at each sky position.

    For pixels covered by multiple filters, the filter with the highest
    camera count wins; ties go to the alphabetically first filter name.
    Adjacent filter strips overlap by one camera each, so every contested
    pixel in a strip layout is a tie -- use :func:`rows_per_filter` when
    each covering band must be kept.  Positions are given in alt/az to
    match the instrument-frame filter maps.

    Parameters
    ----------
    filter_maps : dict[str, HealSparseMap]
        Per-filter coverage maps from :func:`build_filter_maps`.
    az_deg : array-like
        Azimuth in degrees.
    alt_deg : array-like
        Altitude in degrees.

    Returns
    -------
    numpy.ndarray
        Array of filter name strings, one per input position.
        Positions outside all coverage get an empty string.

    """
    az_deg = np.asarray(az_deg, dtype=np.float64)
    alt_deg = np.asarray(alt_deg, dtype=np.float64)
    n = len(az_deg)
    best_filter = np.full(n, "", dtype="U4")
    best_count = np.zeros(n, dtype=np.int16)

    for filt in sorted(filter_maps):
        counts = filter_maps[filt].get_values_pos(az_deg, alt_deg, lonlat=True)
        better = counts > best_count
        best_filter[better] = filt
        best_count[better] = counts[better]

    return best_filter


def rows_per_filter(
    table: tbl.Table,
    filter_maps: dict[str, hsp.HealSparseMap],
) -> tbl.Table:
    """Expand a pixel table to one row per (pixel, covering filter).

    A pixel under two adjacent filter strips is imaged in both bands at
    once, so it gets one row per band.  Pixels outside every filter map
    are dropped: they are the one-to-two-cell rind of partially covered
    edge cells that the footprint MOC includes but no camera box centre-
    covers, and they never contribute to a depth map.  Rows are ordered by
    ``healpix`` then filter name.

    Parameters
    ----------
    table : astropy.table.Table
        One row per pixel with ``az`` and ``alt`` columns in degrees.
    filter_maps : dict[str, HealSparseMap]
        Per-filter camera hit-count maps from :func:`build_filter_maps`.

    """
    az_deg = np.asarray(table["az"].data, dtype=np.float64)
    alt_deg = np.asarray(table["alt"].data, dtype=np.float64)
    covered = {filt: filter_maps[filt].get_values_pos(az_deg, alt_deg, lonlat=True) > 0 for filt in sorted(filter_maps)}
    pieces = []
    for filt, sel in covered.items():
        piece = table[sel]
        piece["filter"] = np.full(len(piece), filt, dtype="U4")
        pieces.append(piece)
    out = tbl.vstack(pieces, metadata_conflicts="silent")
    out.sort(["healpix", "filter"])
    return out
