"""The HDPS sky tessellation (TAN segments and minipixes), reproduced for ArgusSim.

hdps-skymap itself is not a dependency because it pulls hdps-core[gpu].

PROVENANCE: panoptes origin/main d8dc204b, packages/hdps_skymap/src/hdps/skymap/
  tiling_scheme.py:129-131 (DEFAULT_TILE_SIZE_ARCSEC 1800, DEFAULT_OVERLAP_ARCSEC 28.125,
                            DEFAULT_PIXEL_SCALE 900/1024), :534-579 (_ra_centers, _dec_centers),
                            :581-642 (generate_tiles)
  tanseg_ids.py:78-89 (MAX_RA_SLOTS 1440, MAX_DEC_BANDS 720, 64x64 minipixes of 32 px, 2048 px segment),
                :245-321 (compute_stable_tanseg_id), :688-837 (radec_to_composite_id),
                :881-981 (composite_id_to_radec)
  docs/source/components/skymap/explanation/sky-tiling.rst
Reproduced: the all-sky scheme only (dec_min -90, ra_min 0) and only the tile enumeration and ID math.
Where sky-tiling.rst says the lookup rounds to the *nearest* band and slot, the code (and this shim) floors
into the band/slot strips; the code is the reference.  tests/test_skymap_shim.py checks parity against
hdps.skymap at d8dc204b (a source tree named by HDPS_SKYMAP_REF, loaded by path) and skips without it.

The functions take an array module ``xp`` (numpy or cupy) so the same arithmetic runs on either backend.
"""

from __future__ import annotations

import numpy as np

PANOPTES_COMMIT = "d8dc204b"

# tanseg_ids.py:78-89
MAX_RA_SLOTS = 1440
MAX_DEC_BANDS = 720
MINIPIX_GRID_SIZE = 64
MINIPIX_PIXEL_SIZE = 32
SEGMENT_PIXEL_SIZE = 2048
# tiling_scheme.py:129-131 (all_sky defaults)
TILE_SIZE_ARCSEC = 1800.0
OVERLAP_ARCSEC = 28.125
PIXEL_SCALE_ARCSEC = 900.0 / 1024.0
SPACING_DEG = (TILE_SIZE_ARCSEC - OVERLAP_ARCSEC) / 3600.0  # 0.4921875
TILE_SIZE_DEG = TILE_SIZE_ARCSEC / 3600.0  # 0.5
PIXEL_SCALE_DEG = PIXEL_SCALE_ARCSEC / 3600.0
DEC_MIN, RA_MIN = -90.0, 0.0
POLE_COS = 0.013  # bands with cos(dec) below this are skipped (|dec| > ~89.3)
N_MINIPIX = MINIPIX_GRID_SIZE * MINIPIX_GRID_SIZE


def tiles(dec_lo: float = -90.0, dec_hi: float = 90.0) -> dict[str, np.ndarray]:
    """Every tile of the all-sky scheme whose centre dec lies in [dec_lo, dec_hi], as generate_tiles makes them.

    Returns tanseg_id (int32), ra_center, dec_center (deg), dec_band and ra_slot.
    """
    n_dec = int(np.ceil((90.0 - DEC_MIN) / SPACING_DEG))
    tid, ra_c, dec_c, band, slot = [], [], [], [], []
    for j in range(n_dec):
        dec_center = DEC_MIN + (j + 0.5) * SPACING_DEG
        cos_dec = np.cos(np.radians(dec_center))
        if cos_dec < POLE_COS or not (dec_lo <= dec_center <= dec_hi):
            continue
        ra_sp = SPACING_DEG / cos_dec
        n_ra = max(1, int(np.ceil(360.0 / ra_sp)))
        i = np.arange(n_ra)
        lo = RA_MIN + i * ra_sp
        hi = np.minimum(lo + ra_sp, 360.0)
        # the dec band is compute_stable_tanseg_id's floor of the band centre, which is j
        dec_band = min(max(int(np.floor((dec_center - DEC_MIN) / SPACING_DEG)), 0), MAX_DEC_BANDS - 1)
        tid.append(dec_band * MAX_RA_SLOTS + i)
        ra_c.append((0.5 * (lo + hi)) % 360.0)
        dec_c.append(np.full(n_ra, dec_center))
        band.append(np.full(n_ra, dec_band))
        slot.append(i)
    cat = np.concatenate
    return {
        "tanseg_id": cat(tid).astype(np.int32),
        "ra_center": cat(ra_c),
        "dec_center": cat(dec_c),
        "dec_band": cat(band).astype(np.int32),
        "ra_slot": cat(slot).astype(np.int32),
    }


def tile_center(tanseg_id, xp=np):
    """(ra, dec) of the tile centre for tanseg_id, as composite_id_to_radec derives it."""
    tanseg_id = xp.asarray(tanseg_id)
    dec_band = tanseg_id // MAX_RA_SLOTS
    ra_slot = tanseg_id % MAX_RA_SLOTS
    tile_dec = DEC_MIN + dec_band * SPACING_DEG + SPACING_DEG / 2
    cos_t = xp.cos(xp.radians(tile_dec))
    sp = xp.where(cos_t < POLE_COS, 360.0, SPACING_DEG / cos_t)
    lo = RA_MIN + ra_slot * sp
    hi = xp.minimum(RA_MIN + (ra_slot + 1) * sp, 360.0)
    tile_ra = 0.5 * (lo + hi)
    return xp.where(tile_ra >= 360, tile_ra - 360, tile_ra), tile_dec


def radec_to_composite_id(ra, dec, xp=np):
    """(ra, dec) -> (tanseg_id int32, local_idx int16), tanseg_ids.radec_to_composite_id for the all-sky scheme."""
    ra = xp.asarray(ra, dtype=xp.float64)
    dec = xp.asarray(dec, dtype=xp.float64)
    cos_dec, sin_dec = xp.cos(xp.radians(dec)), xp.sin(xp.radians(dec))
    dec_band = xp.clip(xp.floor((dec - DEC_MIN) / SPACING_DEG).astype(xp.int32), 0, MAX_DEC_BANDS - 1)
    band_dec = DEC_MIN + (dec_band + 0.5) * SPACING_DEG
    cos_band = xp.cos(xp.radians(band_dec))
    ra_spacing = xp.where(cos_band < POLE_COS, 360.0, SPACING_DEG / cos_band)
    ra_norm = ra - RA_MIN
    ra_norm = xp.where(ra_norm < 0, ra_norm + 360, ra_norm)
    ra_norm = xp.where(ra_norm >= 360, ra_norm - 360, ra_norm)
    ra_slot = xp.floor(ra_norm / ra_spacing).astype(xp.int32)
    max_slots = xp.ceil(360.0 / ra_spacing).astype(xp.int32)
    ra_slot = ra_slot % max_slots
    ra_slot = xp.where(cos_band < POLE_COS, 0, ra_slot)
    tanseg_id = dec_band * MAX_RA_SLOTS + ra_slot
    tile_ra, tile_dec = tile_center(tanseg_id, xp)
    cos_td, sin_td = xp.cos(xp.radians(tile_dec)), xp.sin(xp.radians(tile_dec))
    dra = xp.radians(ra - tile_ra)
    denom = sin_td * sin_dec + cos_td * cos_dec * xp.cos(dra)
    xi = xp.degrees(cos_dec * xp.sin(dra) / denom)
    eta = xp.degrees((cos_td * sin_dec - sin_td * cos_dec * xp.cos(dra)) / denom)
    crpix = SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    px = xp.clip(xp.floor(-xi / PIXEL_SCALE_DEG + crpix - 0.5).astype(xp.int32), 0, SEGMENT_PIXEL_SIZE - 1)
    py = xp.clip(xp.floor(eta / PIXEL_SCALE_DEG + crpix - 0.5).astype(xp.int32), 0, SEGMENT_PIXEL_SIZE - 1)
    local = (py // MINIPIX_PIXEL_SIZE) * MINIPIX_GRID_SIZE + px // MINIPIX_PIXEL_SIZE
    return tanseg_id.astype(xp.int32), local.astype(xp.int16)


def minipix_offsets(xp=np):
    """Return standard coordinates (xi, eta), radians, of the 4096 minipix centres of any tile, in local_idx order.

    xi is positive toward +RA (FITS pixels run the other way, hence the sign), as in composite_id_to_radec.
    """
    idx = xp.arange(N_MINIPIX)
    ix, iy = idx % MINIPIX_GRID_SIZE, idx // MINIPIX_GRID_SIZE
    crpix = SEGMENT_PIXEL_SIZE / 2.0 + 0.5
    xi = -((ix + 0.5) * MINIPIX_PIXEL_SIZE + 0.5 - crpix) * PIXEL_SCALE_DEG
    eta = ((iy + 0.5) * MINIPIX_PIXEL_SIZE + 0.5 - crpix) * PIXEL_SCALE_DEG
    return xp.radians(xi), xp.radians(eta)


def tile_corner_offsets(xp=np):
    """Return standard coordinates (radians) of the tile's four pixel-grid corners (edges of pixels 0 and 2047)."""
    half = (SEGMENT_PIXEL_SIZE / 2.0) * PIXEL_SCALE_DEG
    c = xp.radians(xp.asarray([half, half, -half, -half]))
    d = xp.radians(xp.asarray([half, -half, -half, half]))
    return c, d


def tangent_basis(ra_deg, dec_deg, xp=np):
    """Return unit vectors (x, y, z in the RA/Dec frame): tile centre T, local east E and north N."""
    a, d = xp.radians(ra_deg), xp.radians(dec_deg)
    T = xp.stack([xp.cos(d) * xp.cos(a), xp.cos(d) * xp.sin(a), xp.sin(d)], axis=-1)
    E = xp.stack([-xp.sin(a), xp.cos(a), xp.zeros_like(a)], axis=-1)
    N = xp.stack([-xp.sin(d) * xp.cos(a), -xp.sin(d) * xp.sin(a), xp.cos(d)], axis=-1)
    return T, E, N
