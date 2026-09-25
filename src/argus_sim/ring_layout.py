"""Ring-based telescope layout for concentric subarray geometry.

Loads pre-computed ring layouts (zenith distance, azimuth, subarray assignment,
rotation) and converts them into the same pointing table format used by the
grid tiling path, so CradlePointing handles both tilings the same way.
"""

from __future__ import annotations

import dataclasses
from importlib.resources import files

import astropy.coordinates as crds
import astropy.table as tbl
import astropy.time as atime
import astropy.units as u
import astropy_healpix as ahpx
import numpy as np
from mocpy import MOC

from argus_sim import c, get_logger

log = get_logger(__name__)

_LAYOUT_DIR = files("argus_sim").joinpath("data", "layouts")


@dataclasses.dataclass(frozen=True)
class RingLayoutSpec:
    """A registered ring layout: its data file and how to read it.

    ``az_offset_deg`` is added to the file's azimuths on load.  ``fov_deg`` is the
    (long, short) camera field the layout was designed with; None leaves the
    footprint to the configured telescope field.
    """

    name: str
    filename: str
    az_offset_deg: float
    fov_deg: tuple[float, float] | None
    description: str
    band_table: str | None = None


LAYOUTS = {
    "A170r_1200": RingLayoutSpec(
        name="A170r_1200",
        filename="A170r_1200_telescopes.csv",
        # az_deg is the station bearing; the footprint is at frame az + 180, compass az + 90
        # (as array_arrangement's fov_hits_xyz and tracking_matrix place it).
        az_offset_deg=90.0,
        fov_deg=(3.252897, 2.442494),
        description="canonical 1200-OTA layout, argus-hdps/array_arrangement 9206cc3",
        # E/W mirror-antisymmetric alternation table at 5% balance
        band_table="A170r_1200_bands_alt5_2026-09-23.csv",
    ),
    "A170r_1200-strips": RingLayoutSpec(
        name="A170r_1200-strips",
        filename="A170r_1200_telescopes.csv",
        az_offset_deg=90.0,
        fov_deg=(3.252897, 2.442494),
        description="A170r_1200 with the strip-rule band assignment (assign_filters), for comparison",
    ),
}
DEFAULT_LAYOUT = "A170r_1200"


def load_ring_layout(path: str | None = None) -> tbl.Table:
    """Load a ring layout by registered name or file path.

    Parameters
    ----------
    path : str or None
        A key of :data:`LAYOUTS`, a path to a layout file, or None for
        :data:`DEFAULT_LAYOUT`.  Files are array_arrangement CSVs
        (``tel_id,subarray,ring_n,az_n,az_deg,...``; compass azimuth is
        ``az_deg + 90``, as array_arrangement projects it).

    Returns
    -------
    astropy.table.Table
        Columns ``tel_id``, ``subarray_n``, ``ring_n``, ``az_n``,
        ``zenith_dist``, ``az``, ``rot_deg``; ``meta`` carries ``layout``,
        ``source`` and ``fov_deg``.

    """
    if path is None:
        path = DEFAULT_LAYOUT
    spec = LAYOUTS.get(path)
    if spec is not None:
        source = str(_LAYOUT_DIR.joinpath(spec.filename))
    else:
        source = str(path)

    with open(source) as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]

    if not lines[0].startswith("tel_id"):
        raise ValueError(f"{source}: not a layout CSV (expected a tel_id,... header row)")
    data = np.genfromtxt(lines, delimiter=",", names=True)
    tel_id = data["tel_id"].astype(int)
    subarray = data["subarray"].astype(int)
    ring_n = data["ring_n"].astype(int)
    az_n = data["az_n"].astype(int)
    zen = data["zen_deg"].astype(float)
    az_file = data["az_deg"].astype(float)
    rot = np.zeros(len(zen))
    az_offset = 90.0 if spec is None else spec.az_offset_deg

    table = tbl.Table(
        {
            "tel_id": tel_id,
            "subarray_n": subarray,
            "ring_n": ring_n,
            "az_n": az_n,
            "zenith_dist": zen,
            "az": (az_file + az_offset) % 360.0,
            "rot_deg": rot,
        }
    )
    table.meta["layout"] = spec.name if spec is not None else source
    table.meta["source"] = source
    table.meta["fov_deg"] = spec.fov_deg if spec is not None else None
    table.meta["band_table"] = (
        str(_LAYOUT_DIR.joinpath(spec.band_table)) if spec is not None and spec.band_table else None
    )
    return table


def subarray_members(cam_table: tbl.Table) -> dict[int, np.ndarray]:
    """Row indices of each subarray, keyed by subarray number.

    A subarray is one mount: its OTAs share a tracking axis and ratchet phase.
    """
    sub = np.asarray(cam_table["subarray_n"])
    return {int(s): np.flatnonzero(sub == s) for s in np.unique(sub)}


def ring_pointing(
    layout_table: tbl.Table,
    long_axis_extent: u.Quantity[u.deg],
    short_axis_extent: u.Quantity[u.deg],
    moc_depth: int,
) -> tuple[MOC, u.Quantity[u.deg**2], tbl.Table, tbl.Table]:
    """Build pointing tables from a ring layout.

    Uses the same intermediate observer/time coordinate system as
    ``grid_pointing`` so that the alt/az → ra/dec mapping is consistent.

    Parameters
    ----------
    layout_table : astropy.table.Table
        Output of :func:`load_ring_layout`.
    long_axis_extent : u.Quantity[u.deg]
        Camera long-axis FoV.
    short_axis_extent : u.Quantity[u.deg]
        Camera short-axis FoV.
    moc_depth : int
        HEALPix depth for MOC construction.

    Returns
    -------
    tuple
        ``(moc, area, cam_table, hpx_table)``, the same 4-tuple as
        ``grid_pointing``.

    """
    alt = 90.0 - np.array(layout_table["zenith_dist"])
    az = np.array(layout_table["az"])

    observer = crds.EarthLocation(
        lat=c.observatory.latitude * u.deg,
        lon=c.observatory.longitude * u.deg,
        height=c.observatory.altitude * u.m,
    )
    t = atime.Time(
        "2024-02-02T17:11:04.9401096100",
        scale="utc",
        location=observer,
    )

    altaz = crds.SkyCoord(
        az * u.deg,
        alt * u.deg,
        frame="altaz",
        obstime=t,
        location=observer,
    )
    radec = altaz.transform_to(crds.CIRS())
    cartesian = altaz.represent_as("cartesian")

    cam_pointings = tbl.Table(
        {
            "x_center": cartesian.x.value,
            "y_center": cartesian.y.value,
            "z_center": cartesian.z.value,
            "alt_center": alt,
            "az_center": az,
            "ra_center": radec.ra.deg,
            "dec_center": radec.dec.deg,
            "subarray_n": np.array(layout_table["subarray_n"]),
            "rot_deg": np.array(layout_table["rot_deg"]),
        }
    )
    for col in ("tel_id", "ring_n", "az_n"):
        if col in layout_table.colnames:
            cam_pointings[col] = np.array(layout_table[col])
    cam_pointings.meta.update(layout_table.meta)

    # The layout's long axis runs along the ring (+az); mocpy's ``angle`` is the
    # position angle of the semi-major axis east of celestial north, so in this
    # RA/Dec frame each box needs the position angle of its own +az direction
    # (the parallactic angle varies across the cradle) plus the camera rotation.
    d = 1e-3
    shifted = crds.SkyCoord(
        (az + d / np.cos(np.radians(alt))) * u.deg, alt * u.deg, frame="altaz", obstime=t, location=observer
    ).transform_to(crds.CIRS())
    pa_az = radec.position_angle(shifted).deg
    box_angle = (pa_az + np.array(layout_table["rot_deg"])) % 180.0
    cam_pointings["box_angle_deg"] = box_angle
    boxes = [
        MOC.from_boxes(
            lon=[ra] * u.deg,
            lat=[dec] * u.deg,
            a=long_axis_extent / 2,
            b=short_axis_extent / 2,
            angle=ang * u.deg,
            max_depth=moc_depth,
        )[0]
        for ra, dec, ang in zip(cam_pointings["ra_center"], cam_pointings["dec_center"], box_angle)
    ]
    fov = sum(boxes)
    fov_area = ((fov.sky_fraction * 4 * np.pi) * (180 / np.pi) ** 2) * u.deg**2

    hpx = ahpx.HEALPix(order="nested", nside=2**moc_depth, frame="cirs")
    longitude, latitude = hpx.healpix_to_lonlat(fov.flatten().astype(int))
    coords = crds.SkyCoord(ra=longitude, dec=latitude, frame="cirs")
    coords = coords.transform_to(crds.AltAz(obstime=t, location=observer))

    hpx_table = tbl.Table({"alt": coords.alt.deg, "az": coords.az.deg, "hpx": fov.flatten().astype(int)})

    n_cam = len(cam_pointings)
    log.info(f"Ring layout: {n_cam} cameras, {fov_area:.0f} coverage.")

    return fov, fov_area, cam_pointings, hpx_table


def assign_filters(
    cam_table: tbl.Table,
    options: list[str],
    cell_size: float,
) -> tbl.Table:
    """Add a ``filter`` column to the camera table.

    Assigns filters in repeating N-S strips across the focal plane so
    that a star drifting in RA crosses alternating filters.  Each
    camera's E-W coordinate (``zd * sin(az)``) is divided into strips
    of width ``cell_size``; the ``options`` sequence cycles across
    strips.

    Parameters
    ----------
    cam_table : astropy.table.Table
        Camera pointings table with ``alt_center``, ``az_center``, and
        ``subarray_n`` columns.
    options : list[str]
        Filter names to cycle through across strips.
    cell_size : float
        Strip width in degrees (typically the camera short-axis extent).

    Returns
    -------
    astropy.table.Table
        The input table with a ``filter`` column added.

    """
    n = len(options)
    zd = 90.0 - np.array(cam_table["alt_center"])
    az_rad = np.radians(np.array(cam_table["az_center"]))
    sx = 1.0 / cell_size
    x = zd * np.sin(az_rad)

    col_int = (x * sx).astype(int)
    cam_table["filter"] = [options[c % n] for c in col_int]
    return cam_table


def load_band_table(path: str) -> dict[int, str]:
    """Read a ``tel_id,band`` table; band is ``b`` (the cycle's first band) or ``rho`` (the other)."""
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if lines[0] != "tel_id,band":
        raise ValueError(f"{path}: expected header 'tel_id,band', got {lines[0]!r}")
    out = {}
    for line in lines[1:]:
        tel, band = line.split(",")
        if band not in ("b", "rho"):
            raise ValueError(f"{path}: unknown band {band!r} for tel_id {tel}")
        out[int(tel)] = band
    return out


def table_applies(options: list[str]) -> bool:
    """Return whether a two-class band table applies: it encodes a 1:1 split, so only a two-band equal cycle."""
    names = list(dict.fromkeys(options))
    return len(names) == 2 and options.count(names[0]) == options.count(names[1])


def assign_filters_from_table(cam_table: tbl.Table, options: list[str], table_path: str) -> tbl.Table:
    """Add a ``filter`` column from a per-OTA band table.

    Class ``b`` gets the cycle's first band (``options[0]``) and class ``rho`` the other band.  Every
    ``tel_id`` in ``cam_table`` must be in the table.
    """
    if not table_applies(options):
        raise ValueError(f"band table needs a two-band, equal-share cycle; got {options}")
    first = options[0]
    other = [o for o in dict.fromkeys(options) if o != first][0]
    bands = load_band_table(table_path)
    missing = [int(t) for t in cam_table["tel_id"] if int(t) not in bands]
    if missing:
        raise ValueError(f"{table_path}: {len(missing)} tel_ids missing, e.g. {missing[:5]}")
    cam_table["filter"] = [first if bands[int(t)] == "b" else other for t in cam_table["tel_id"]]
    return cam_table
