"""Helper functions for grid pointing geometry."""

from __future__ import annotations

import astropy.coordinates as crds
import astropy.table as tbl
import astropy.time as atime
import astropy.units as u
import astropy_healpix as ahpx
import numpy as np
import quaternion as quat
from mocpy import MOC
from numpy.typing import ArrayLike

from argus_sim import c, get_logger

log = get_logger(__name__)


def rotate(c: ArrayLike, axis: ArrayLike, theta: float) -> ArrayLike:
    """Rotate a Cartesian 3-vector about an arbitrary axis by a given angle using quaternions.

    Parameters
    ----------
    c : ArrayLike
        The Cartesian 3-vector to be rotated.
    axis : ArrayLike
        The axis about which to rotate the vector.
    theta : float
        The angle by which to rotate the vector, in radians.

    Returns
    -------
    ArrayLike
        The rotated Cartesian 3-vector.

    """
    v = np.concatenate(([0.0], c))
    rot_axis = np.concatenate(([0.0], axis))

    axis_angle = (theta / 2) * rot_axis / np.linalg.norm(rot_axis)
    vec = quat.quaternion(*v)
    qlog = quat.quaternion(*axis_angle)

    q = np.exp(qlog)

    v_prime = q * vec * np.conjugate(q)

    return v_prime.imag


def xy_to_z(
    x: u.Quantity | float = 1,
    y: u.Quantity | float = 1,
    rad: u.Quantity | float = 1,
    *,
    use_mask: ArrayLike | None = None,
) -> tuple:
    """Convert 2D Cartesian coordinates to 3D coordinates on the surface of a sphere of a given radius.

    Parameters
    ----------
    x : u.Quantity or float, optional
        The x-coordinates. Default is 1.
    y : u.Quantity or float, optional
        The y-coordinates. Default is 1.
    rad : u.Quantity or float, optional
        The radius within which to convert coordinates. Default is 1.
    use_mask : ArrayLike or None, optional
        Pre-computed mask array. Default is None.

    Returns
    -------
    tuple
        A tuple containing the x, y, and z coordinates, and optionally the mask.

    """
    x, y = np.meshgrid(x, y)
    mask = x**2 + y**2 <= rad**2 if use_mask is None else use_mask

    eta = rad**2 - x**2 - y**2
    eta = eta[mask]
    x = x[mask]
    y = y[mask]

    eta = np.abs(eta)
    z = -1 * np.sqrt(eta)
    if use_mask is not None:
        return x.flatten(), y.flatten(), z.flatten()
    return x.flatten(), y.flatten(), z.flatten(), mask


def grid_pointing(  # noqa: PLR0913, PLR0915
    long_axis_extent: u.Quantity[u.deg],
    short_axis_extent: u.Quantity[u.deg],
    short_axis_overlap: u.Quantity[u.deg],
    long_axis_overlap: u.Quantity[u.deg],
    *,
    contract_lon: bool = False,
    n_telescopes: int = 900,
    min_alt: float = 30,
) -> tuple[MOC, u.Quantity[u.deg**2], tbl.Table, tbl.Table]:
    """Generate a grid of pointings for a given single-camera field of view and overlap parameters.

    Parameters
    ----------
    long_axis_extent : u.Quantity[u.deg]
        The extent of the field of view along the long axis.
    short_axis_extent : u.Quantity[u.deg]
        The extent of the field of view along the short axis.
    short_axis_overlap : u.Quantity[u.deg]
        The overlap between adjacent pointings along the short axis.
    long_axis_overlap : u.Quantity[u.deg ]
        The overlap between adjacent pointings along the long axis.
    contract_lon : bool, optional
        Whether to contract the longitude as a function of latitude. Default is False.
    n_telescopes : int, optional
        The number of telescopes. Default is 900.
    min_alt : float, optional
        The minimum altitude. Default is 30.

    Returns
    -------
    tuple
        A tuple containing the field of view (MOC), the field of view area, and
        a tables of pointings per camera and re-gridded into HEALPix.

    """
    long_diameter = (long_axis_extent).to(u.radian)
    short_diameter = (short_axis_extent).to(u.radian)

    short_axis_spacing = short_diameter - short_axis_overlap
    long_axis_spacing = long_diameter - long_axis_overlap

    short_axis_offset = short_axis_spacing / 2
    long_axis_offset = long_axis_spacing / 2

    # Start in a coordinate system where x is parallel to the short axis of the sensor and y to the long axis.
    # The meridian runs down the centre of the array.

    zshort_vec = np.array([1, 0, 0])
    zlong_vec = np.array([0, 1, 0])

    zenith_vec = np.array([0, 0, 1])
    zenith_upcen = rotate(
        zenith_vec,
        zshort_vec,
        (long_axis_extent / 2).to(u.radian).value,
    )
    zenith_upleft = rotate(
        zenith_upcen,
        zlong_vec,
        -1 * (short_axis_extent / 2).to(u.radian).value,
    )
    zenith_upright = rotate(
        zenith_upcen,
        zlong_vec,
        1 * (short_axis_extent / 2).to(u.radian).value,
    )
    zenith_bottomcen = rotate(
        zenith_vec,
        zshort_vec,
        -1 * (long_axis_extent / 2).to(u.radian).value,
    )
    zenith_bottomleft = rotate(
        zenith_bottomcen,
        zlong_vec,
        -1 * (short_axis_extent / 2).to(u.radian).value,
    )
    zenith_bottomright = rotate(
        zenith_bottomcen,
        zlong_vec,
        1 * (short_axis_extent / 2).to(u.radian).value,
    )

    # These are not the RA/Dec of the pointing, only an intermediate coordinate system for the grid.
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

    # 8 points around the edge of the chip, plus center of chip 'inside'
    zenith_polygon = [
        zenith_vec,
        # zenith_upcen,
        zenith_upleft,
        # zenith_leftcen,
        zenith_bottomleft,
        # zenith_bottomcen,
        zenith_bottomright,
        # zenith_rightcen,
        zenith_upright,
        # zenith_upcen,
        zenith_upleft,
    ]

    x_polys = []
    y_polys = []
    z_polys = []

    alt_polys = []
    az_polys = []
    ra_polys = []
    dec_polys = []

    if c.structure.n_long_axis is None:
        n_long = np.round(((180 * u.deg / long_diameter.to(u.deg)) / 2).value).astype(int)
    else:
        n_long = c.structure.n_long_axis

    if c.structure.n_short_axis is None:
        n_short = np.round(((180 * u.deg / short_diameter.to(u.deg)) / 2).value).astype(int)
    else:
        n_short = c.structure.n_short_axis
    for j, zenith in enumerate(zenith_polygon):
        # For the chip-edge points the meridian is an offset spherical frame; the true meridian is the zero point.
        meridian = np.array(
            [
                rotate(
                    -1 * zenith,
                    zshort_vec,
                    (i * long_axis_spacing.to(u.radian).value) + long_axis_offset.to(u.radian).value,
                )
                for i in range(-n_long, n_long)
            ],
        )
        true_meridian = np.array(
            [
                rotate(
                    -1 * zenith_vec,
                    zshort_vec,
                    (i * long_axis_spacing.to(u.radian).value) + long_axis_offset.to(u.radian).value,
                )
                for i in range(-n_long, n_long)
            ],
        )

        if contract_lon:
            pointings = []
            for k, true_row in enumerate(true_meridian):
                colat = (
                    (
                        np.rad2deg(np.arctan2(true_row[1], true_row[2]))
                        - np.rad2deg(np.arctan2(zenith_vec[1], zenith_vec[2]))
                    )
                    % 360
                ) - 180
                colat = np.deg2rad(colat)
                row = meridian[k]

                row_vals = np.array(
                    [
                        rotate(
                            row,
                            zlong_vec,
                            ((i * short_axis_spacing.to(u.radian).value) + short_axis_offset.to(u.radian).value)
                            / np.cos(colat),
                        )
                        for i in range(-60, 60)
                    ],
                )
                pointings.append(row_vals)

            pointings = np.array(pointings).reshape(-1, 3)
            xp, yp, zp = pointings.T

        else:
            equator = np.array(
                [
                    rotate(
                        -1 * zenith,
                        zlong_vec,
                        (i * short_axis_spacing.to(u.radian).value) + short_axis_offset.to(u.radian).value,
                    )
                    for i in range(-n_short, n_short)
                ],
            )

            xe, ye, ze = equator.T
            xm, ym, zm = meridian.T
            if j == 0:
                xp, yp, zp, mask = xy_to_z(xe, ym)
            else:
                xp, yp, zp = xy_to_z(xe, ym, use_mask=mask)

        altaz = crds.AltAz(
            yp,
            xp,
            -zp,
            representation_type="cartesian",
            obstime=t,
            location=observer,
        )
        radec = altaz.transform_to(crds.CIRS())

        altaz = altaz.represent_as("spherical")
        radec = radec.represent_as("spherical")

        alt, az = altaz.lat.deg, altaz.lon.deg
        ra, dec = radec.lon.deg, radec.lat.deg

        x_polys.append(xp)
        y_polys.append(yp)
        z_polys.append(zp)

        alt_polys.append(alt)
        az_polys.append(az)

        ra_polys.append(ra)
        dec_polys.append(dec)

    x_polys = np.array(x_polys)
    y_polys = np.array(y_polys)
    z_polys = np.array(z_polys)
    alt_polys = np.array(alt_polys)
    az_polys = np.array(az_polys)
    ra_polys = np.array(ra_polys)
    dec_polys = np.array(dec_polys)

    x = x_polys[0, :]
    y = y_polys[0, :]
    z = z_polys[0, :]

    alt = alt_polys[0, :]
    az = az_polys[0, :]

    ra = ra_polys[0, :]
    dec = dec_polys[0, :]

    x_polys = x_polys[1:, :]
    y_polys = y_polys[1:, :]
    z_polys = z_polys[1:, :]

    alt_polys = alt_polys[1:, :]
    az_polys = az_polys[1:, :]

    ra_polys = ra_polys[1:, :]
    dec_polys = dec_polys[1:, :]

    ot = tbl.Table(
        {
            "x_center": x,
            "y_center": y,
            "z_center": z,
            "alt_center": alt,
            "az_center": az,
            "ra_center": ra,
            "dec_center": dec,
            "x_polygon": x_polys.T,
            "y_polygon": y_polys.T,
            "z_polygon": z_polys.T,
            "alt_polygon": alt_polys.T,
            "az_polygon": az_polys.T,
        },
    )
    n = 0
    while len(ot) > n_telescopes:
        ot = ot[ot["alt_center"] > min_alt + n * 0.05]
        n += 1
    log.info(f"Reduced to {len(ot)} pointings with max AOI {90 - (30 + n * 0.05)}.")

    boxes = MOC.from_boxes(
        lon=ot["ra_center"] * u.deg,
        lat=ot["dec_center"] * u.deg,
        a=long_axis_extent / 2,
        b=short_axis_extent / 2,
        angle=0 * u.deg,
        max_depth=c.survey.sim_depth,
    )
    fov = sum(boxes)
    fov_area = ((fov.sky_fraction * 4 * np.pi) * (180 / np.pi) ** 2) * u.deg**2

    hpx = ahpx.HEALPix(order="nested", nside=2 ** (c.survey.sim_depth), frame="cirs")

    longitude, latitude = hpx.healpix_to_lonlat(fov.flatten().astype(int))

    coords = crds.SkyCoord(
        ra=longitude,
        dec=latitude,
        frame="cirs",
    )

    coords = coords.transform_to(crds.AltAz(obstime=t, location=observer))

    sim_table = tbl.Table(
        {"alt": coords.alt.deg, "az": coords.az.deg, "hpx": fov.flatten().astype(int)},
    )

    return fov, fov_area, ot, sim_table
