"""Grid-based pointing model for telescope arrays."""

from __future__ import annotations

import astropy.coordinates as crds
import astropy.table as tbl
import astropy.time as atime
import astropy.units as u
import astropy_healpix as ahpx
import numpy as np
from scipy.spatial import distance

import argus_sim.observatory as obs
from argus_sim import c, get_logger

from .grid_helper import grid_pointing


class CradlePointing:
    """A class to handle cradle pointings for a telescope array.

    Parameters
    ----------
    long_axis_extent : u.Quantity[u.deg] | None, optional
        The extent of the long axis in degrees. Default is None.
    short_axis_extent : u.Quantity[u.deg] | None, optional
        The extent of the short axis in degrees. Default is None.
    short_axis_overlap : u.Quantity[u.deg] | None, optional
        The overlap of the short axis in degrees. Default is None.
    long_axis_overlap : u.Quantity[u.deg] | None, optional
        The overlap of the long axis in degrees. Default is None.
    contract_lon : bool, optional
        Whether to contract the longitude. Default is False.
    n_telescopes : int | None, optional
        The number of telescopes. Default is None.

    """

    def __init__(
        self,
        long_axis_extent: u.Quantity[u.deg] | None = None,
        short_axis_extent: u.Quantity[u.deg] | None = None,
        short_axis_overlap: u.Quantity[u.deg] | None = None,
        long_axis_overlap: u.Quantity[u.deg] | None = None,
        *,
        contract_lon: bool = False,
        n_telescopes: int | None = None,
        tiling: str | None = None,
    ) -> None:
        """Initialize the CradlePointing object.

        Parameters
        ----------
        long_axis_extent : u.Quantity[u.deg] | None, optional
            The extent of the long axis in degrees. Default is None.
        short_axis_extent : u.Quantity[u.deg] | None, optional
            The extent of the short axis in degrees. Default is None.
        short_axis_overlap : u.Quantity[u.deg] | None, optional
            The overlap of the short axis in degrees. Default is None.
        long_axis_overlap : u.Quantity[u.deg] | None, optional
            The overlap of the long axis in degrees. Default is None.
        contract_lon : bool, optional
            Whether to contract the longitude. Default is False.
        n_telescopes : int | None, optional
            The number of telescopes. Default is None.
        tiling : str or None, optional
            Tiling mode: ``"ring"`` or ``"grid"``. Default reads from config.

        """
        self.log = get_logger(__name__)
        self.observatory = obs.Observatory()
        self.hpx = ahpx.HEALPix(order="nested", nside=2**c.survey.sim_depth, frame="cirs")

        if long_axis_extent is None:
            self.long_axis_extent = c.telescope.long_axis_extent * u.deg
        else:
            self.long_axis_extent = long_axis_extent

        if short_axis_extent is None:
            self.short_axis_extent = c.telescope.short_axis_extent * u.deg
        else:
            self.short_axis_extent = short_axis_extent

        if short_axis_overlap is None:
            self.short_axis_overlap = c.packing_strategy.short_axis_overlap * u.deg
        else:
            self.short_axis_overlap = short_axis_overlap

        if long_axis_overlap is None:
            self.long_axis_overlap = c.packing_strategy.long_axis_overlap * u.deg
        else:
            self.long_axis_overlap = long_axis_overlap

        self.contract_lon = contract_lon
        if n_telescopes is None:
            self.n_telescopes = c.structure.n_telescopes
        else:
            self.n_telescopes = n_telescopes

        if tiling is None:
            tiling = c.packing_strategy.tiling
        self.tiling = tiling

        if tiling == "ring":
            from .ring_layout import (
                assign_filters,
                assign_filters_from_table,
                load_ring_layout,
                ring_pointing as _ring_pointing,
                table_applies,
            )

            layout_table = load_ring_layout(c.packing_strategy.layout_file)
            self.layout = layout_table.meta["layout"]
            # A layout designed with its own camera field (the canonical A170r_1200) builds
            # its footprints with it unless the caller passed extents explicitly.
            layout_fov = layout_table.meta.get("fov_deg")
            if layout_fov is not None:
                if long_axis_extent is None:
                    self.long_axis_extent = layout_fov[0] * u.deg
                if short_axis_extent is None:
                    self.short_axis_extent = layout_fov[1] * u.deg
            fov, area, cam_pointings, hpx_pointings = _ring_pointing(
                layout_table,
                self.long_axis_extent,
                self.short_axis_extent,
                c.survey.sim_depth,
            )
            # Strip width stays on the configured telescope field, not the layout's:
            # band assignment is unchanged by the choice of layout.
            cell_size = c.filter_strategy.cell_size
            if cell_size is None:
                cell_size = c.telescope.short_axis_extent
            band_table = layout_table.meta.get("band_table")
            if band_table and table_applies(c.filter_strategy.options):
                cam_pointings = assign_filters_from_table(cam_pointings, c.filter_strategy.options, band_table)
                self.band_assignment = f"table:{band_table.rsplit('/', 1)[-1]}"
            else:
                if band_table:
                    self.log.info(
                        f"Band table {band_table} is 1:1 only; cycle {c.filter_strategy.options} uses the strip rule."
                    )
                cam_pointings = assign_filters(cam_pointings, c.filter_strategy.options, cell_size)
                self.band_assignment = f"strips:{cell_size:.6f}"

            from .coverage_map import build_filter_maps

            self.filter_maps = build_filter_maps(
                cam_pointings,
                self.long_axis_extent,
                self.short_axis_extent,
                c.survey.sim_depth,
                2**c.survey.sim_depth,
            )
        elif tiling == "grid":
            fov, area, cam_pointings, hpx_pointings = grid_pointing(
                long_axis_extent=self.long_axis_extent,
                short_axis_extent=self.short_axis_extent,
                short_axis_overlap=self.short_axis_overlap,
                long_axis_overlap=self.long_axis_overlap,
                contract_lon=self.contract_lon,
                n_telescopes=self.n_telescopes,
            )
            self.filter_maps = None
            self.layout = "grid"
            self.band_assignment = None
        else:
            raise ValueError(f"Unknown tiling mode: {tiling!r}")

        self.approximate_area = area
        self.log.info(
            f"Layout {self.layout}: {len(cam_pointings)} cameras, footprint {self.long_axis_extent:.6f} x {self.short_axis_extent:.6f}."
        )

        self.moc = fov
        self.cam_pointings = cam_pointings
        self.hpx_pointings = hpx_pointings

    def to_hpx_table_at(self, time: atime.Time) -> tbl.Table:
        """Convert the grid pointings to a table at a specific time.

        Parameters
        ----------
        time : atime.Time
            The observation time.

        Returns
        -------
        tbl.Table
            The table representation of the grid pointings at the specified
            time. The table includes columns for RA, Dec, Alt, Az, and healpix.
            With filter maps, a pixel covered by more than one filter has one
            row per covering filter.

        """
        if self.filter_maps:
            return self._hpx_table_from_filter_maps(time)

        altaz_pointings = crds.SkyCoord(
            self.hpx_pointings["az"] * u.deg,
            self.hpx_pointings["alt"] * u.deg,
            frame="altaz",
            obstime=time,
            location=self.observatory.el,
        )
        radecs = altaz_pointings.transform_to(crds.CIRS)
        h = self.hpx.lonlat_to_healpix(radecs.ra, radecs.dec)
        out = tbl.Table(
            [
                tbl.Column(name="ra", data=radecs.ra.deg),
                tbl.Column(name="dec", data=radecs.dec.deg),
                tbl.Column(name="alt", data=altaz_pointings.alt.deg),
                tbl.Column(name="az", data=altaz_pointings.az.deg),
                tbl.Column(name="healpix", data=h),
                tbl.Column(
                    name="nside",
                    data=[
                        self.hpx.nside,
                    ]
                    * len(h),
                ),
            ],
        )

        return out.group_by("healpix").groups.aggregate(np.mean)

    def _hpx_table_from_filter_maps(self, time: atime.Time) -> tbl.Table:
        """Sample the ratchet-time CIRS grid directly against the alt/az filter maps.

        Every HEALPix cell that can reach the footprint (declination within the
        cradle's zenith-distance range of the site latitude) has its centre
        transformed to alt/az at ``time`` and tested against the per-filter
        camera maps; covered cells get one row per covering filter.  Moving
        the reference-epoch cell centres and re-binning them instead would
        alias 0-9% of the footprint away, depending on the grid phase.
        """
        from .coverage_map import rows_per_filter

        if getattr(self, "_grid_cells", None) is None:
            ipix = np.arange(self.hpx.npix)
            lon, lat = self.hpx.healpix_to_lonlat(ipix)
            zd_max = (
                90.0
                - float(np.min(self.cam_pointings["alt_center"]))
                + float(max(self.long_axis_extent, self.short_axis_extent).to_value(u.deg))
            )
            lat_site = self.observatory.el.lat.deg
            keep = np.abs(lat.deg - lat_site) <= zd_max
            self._grid_cells = (ipix[keep], lon[keep], lat[keep])
        ipix, lon, lat = self._grid_cells
        altaz = crds.SkyCoord(ra=lon, dec=lat, frame="cirs", obstime=time).transform_to(
            crds.AltAz(obstime=time, location=self.observatory.el)
        )
        az, alt = altaz.az.deg, altaz.alt.deg
        covered = np.zeros(len(ipix), dtype=bool)
        for filt in sorted(self.filter_maps):
            covered |= self.filter_maps[filt].get_values_pos(az, alt, lonlat=True) > 0
        out = tbl.Table(
            [
                tbl.Column(name="ra", data=lon.deg[covered]),
                tbl.Column(name="dec", data=lat.deg[covered]),
                tbl.Column(name="alt", data=alt[covered]),
                tbl.Column(name="az", data=az[covered]),
                tbl.Column(name="healpix", data=ipix[covered]),
                tbl.Column(name="nside", data=np.full(int(covered.sum()), self.hpx.nside)),
            ]
        )
        return rows_per_filter(out, self.filter_maps)

    @property
    def subarrays(self) -> dict[int, np.ndarray]:
        """Row indices into ``cam_pointings`` of each subarray (one mount each)."""
        from .ring_layout import subarray_members

        return subarray_members(self.cam_pointings)

    @property
    def collecting_area(self):
        """Total collecting area of the telescope array.

        Returns
        -------
        u.Quantity
            The total collecting area of the telescope array in square millimeters.

        """
        return (len(self.cam_pointings) * c.telescope.collecting_area).to("m^2")

    @property
    def equivalent_diameter(self):
        """Effective diameter of the telescope array.

        Returns
        -------
        u.Quantity
            The effective diameter of the telescope array in millimeters.

        """
        effective_diameter = 2 * np.sqrt(self.collecting_area / np.pi)
        return effective_diameter.to("m")

    @property
    def fov(self):
        """Combined field of view of the telescope array.

        Returns
        -------
        u.Quantity
            The field of view of the telescope array in square degrees.

        """
        return self.approximate_area

    def to_telescope_pointings(self, scale_to_separation: bool = False) -> tbl.Table:
        """Convert the camera pointings to telescope pointings.

        Parameters
        ----------
        scale_to_separation : bool, optional
            Whether to scale the pointings to the telescope separation. Default is False.

        Returns
        -------
        tbl.Table
            The table representation of the telescope pointings.

        Examples
        --------
        Converting to telescope pointings:

        .. code:: python

            >>> cradle_pointing = CradlePointing()
            >>> telescope_pointings = cradle_pointing.to_telescope_pointings(scale_to_separation=True)
            >>> print(telescope_pointings)

        """
        if scale_to_separation:
            radii = np.arange(c.structure.min_cradle_diameter, c.structure.max_cradle_diameter, 0.001)
            for rad in radii:
                coords = rad * np.array(
                    list(
                        zip(
                            self.cam_pointings["x_center"],
                            self.cam_pointings["y_center"],
                            self.cam_pointings["z_center"],
                        ),
                    )
                )
                dists = distance.cdist(coords, coords, "euclidean")
                dists = dists[dists > 0]
                if dists.min() >= c.structure.telescope_min_spacing:
                    self.log.info(f"Found minimum cradle radius: {rad: 0.2f} meters.")
                    self.log.info(f"Cradle extent is {dists.max(): 0.2f} meters.")
                    break
            self.cam_pointings["x_center_meters"] = self.cam_pointings["x_center"] * rad
            self.cam_pointings["y_center_meters"] = self.cam_pointings["y_center"] * rad
            self.cam_pointings["z_center_meters"] = self.cam_pointings["z_center"] * rad

        out_table = self.cam_pointings.copy()
        out_table.remove_columns([col for col in out_table.colnames if "poly" in col])
        out_table.remove_columns([col for col in out_table.colnames if "ra_" in col])
        out_table.remove_columns([col for col in out_table.colnames if "dec_" in col])
        return out_table
