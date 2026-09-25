"""Configuration dataclasses for telescope, camera, site, and survey parameters."""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import astropy.units as u
import tosholi
from dacite.exceptions import MissingValueError

_RAD_TO_ARCSEC = 180 / math.pi * 3600

_PACKAGE_DATA = Path(__file__).resolve().parent / "data"
# Delivered optics PSF: optics-only images at 7 field angles for a blue (g) and a red (rho) band, sampled at half a
# detector pixel.  Their EE50 diameters, before seeing and after pixelisation, lie between 2.0 and 3.0 px across the
# field, with an area mean of 2.3 px over the sensor.
DEFAULT_PSF_FITS = str(_PACKAGE_DATA / "delivered_psf.fits")


class ConfigException(Exception):
    """Raised for an invalid configuration."""

    pass


@dataclasses.dataclass
class Structure:
    """Instrument structure parameters."""

    cradle_diameter: Optional[float] = None  # meters or None; informational, not read by the simulation
    min_cradle_diameter: float = 2  # meters
    max_cradle_diameter: float = 10  # meters
    telescope_min_spacing: float = 0.24  # meters

    n_telescopes: int = 1200
    # baseline config gives up to 36-across at equator
    n_short_axis: Optional[int] = None
    n_long_axis: Optional[int] = None


@dataclasses.dataclass
class Observatory:
    """Observatory and site parameters."""

    min_alt: float = 32.84  # degrees
    # Site: a representative dark site in the southwestern US.
    latitude: float = 31.0  # degrees
    longitude: float = -104.0  # degrees
    altitude: float = 2000  # meters

    timezone: str = "America/Chicago"

    artificial_light_cd_m2: float = 1.94  # microcandela per square meter
    # Unused while seasonal_seeing is true (the default): the seeing then comes per epoch from
    # observatory.site_seeing.  They are the fallback single-season lognormal(ln seeing_mean, seeing_std).
    seeing_mean: float = 0.87  # arcseconds
    seeing_std: float = 0.15  # arcseconds
    # Seasonal seeing on by default: the DIMM seasons of Barker et al. (2003, 2003SPIE.4837..225B), winter apart from
    # the rest (see observatory.site_seeing).
    seasonal_seeing: bool = True
    solar_activity: float = 0.8  # dimensionless, 0.8 (solar min) to 2.0 (solar max)
    # Photometric (cloud-free) grey transparency, 1.0.  Not what Survey or ForwardMC draw: their per-ratchet
    # transparency is the cloud/cirrus model in forward_mc.DEFAULT_RATCHET_PRIORS.
    transparency: float = 1.0


@dataclasses.dataclass
class Uptime:
    """Estimates of uptime and weather parameters."""

    weather_model: Literal["random"] = "random"
    engineering_uptime_prob: float = 0.98
    slew_len: float = 0.75  # minutes
    cam_fail_rate: float = 0.0
    cam_dead_time: float = 0.01  # seconds


@dataclasses.dataclass
class TelescopeUnit:
    """Specifications of the telescope and camera unit."""

    aperture_diameter: float = 280  # mm
    # Effective collecting area of one telescope.
    collecting_area_cm2: float = 394.08138246630364  # cm^2

    # RMS spot radius in microns, used as a Gaussian sigma by GaussianPSF (psf_model = "gaussian").
    # 3.2 µm is a field-averaged Gaussian width consistent with the delivered image quality.
    spot_size: float = 3.2  # microns

    focal_length: float = 770.0  # mm
    pixel_size: float = 3.76  # microns

    n_pixels_x: int = 11648  # pixels (long axis)
    n_pixels_y: int = 8742  # pixels (short axis)

    readnoise: float = 1.2  # electrons
    dark_current: float = 0.0022  # electrons/s/pixel

    fullwell: int = 21657  # electrons

    readnoise_variability: float = 0.10  # fractional camera-to-camera readnoise variation
    dark_current_variability: float = 0.10  # fractional camera-to-camera dark current variation

    throughput_loss: float = 1.0  # multiplicative degradation factor (0-1)

    psf_model: str = "delivered"  # "delivered" (tabulated delivered optics, the default) or "gaussian"
    psf_fits_path: Optional[str] = DEFAULT_PSF_FITS
    # Optional CSV (band, theta_deg, target_um) of optics-only EE50 diameters per field angle.  Where a target is
    # larger than the tabulated PSF's EE50, DeliveredPSF broadens the PSF to meet it; use it to ask how depth
    # changes with worse optics.
    ee50_profile_path: Optional[str] = None
    randomize_pixel_phase: bool = False

    @property
    def plate_scale(self) -> float:
        """Arcsec per pixel."""
        return (self.pixel_size / 1000) / self.focal_length * _RAD_TO_ARCSEC

    @property
    def long_axis_extent(self) -> float:
        """Degrees."""
        return self.n_pixels_x * self.plate_scale / 3600

    @property
    def short_axis_extent(self) -> float:
        """Degrees."""
        return self.n_pixels_y * self.plate_scale / 3600

    @property
    def collecting_area(self) -> u.Quantity:
        """Single-telescope collecting area in cm^2."""
        return self.collecting_area_cm2 * u.cm**2


@dataclasses.dataclass
class PackingStrategy:
    """Telescope packing strategy parameters."""

    tiling: str = "ring"

    # Grid tiling parameters
    short_axis_overlap: float = 0.257
    long_axis_overlap: float = 0.257

    # Ring tiling parameters
    layout_file: Optional[str] = None
    max_aoi: float = 53.2
    n_subarray_radial: int = 3
    n_subarray_azimuthal: int = 2


@dataclasses.dataclass
class FilterStrategy:
    """Filter strategy parameters."""

    # beta (named "g" in the code) and rho at 1:1
    options: List = dataclasses.field(default_factory=lambda: ["g", "rho", "g", "rho"])
    band: str = "g"
    cell_size: Optional[float] = None


@dataclasses.dataclass
class Output:
    """Simulation output parameters."""

    save_grid_tables: bool = True
    output_dir: str = "output"

    metadata: bool = True

    cadence_levels: List[str] = dataclasses.field(default_factory=lambda: ["hour", "night", "season"])


@dataclasses.dataclass
class Survey:
    """Survey parameters."""

    base_cadence_s: float = 60.0
    # minutes; the hardware ratchet: 15 exposures of base_cadence_s followed by a reset_min reset.
    # The sky step per block is 16 min of sidereal motion, 4.01 deg.
    ratchet_len: float = 16
    # Of each ratchet, the last reset_min minutes are the reset (no exposures); ratchets start on
    # ratchet_len boundaries of UTC (coverage_exact.Schedule).
    reset_min: float = 1.0
    # "exact": gnomonic OTA coverage accumulated on the HDPS tanseg/minipix grid (tanseg_coverage);
    # "moc": the HEALPix MOC / filter-map path, kept for comparison.
    coverage: str = "exact"
    # Cap on CuPy's memory pool as a fraction of device memory (exact engine).  At the cap the pool frees its
    # cached blocks and retries, leaving room for allocations outside the pool (thrust sorts, cuBLAS).
    gpu_pool_fraction: float = 0.7

    # HEALPix depth for the MOC footprint and simulation grid.
    # Both bands converge to <0.03 mag at sim_depth=9 (NSIDE=512,
    # ~47 arcmin² pixels).
    # Use sim_depth=5 for fast quick-look sims (~3s/night).
    sim_depth: int = 9

    # Stamp size for photometry calculations
    stamp_size: int = 28  # pixels
    # subpixel sampling for photometry calculations
    stamp_resolution: float = 0.01  # pixels

    detection_snr: float = 5.0

    # Observing limits
    min_sun_alt: float = -18  # deg
    min_moon_sep: float = 1.5  # deg
    # Solar activity per night: "epoch" = ForwardMC's 2028-2033 prior (the month's value inside 2028-2033, else one
    # draw per night from it); "fixed" = observatory.solar_activity.
    solar_activity_prior: str = "epoch"
    # Fast mode (1-s frames) from this lunar illumination up: ~3.6 nights per synodic month.
    highspeed_start_moonfrac: float = 0.96
    fast_cadence_s: float = 1.0
    skip_fast_cadence: bool = False

    start_date: datetime = datetime(2000, 1, 1)
    end_date: datetime = datetime(2001, 1, 1)
    # Reference magnitude for volume and distance calculations
    reference_magnitude: float = -15.7

    n_healpix: Optional[int] = None
    # The weather model gates usable nights by default: observatory.site_weather's clear fractions,
    # one Bernoulli draw per night.  `asim survey --no-weather` (or apply_weather = false) restores an every-night run.
    apply_weather: bool = True
    fraction_of_ratchets_to_run: float = 1.0


@dataclasses.dataclass
class Config:
    """Configuration of an Argus Array simulation run.

    Most defaults follow Galliher et al. (2022, https://arxiv.org/abs/2208.08794), updated for the current
    hardware.

    """

    sim_version: str
    run_name: str = "argussim"
    nproc: int = 4
    output: Output = dataclasses.field(default_factory=Output)
    structure: Structure = dataclasses.field(default_factory=Structure)
    observatory: Observatory = dataclasses.field(default_factory=Observatory)
    telescope: TelescopeUnit = dataclasses.field(default_factory=TelescopeUnit)
    packing_strategy: PackingStrategy = dataclasses.field(default_factory=PackingStrategy)
    filter_strategy: FilterStrategy = dataclasses.field(default_factory=FilterStrategy)
    survey: Survey = dataclasses.field(default_factory=Survey)
    uptime: Uptime = dataclasses.field(default_factory=Uptime)


def get_config(path: str | Path | None = None) -> Optional[Config]:
    """Load a Config from ``path``, or from the first TOML file in the working directory that parses as one.

    Parameters
    ----------
    path : str, Path, or None, optional
        The path to the TOML file. If None, every TOML file in the current working directory is tried.

    Returns
    -------
    Optional[Config]
        The loaded configuration object if a valid TOML file is found, otherwise None.

    """
    tomls = [path] if path is not None else list(Path.cwd().glob("*.toml"))

    conf_found = False
    for toml in tomls:
        with Path(toml).open("rb") as f:
            try:
                config = tosholi.load(Config, f)
            except MissingValueError:
                continue
            conf_found = True
            break
    if conf_found:
        return config
    return None


def write_config(config: Config) -> Path:
    """Write the configuration to a TOML file.

    Parameters
    ----------
    config : Config
        The configuration object to write.

    Returns
    -------
    Path
        The path to the written TOML file.

    """
    out_path = Path(f"{config.run_name.strip()}.toml")
    with out_path.open("wb") as f:
        tosholi.dump(config, f)
    return out_path
