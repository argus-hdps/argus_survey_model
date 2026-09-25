import os
from datetime import datetime

import astropy.units as u
import pytest

from argus_sim.config import (
    Config,
    FilterStrategy,
    Observatory,
    Output,
    PackingStrategy,
    Structure,
    Survey,
    TelescopeUnit,
    Uptime,
    get_config,
    write_config,
)


def test_structure_defaults():
    """Test default values for Structure."""
    structure = Structure()
    assert structure.cradle_diameter is None
    assert structure.n_telescopes == 1200


def test_observatory_defaults():
    """Test default values for Observatory."""
    observatory = Observatory()
    assert observatory.min_alt == 32.84
    assert observatory.latitude == 31.0
    assert observatory.longitude == -104.0
    assert observatory.altitude == 2000
    assert observatory.timezone == "America/Chicago"
    assert observatory.artificial_light_cd_m2 == 1.94
    assert observatory.seeing_mean == 0.87
    assert observatory.seasonal_seeing is True  # the DIMM seasons of observatory.site_seeing


def test_uptime_defaults():
    """Test default values for Uptime."""
    uptime = Uptime()
    assert uptime.weather_model == "random"
    assert uptime.engineering_uptime_prob == 0.98
    assert uptime.slew_len == 0.75
    assert uptime.cam_fail_rate == 0.0
    assert uptime.cam_dead_time == 0.01


def test_telescope_unit_defaults():
    """Test default values for TelescopeUnit."""
    telescope = TelescopeUnit()
    assert telescope.focal_length == 770.0
    assert telescope.plate_scale == pytest.approx(1.007215157778029)
    assert telescope.pixel_size == 3.76
    assert telescope.n_pixels_x == 11648
    assert telescope.n_pixels_y == 8742
    assert telescope.aperture_diameter == 280
    assert telescope.spot_size == 3.2
    assert telescope.readnoise == 1.2
    assert telescope.fullwell == 21657
    assert telescope.psf_model == "delivered"
    assert telescope.ee50_profile_path is None
    assert telescope.long_axis_extent == pytest.approx(telescope.n_pixels_x * telescope.plate_scale / 3600)
    assert telescope.short_axis_extent == pytest.approx(telescope.n_pixels_y * telescope.plate_scale / 3600)


def test_telescope_collecting_area():
    """Collecting area is the configured area in cm^2, 394.08 cm^2 by default."""
    telescope = TelescopeUnit()
    result = telescope.collecting_area
    assert result.unit == u.cm**2
    assert result.value == pytest.approx(394.08138, rel=1e-6)


def test_telescope_collecting_area_custom():
    """Collecting area follows a non-default setting."""
    telescope = TelescopeUnit(collecting_area_cm2=250.0)
    assert telescope.collecting_area.value == pytest.approx(250.0, rel=1e-10)


def test_packing_strategy_defaults():
    """Test default values for PackingStrategy."""
    packing_strategy = PackingStrategy()
    assert packing_strategy.tiling == "ring"
    assert packing_strategy.short_axis_overlap == 0.257
    assert packing_strategy.long_axis_overlap == 0.257


def test_filter_strategy_defaults():
    """Test default values for FilterStrategy."""
    filter_strategy = FilterStrategy()
    assert filter_strategy.options == ["g", "rho", "g", "rho"]
    assert filter_strategy.band == "g"
    assert filter_strategy.cell_size is None


def test_survey_defaults():
    """Test default values for Survey."""
    survey = Survey()
    assert survey.base_cadence_s == 60.0
    assert survey.fast_cadence_s == 1.0
    assert survey.ratchet_len == 16
    assert survey.stamp_size == 28
    assert survey.stamp_resolution == 0.01
    assert survey.min_sun_alt == -18
    assert survey.min_moon_sep == 1.5
    assert survey.highspeed_start_moonfrac == 0.96
    assert survey.start_date == datetime(2000, 1, 1)
    assert survey.end_date == datetime(2001, 1, 1)


def test_config_defaults():
    """Test default values for Config."""
    config = Config(sim_version="1.0")
    assert config.sim_version == "1.0"
    assert config.run_name == "argussim"
    assert isinstance(config.structure, Structure)
    assert isinstance(config.observatory, Observatory)
    assert isinstance(config.telescope, TelescopeUnit)
    assert isinstance(config.packing_strategy, PackingStrategy)
    assert isinstance(config.filter_strategy, FilterStrategy)
    assert isinstance(config.survey, Survey)
    assert isinstance(config.uptime, Uptime)


def test_output_defaults():
    """Test default values for Output."""
    output = Output()
    assert output.save_grid_tables is True
    assert output.output_dir == "output"
    assert output.metadata is True
    assert output.cadence_levels == ["hour", "night", "season"]


def test_cadence_levels_roundtrip(tmp_path, monkeypatch):
    """Test that cadence_levels survives TOML serialization and deserialization."""
    monkeypatch.chdir(tmp_path)
    config = Config(sim_version="1.0")
    write_config(config)
    loaded = get_config("./argussim.toml")
    assert loaded.output.cadence_levels == ["hour", "night", "season"]


def test_cadence_levels_custom_roundtrip(tmp_path, monkeypatch):
    """Test that non-default cadence_levels roundtrips through TOML."""
    monkeypatch.chdir(tmp_path)
    config = Config(sim_version="1.0")
    config.output.cadence_levels = ["minute", "hour"]
    write_config(config)
    loaded = get_config("./argussim.toml")
    assert loaded.output.cadence_levels == ["minute", "hour"]


def test_get_config(tmp_path, monkeypatch):
    """Test loading a configuration from a TOML file."""
    monkeypatch.chdir(tmp_path)
    config = Config(sim_version="1.0")
    write_config(config)
    loaded_config = get_config("./argussim.toml")
    assert loaded_config.sim_version == "1.0"


def test_write_config(tmp_path, monkeypatch):
    """Test writing a configuration to a TOML file."""
    monkeypatch.chdir(tmp_path)
    config = Config(sim_version="1.0")
    config_path = write_config(config)
    assert os.path.exists(config_path)
    assert config_path.name == "argussim.toml"


if __name__ == "__main__":
    pytest.main()
