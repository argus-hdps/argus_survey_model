import astropy.time as atime
import astropy.units as u
import pytest

from argus_sim import CradlePointing, c


def test_cradle_pointing_grid_initialization():
    """Test the initialization of CradlePointing in grid mode."""
    cradle_pointing = CradlePointing(tiling="grid", n_telescopes=650)
    assert cradle_pointing.long_axis_extent.value == pytest.approx(3.26, rel=1e-2)
    assert cradle_pointing.short_axis_extent.value == pytest.approx(2.44, rel=1e-2)
    assert cradle_pointing.short_axis_overlap == 0.257 * u.deg
    assert cradle_pointing.long_axis_overlap == 0.257 * u.deg
    assert cradle_pointing.n_telescopes == 650
    assert cradle_pointing.tiling == "grid"


def test_cradle_pointing_grid_to_hpx_table_at():
    """Test the to_hpx_table_at method with grid tiling."""
    cradle_pointing = CradlePointing(tiling="grid", n_telescopes=650)
    time = atime.Time.now()
    grid_table = cradle_pointing.to_hpx_table_at(time)
    assert "ra" in grid_table.colnames
    assert "dec" in grid_table.colnames
    assert "alt" in grid_table.colnames
    assert "az" in grid_table.colnames
    assert "healpix" in grid_table.colnames


def test_cradle_pointing_ring_initialization():
    """Test CradlePointing defaults to ring tiling."""
    cradle_pointing = CradlePointing()
    assert cradle_pointing.tiling == "ring"
    assert len(cradle_pointing.cam_pointings) == 1200
    assert "subarray_n" in cradle_pointing.cam_pointings.colnames
    assert "rot_deg" in cradle_pointing.cam_pointings.colnames
    assert "filter" in cradle_pointing.cam_pointings.colnames


def test_cradle_pointing_ring_to_hpx_table_at():
    """Test to_hpx_table_at works with ring tiling."""
    cradle_pointing = CradlePointing()
    time = atime.Time.now()
    table = cradle_pointing.to_hpx_table_at(time)
    assert "ra" in table.colnames
    assert "dec" in table.colnames
    assert "alt" in table.colnames
    assert "az" in table.colnames
    assert "healpix" in table.colnames


@pytest.mark.parametrize("n_telescopes", [50, 650, 900])
def test_collecting_area_scales_with_n_telescopes(n_telescopes):
    """Collecting area equals N times the single-telescope annular area."""
    cradle = CradlePointing(tiling="grid", n_telescopes=n_telescopes)
    n = len(cradle.cam_pointings)

    expected = (n * c.telescope.collecting_area).to("m^2")

    assert cradle.collecting_area.unit == expected.unit
    assert cradle.collecting_area.value == pytest.approx(expected.value, rel=1e-10)


if __name__ == "__main__":
    pytest.main()
