import astropy.table as tbl
import astropy.units as u
import numpy as np
import pytest
from mocpy import MOC
from numpy.testing import assert_almost_equal

from argus_sim.grid_helper import grid_pointing, rotate, xy_to_z


def test_rotate():
    """Test the rotate function."""
    c = np.array([1, 0, 0])
    axis = np.array([0, 0, 1])
    theta = np.pi / 2
    rotated_vector = rotate(c, axis, theta)
    expected_vector = np.array([0, 1, 0])
    assert_almost_equal(rotated_vector, expected_vector, decimal=6)


def test_xy_to_z():
    """Test the xy_to_z function."""
    x = np.array([0, 1])
    y = np.array([0, 1])
    rad = 1
    x_out, y_out, z_out, mask = xy_to_z(x, y, rad)
    expected_x = np.array([0, 1, 0])
    expected_y = np.array([0, 0, 1])
    expected_z = np.array([-1, 0, 0])
    expected_mask = np.array([[True, True], [True, False]])
    assert_almost_equal(x_out, expected_x, decimal=6)
    assert_almost_equal(y_out, expected_y, decimal=6)
    assert_almost_equal(z_out, expected_z, decimal=6)
    assert np.all(mask == expected_mask)


def test_grid_pointing():
    """Test the grid_pointing function."""
    long_axis_extent = 10 * u.deg
    short_axis_extent = 5 * u.deg
    short_axis_overlap = 1 * u.deg
    long_axis_overlap = 1 * u.deg
    fov, fov_area, ot, sim_table = grid_pointing(
        long_axis_extent, short_axis_extent, short_axis_overlap, long_axis_overlap
    )

    assert isinstance(fov, MOC)
    assert isinstance(fov_area, u.Quantity)
    assert fov_area.unit == u.deg**2
    assert isinstance(ot, tbl.Table)
    assert "x_center" in ot.colnames
    assert "y_center" in ot.colnames
    assert "z_center" in ot.colnames
    assert "alt_center" in ot.colnames
    assert "az_center" in ot.colnames
    assert "ra_center" in ot.colnames
    assert "dec_center" in ot.colnames
    assert isinstance(sim_table, tbl.Table)
    assert "alt" in sim_table.colnames
    assert "az" in sim_table.colnames


if __name__ == "__main__":
    pytest.main()
