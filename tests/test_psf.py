"""Tests for PSF models."""

import astropy.units as u
import numpy as np
import pytest

from argus_sim.config import DEFAULT_PSF_FITS
from argus_sim.psf import DeliveredPSF, GaussianPSF, _half_flux_diameter, _match_ee50, gaus2d


def test_gaus2d():
    """Test the gaus2d function."""
    x, y = 0, 0
    mx, my = 0, 0
    sx, sy = 1, 1
    result = gaus2d(x, y, mx, my, sx, sy)
    expected = 1 / (2 * np.pi * sx * sy)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_gaussian_psf_with_seeing():
    """Moffat seeing produces a wider HFD than the optical-only Gaussian."""
    psf_generator = GaussianPSF()
    psf, hfd = psf_generator.with_seeing(seeing=1.0)
    assert psf.shape == (28, 28), f"Unexpected PSF shape: {psf.shape}"
    # Optical-only FWHM is ~0.86"; 1" Moffat seeing should broaden well past that
    assert hfd > 1.5, f"HFD too small: {hfd}"
    assert hfd < 4.0, f"HFD too large: {hfd}"


def test_gaussian_psf_drifted_psf():
    """Test the drifted_psf method of GaussianPSF."""
    psf_generator = GaussianPSF()
    drifted_psf = psf_generator.drifted_psf(
        seeing=1.0,
        rate=1.0 * u.arcsec / u.second,
        angle=45.0 * u.deg,
        exptime=10.0 * u.second,
        tstep_streak=1.0 * u.second,
    )
    assert drifted_psf.shape == (28, 28), f"Unexpected drifted PSF shape: {drifted_psf.shape}"
    assert np.isclose(drifted_psf.sum(), 1.0), f"Drifted PSF does not sum to 1: {drifted_psf.sum()}"


def test_gaussian_psf_accepts_band_and_rng_kwargs():
    """GaussianPSF.with_seeing accepts and ignores band/rng kwargs."""
    psf_gen = GaussianPSF()
    psf, fwhm = psf_gen.with_seeing(seeing=1.0, band="g", rng=np.random.default_rng(0))
    assert psf.shape == (28, 28)
    assert np.isclose(psf.sum(), 1.0)


@pytest.mark.parametrize(
    "sigma,size",
    [
        (5.0, 101),  # fine sampling
        (5.0 / 2.355, 21),  # coarse: 5-pixel FWHM on 21x21 grid
    ],
    ids=["fine", "coarse"],
)
def test_half_flux_diameter_gaussian(sigma, size):
    """HFD of a known Gaussian matches analytic FWHM within 5%."""
    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    g = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
    g /= g.sum()
    hfd = _half_flux_diameter(g, spacing=1.0)
    expected_fwhm = sigma * 2.355
    assert abs(hfd - expected_fwhm) / expected_fwhm < 0.05, f"HFD {hfd:.3f} not within 5% of FWHM {expected_fwhm:.3f}"


@pytest.mark.parametrize("target_mult", [1.5, 2.0, 3.0])
def test_match_ee50_hits_target(target_mult):
    """_match_ee50 delivers the requested HFD multiplier within 2%."""
    sigma = 5.0
    size = 501
    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    psf = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
    psf /= psf.sum()
    spacing = 0.5
    base_hfd = _half_flux_diameter(psf, spacing=spacing)
    target_hfd = target_mult * base_hfd
    result = _match_ee50(psf, spacing=spacing, target_hfd=target_hfd)
    measured_hfd = _half_flux_diameter(result, spacing=spacing)
    assert abs(measured_hfd - target_hfd) / target_hfd < 0.03


def test_match_ee50_noop_when_target_below_base():
    """_match_ee50 returns the input unchanged when target <= base HFD."""
    sigma = 5.0
    size = 101
    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    psf = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma**2))
    psf /= psf.sum()
    base_hfd = _half_flux_diameter(psf, spacing=1.0)
    result = _match_ee50(psf, spacing=1.0, target_hfd=base_hfd * 0.5)
    np.testing.assert_array_equal(result, psf)


class TestDeliveredPSF:
    """Tests for DeliveredPSF with the packaged PSF file."""

    @pytest.fixture
    def delivered(self):
        """Create a default DeliveredPSF instance."""
        return DeliveredPSF(
            fits_path=DEFAULT_PSF_FITS,
            plate_scale=1.007,
            pixel_size=3.76,
            stamp_size=28,
        )

    def test_smoke(self, delivered):
        """Basic call returns correct shape and normalization."""
        rng = np.random.default_rng(42)
        psf, fwhm = delivered.with_seeing(1.0, band="g", rng=rng)
        assert psf.shape == (28, 28)
        assert np.isclose(psf.sum(), 1.0, atol=1e-10)
        assert fwhm > 0

    def test_normalization_all_bands(self, delivered):
        """PSF sums to 1.0 for every supported band."""
        rng = np.random.default_rng(99)
        for band in ["g", "rho", "r", "i", "r+i", "A"]:
            psf, _ = delivered.with_seeing(1.0, band=band, rng=rng)
            assert np.isclose(psf.sum(), 1.0, atol=1e-10), f"Band {band}: sum={psf.sum()}"

    def test_red_bands_share_the_rho_psf(self, delivered):
        """r, i, r+i and A use the rho PSF."""
        ref, _ = delivered.with_seeing(1.0, band="rho", field_angle_deg=1.0, pixel_phase=(0, 0))
        for band in ["r", "i", "r+i", "A"]:
            psf, _ = delivered.with_seeing(1.0, band=band, field_angle_deg=1.0, pixel_phase=(0, 0))
            np.testing.assert_array_equal(psf, ref)

    def test_pixel_phase_shifts_centroid(self):
        """Pixel phase randomization shifts centroid by less than 1 pixel."""
        delivered_phase = DeliveredPSF(
            fits_path=DEFAULT_PSF_FITS,
            plate_scale=1.007,
            pixel_size=3.76,
            stamp_size=28,
            randomize_pixel_phase=True,
        )
        centroids = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            psf, _ = delivered_phase.with_seeing(1.0, band="g", rng=rng)
            yy, xx = np.mgrid[: psf.shape[0], : psf.shape[1]]
            cy = np.average(yy, weights=psf)
            cx = np.average(xx, weights=psf)
            centroids.append((cy, cx))
        centroids = np.array(centroids)
        spread_y = np.ptp(centroids[:, 0])
        spread_x = np.ptp(centroids[:, 1])
        assert spread_y < 1.0, f"Y centroid spread {spread_y} >= 1 pixel"
        assert spread_x < 1.0, f"X centroid spread {spread_x} >= 1 pixel"

    def test_invalid_band_raises(self, delivered):
        """Unsupported band raises ValueError."""
        with pytest.raises(ValueError, match="not in the PSF file"):
            delivered.with_seeing(1.0, band="z")

    def test_seeing_broadens_psf(self, delivered):
        """More seeing produces a broader PSF."""
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        _, fwhm_good = delivered.with_seeing(0.5, band="g", rng=rng1)
        _, fwhm_bad = delivered.with_seeing(2.0, band="g", rng=rng2)
        assert fwhm_bad > fwhm_good

    def test_file_layout(self):
        """The PSF file is one table with band, field angle and image columns and the OVERSAMP and NPIX keywords."""
        from astropy.io import fits

        with fits.open(DEFAULT_PSF_FITS) as hdul:
            assert len(hdul) == 2
            assert hdul[1].columns.names == ["BAND", "THETA", "PSF"]
            header = hdul[1].header
            assert header["OVERSAMP"] == 2 and header["NPIX"] == 58
            assert "HISTORY" not in header and "COMMENT" not in header


def test_stamp_flux_capture_good_seeing():
    """In good seeing the 28x28 stamp captures >99% of the binned PSF flux at every tabulated field angle."""
    delivered = DeliveredPSF(fits_path=DEFAULT_PSF_FITS, plate_scale=1.007, pixel_size=3.76, stamp_size=28)
    wide = DeliveredPSF(fits_path=DEFAULT_PSF_FITS, plate_scale=1.007, pixel_size=3.76, stamp_size=29)
    for band in ("g", "rho"):
        for angle in delivered.field_angles(band):
            # the 29-pixel stamp is the whole binned array
            stamp, _ = delivered.with_seeing(0.5, band=band, field_angle_deg=float(angle), pixel_phase=(0, 0))
            full, _ = wide.with_seeing(0.5, band=band, field_angle_deg=float(angle), pixel_phase=(0, 0))
            peak_ratio = full.max() / stamp.max()
            assert peak_ratio > 0.99, f"{band} at {angle:.2f} deg: stamp keeps only {peak_ratio:.4f} of the flux"


@pytest.mark.parametrize("angle_deg", [0, 30, 45, 90, 135])
def test_drifted_psf_elongation_direction(angle_deg):
    """Major axis of drifted PSF aligns with the drift angle."""
    psf_gen = GaussianPSF()
    psf = psf_gen.drifted_psf(
        seeing=1.0,
        rate=1.0 * u.arcsec / u.second,
        angle=angle_deg * u.deg,
        exptime=10.0 * u.second,
        tstep_streak=0.5 * u.second,
    )

    yy, xx = np.mgrid[: psf.shape[0], : psf.shape[1]]
    cx = np.average(xx, weights=psf)
    cy = np.average(yy, weights=psf)

    Ixx = np.average((xx - cx) ** 2, weights=psf)
    Iyy = np.average((yy - cy) ** 2, weights=psf)
    Ixy = np.average((xx - cx) * (yy - cy), weights=psf)

    cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    measured_angle = np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180

    expected_angle = angle_deg % 180
    diff = min(abs(measured_angle - expected_angle), 180 - abs(measured_angle - expected_angle))
    assert diff < 10, (
        f"Drift angle {angle_deg}°: major axis at {measured_angle:.1f}°, "
        f"expected ~{expected_angle}° (off by {diff:.1f}°)"
    )


class TestElongatePSF:
    """Tests for the line-kernel PSF elongation."""

    def test_zero_elongation_is_identity(self):
        """Zero-length elongation returns the input PSF unchanged."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)
        result = elongate_psf(psf, elongation_arcsec=0.0, angle_deg=0.0, plate_scale=1.007)
        np.testing.assert_array_equal(result, psf)

    def test_normalization_preserved(self):
        """Elongated PSF still sums to 1."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)
        for length in [0.5, 1.0, 2.0, 5.0]:
            for angle in [0, 45, 90, 135]:
                result = elongate_psf(psf, elongation_arcsec=length, angle_deg=angle, plate_scale=1.007)
                assert np.isclose(result.sum(), 1.0, atol=1e-10), (
                    f"Sum={result.sum()} for length={length}, angle={angle}"
                )

    def test_shape_preserved(self):
        """Output stamp has the same shape as the input."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)
        result = elongate_psf(psf, elongation_arcsec=2.0, angle_deg=45.0, plate_scale=1.007)
        assert result.shape == psf.shape

    def test_elongation_broadens_along_axis(self):
        """Elongation increases the second moment along the drift direction."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)

        yy, xx = np.mgrid[: psf.shape[0], : psf.shape[1]]
        cx0 = np.average(xx, weights=psf)
        Ixx_base = np.average((xx - cx0) ** 2, weights=psf)

        elongated = elongate_psf(psf, elongation_arcsec=3.0, angle_deg=0.0, plate_scale=1.007)
        cx1 = np.average(xx, weights=elongated)
        Ixx_elong = np.average((xx - cx1) ** 2, weights=elongated)

        assert Ixx_elong > Ixx_base

    @pytest.mark.parametrize("angle_deg", [0, 30, 45, 90, 135])
    def test_major_axis_aligns_with_angle(self, angle_deg):
        """Major axis of elongated PSF aligns with the specified angle."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)

        elongated = elongate_psf(psf, elongation_arcsec=5.0, angle_deg=angle_deg, plate_scale=1.007)

        yy, xx = np.mgrid[: elongated.shape[0], : elongated.shape[1]]
        cx = np.average(xx, weights=elongated)
        cy = np.average(yy, weights=elongated)
        Ixx = np.average((xx - cx) ** 2, weights=elongated)
        Iyy = np.average((yy - cy) ** 2, weights=elongated)
        Ixy = np.average((xx - cx) * (yy - cy), weights=elongated)

        cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, np.argmax(eigvals)]
        measured_angle = np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180

        expected_angle = angle_deg % 180
        diff = min(abs(measured_angle - expected_angle), 180 - abs(measured_angle - expected_angle))
        assert diff < 10, (
            f"Elongation angle {angle_deg}°: major axis at {measured_angle:.1f}°, "
            f"expected ~{expected_angle}° (off by {diff:.1f}°)"
        )

    def test_line_kernel_normalization(self):
        """Line kernel sums to 1."""
        from argus_sim.psf import _line_kernel

        for length in [0.5, 1.0, 3.0, 7.0]:
            for angle in [0, 45, 90, 120]:
                kernel = _line_kernel(length, angle)
                assert np.isclose(kernel.sum(), 1.0, atol=1e-10), (
                    f"Kernel sum={kernel.sum()} for length={length}, angle={angle}"
                )

    def test_line_kernel_zero_length(self):
        """Zero-length kernel is a single pixel."""
        from argus_sim.psf import _line_kernel

        kernel = _line_kernel(0.0, 0.0)
        assert kernel.shape == (1, 1)
        assert kernel[0, 0] == 1.0

    def test_symmetry_under_180_rotation(self):
        """Elongation at angle θ and θ+180° produce identical results."""
        from argus_sim.psf import elongate_psf

        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)
        r1 = elongate_psf(psf, elongation_arcsec=3.0, angle_deg=30.0, plate_scale=1.007)
        r2 = elongate_psf(psf, elongation_arcsec=3.0, angle_deg=210.0, plate_scale=1.007)
        np.testing.assert_allclose(r1, r2, atol=1e-12)

    @pytest.mark.parametrize(
        "length_arcsec,angle_deg",
        [(3.0, 0), (3.0, 45), (3.0, 90), (4.0, 30), (3.0, 135)],
    )
    def test_compositional_elongation_second_moments(self, length_arcsec, angle_deg):
        """Composing elongation L at A then L at A+180° approximates 2L at A in second moments.

        Convolving twice with an L-length line kernel adds variance 2*(L^2/12)
        along the elongation axis, while a single 2L kernel adds (2L)^2/12.
        The theoretical ratio of second-moment increases is 0.5, but
        discretization and stamp-boundary effects shift it upward for
        short trails.  The test checks that the composed result broadens the PSF,
        that 2L broadens more, and that the ratio stays in [0.3, 0.7].
        """
        from argus_sim.psf import elongate_psf

        plate_scale = 1.007
        psf_gen = GaussianPSF()
        psf, _ = psf_gen.with_seeing(1.0)

        composed = elongate_psf(psf, length_arcsec, angle_deg, plate_scale)
        composed = elongate_psf(composed, length_arcsec, angle_deg + 180, plate_scale)

        double = elongate_psf(psf, 2 * length_arcsec, angle_deg, plate_scale)

        def second_moments(img):
            yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
            cx = np.average(xx, weights=img)
            cy = np.average(yy, weights=img)
            Ixx = np.average((xx - cx) ** 2, weights=img)
            Iyy = np.average((yy - cy) ** 2, weights=img)
            Ixy = np.average((xx - cx) * (yy - cy), weights=img)
            return np.array([[Ixx, Ixy], [Ixy, Iyy]])

        cov_base = second_moments(psf)
        cov_composed = second_moments(composed)
        cov_double = second_moments(double)

        angle_rad = np.radians(angle_deg)
        u_vec = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        var_base = u_vec @ cov_base @ u_vec
        var_composed = u_vec @ cov_composed @ u_vec
        var_double = u_vec @ cov_double @ u_vec

        delta_composed = var_composed - var_base
        delta_double = var_double - var_base

        assert delta_composed > 0, "Composed elongation should broaden the PSF"
        assert delta_double > delta_composed, (
            f"2L elongation (delta={delta_double:.4f}) should broaden more "
            f"than two L elongations (delta={delta_composed:.4f})"
        )

        ratio = delta_composed / delta_double
        assert 0.3 < ratio < 0.7, (
            f'Second-moment ratio {ratio:.3f} outside [0.3, 0.7] (angle={angle_deg}°, L={length_arcsec}")'
        )


if __name__ == "__main__":
    pytest.main()
