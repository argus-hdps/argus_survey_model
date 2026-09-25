"""Conversions between the AB magnitude system and instrument-native units."""

from __future__ import annotations

from typing import Optional, Union

import astropy.constants as c
import astropy.units as u
import numpy as np
import numpy.typing as npt
import scipy.integrate as sing

from . import c as config
from .throughput import SystemThroughput


class ABPhot:
    """A class to handle AB photometry calculations.

    Parameters
    ----------
    throughputs_dir : str
        Directory containing throughput files.
    collecting_area : Optional[u.Quantity[u.cm**2]], optional
        The collecting area of one telescope. If None, the configured ``telescope.collecting_area`` is used.
    throughput_loss : float, optional
        The throughput loss factor. Default is 1.0.

    Attributes
    ----------
    tp : SystemThroughput
        The system throughput object.
    zp_ergs : u.Quantity
        The zero-point flux density in ergs.
    collecting_area : u.Quantity
        The collecting area of the telescope.
    freqs : npt.ArrayLike
        The throughput frequencies.

    Methods
    -------
    photons_from_flux_density(band_name, flux_density)
        Calculate the number of photons from a given flux density.
    photons_from_mag(band_name, mag)
        Calculate the number of photons from a given magnitude.
    mag_from_photons(band_name, photons)
        Calculate the magnitude from a given number of photons.

    """

    def __init__(
        self,
        throughput: Optional[SystemThroughput] = None,
        throughputs_dir: Optional[str] = None,
        collecting_area: u.Quantity[u.cm**2] | None = None,
        throughput_loss: float = 1.0,
    ) -> None:
        """Initialize the ABPhot object.

        Parameters
        ----------
        throughput : Optional[SystemThroughput], optional
            A pre-constructed SystemThroughput object. If provided,
            throughputs_dir and throughput_loss are ignored.
        throughputs_dir : Optional[str], optional
            Directory containing throughput files. Used only if
            throughput is None.
        collecting_area : Optional[u.Quantity[u.cm**2]], optional
            The collecting area of one telescope. If None, the configured ``telescope.collecting_area`` is used.
        throughput_loss : float, optional
            The throughput loss factor. Default is 1.0.

        """
        if throughput is None:
            self.tp = SystemThroughput(throughputs_dir, throughput_loss=throughput_loss)
        else:
            self.tp = throughput
        # AB zero-point: m_AB = -2.5 log10(F_nu) - 48.60 (Oke & Gunn 1983,
        # ApJ 266, 713, Eq. 7). The constant 48.60 = -2.5 log10(3631 Jy in
        # cgs), so F_nu(m=0) = 3631 Jy = 3.631e-20 erg/s/cm^2/Hz.
        self.zp_ergs = 10 ** (48.60 / -2.5) * (u.erg / (u.second * u.cm**2 * u.Hertz))
        if collecting_area is None:
            collecting_area = config.telescope.collecting_area
        self.collecting_area = collecting_area
        self.freqs = self.tp.freq

    def photons_from_flux_density(
        self,
        band_name: str,
        flux_density: u.Quantity[u.erg / (u.second * u.cm**2 * u.Hertz)],
    ) -> u.Quantity[u.second**-1]:
        """Calculate the number of photons from a given flux density.

        Parameters
        ----------
        band_name : str
            The name of the band.
        flux_density : u.Quantity[u.erg / (u.second * u.cm**2 * u.Hertz)]
            The flux density in ergs per second per square centimeter per Hertz.

        Returns
        -------
        u.Quantity[u.second**-1]
            The number of photons per second.

        """
        # Integrate photon rate: N = A * ∫ (F_nu * R(nu)) / (h * nu) dnu
        # where R(nu) is the band response and h*nu is the photon energy
        resp_func = self.tp.as_callable(
            self.freqs,
            self.tp.filt_freq[band_name],
        )
        y = (flux_density * resp_func(self.freqs) / (c.h * self.freqs)).cgs.value
        x = self.freqs.to(u.Hertz).value
        photon_density = (
            sing.trapezoid(
                y,
                x,
                dx=np.mean(np.diff(x)),
            )
            * u.Hertz
            * u.cm ** (-2)
        )
        return (photon_density).cgs * self.collecting_area * u.photon

    def photons_from_mag(self, band_name: str, mag: Union[float, npt.ArrayLike]) -> u.Quantity[u.second**-1]:
        """Calculate the number of photons from a given magnitude.

        Parameters
        ----------
        band_name : str
            The name of the band.
        mag : float
            The magnitude.

        Returns
        -------
        u.Quantity[u.second**-1]
            The number of photons per second.

        """
        zp_photons = self.photons_from_flux_density(band_name, self.zp_ergs)
        return zp_photons * 10 ** (mag / -2.5)

    def mag_from_photons(self, band_name: str, photons: u.Quantity[u.second**-1]) -> float:
        """Calculate the magnitude from a given number of photons.

        Parameters
        ----------
        band_name : str
            The name of the band.
        photons : u.Quantity[u.second**-1]
            The number of photons per second.

        Returns
        -------
        float
            The magnitude.

        """
        zp_photons = self.photons_from_flux_density(band_name, self.zp_ergs)
        return -2.5 * np.log10(photons / zp_photons)
