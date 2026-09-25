"""Signal-to-noise and noise budget calculations for Argus observations."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import numpy.typing as npt

from typing import Optional


class NoiseBudget:
    """CCD-equation noise budget: source, sky and dark-current shot noise plus read noise, weighted by PSF sharpness."""

    def __init__(
        self,
        read_noise: u.Quantity[u.electron] | None = 1.7 * u.electron,  # Electrons
        dark_current: u.Quantity[u.electron / u.second] = 0.0022 * u.electron / u.second,
    ) -> None:
        """Initialize the NoiseBudget object.

        Parameters
        ----------
        read_noise : Optional[u.Quantity[u.electron]], optional
            The read noise of the system in electrons. Default is 1.7 electrons.
        dark_current : u.Quantity[u.electron / u.second], optional
            The dark current of the system in electrons per second. Default is
            0.0022 electrons per second.

        """
        self.read_noise = read_noise
        self.dark_current = dark_current

    def get_snr(
        self,
        eflux: u.Quantity[u.electron / u.second],
        psf: npt.ArrayLike,
        eskyflux: u.Quantity[u.electron / u.second],
        exptime: u.Quantity[u.second],
    ) -> tuple[float, dict]:
        """Calculate the signal-to-noise ratio (SNR) for a given exposure.

        Parameters
        ----------
        eflux : u.Quantity[u.electron / u.second]
            The electron flux of the source.
        psf : npt.ArrayLike
            The point spread function (PSF) of the source.
        eskyflux : u.Quantity[u.electron / u.second]
            The electron flux of the sky background.
        exptime : u.Quantity[u.second]
            The exposure time.

        Returns
        -------
        tuple[float, dict]
            A tuple containing the SNR and a budget dictionary with detailed noise components.
            The budget dictionary contains:
            - 'sig_flux': The signal flux.
            - 'sky_flux': The sky flux.
            - 'signal': The total signal electrons.
            - 'noise': The total noise.
            - 'src_shot': The source shot noise.
            - 'sky_shot': The sky shot noise.
            - 'drk_shot': The dark current shot noise.
            - 'readnoise': The read noise.
            - 'bkg_only_noise': Approximate noise in the background-limited case.

        """
        # Sharpness S = sum(psf^2) for a sum-normalized PSF. The effective
        # number of background noise pixels is 1/S.
        # Ref: STScI WFPC2 Handbook Eq 6.5
        sharpness = np.sum(psf**2)

        signal_electrons = eflux * exptime
        sky_electrons = eskyflux * exptime
        dark_electrons = self.dark_current * exptime

        source_shot_noise = np.sqrt(signal_electrons)
        sky_shot_noise = np.sqrt(sky_electrons / sharpness)
        readnoise = np.sqrt(self.read_noise**2 / sharpness)
        dark_shot_noise = np.sqrt(dark_electrons / sharpness)

        # Total noise: signal shot noise + background terms (read, dark, sky)
        # scaled by 1/sharpness (the effective number of background pixels)
        noise = np.sqrt(
            signal_electrons.value
            + ((self.read_noise.value**2 + dark_electrons.value + sky_electrons.value) / sharpness),
        )

        # Background-limited noise floor (signal contribution excluded)
        bkg_only_noise = np.sqrt(
            (self.read_noise.value**2 + dark_electrons.value + sky_electrons.value) / sharpness,
        )

        budget = {
            "sig_flux": eflux.value,
            "sky_flux": eskyflux.value,
            "signal": signal_electrons.value,
            "noise": noise,
            "src_shot": source_shot_noise.value,
            "sky_shot": sky_shot_noise.value,
            "drk_shot": dark_shot_noise.value,
            "readnoise": readnoise.value,
            "bkg_only_noise": bkg_only_noise,
        }

        return signal_electrons.value / noise, budget

    def get_flux_at_snr(  # noqa: PLR0913
        self,
        snr: float,
        psf: npt.ArrayLike | None,
        eskyflux: u.Quantity[u.electron / u.second],
        exptime: u.Quantity[u.second],
        signal_throughput: float,
        coadd_n: float | None = 1,
        read_noise: Optional[float] = None,
        dark_current: Optional[float] = None,
        sharpness: npt.ArrayLike | None = None,
    ) -> tuple[u.Quantity[u.electron / u.second], dict]:
        """Calculate the required flux to achieve a given signal-to-noise ratio (SNR).

        Parameters
        ----------
        snr : float
            The desired signal-to-noise ratio.
        psf : npt.ArrayLike
            The point spread function (PSF) of the source.
        eskyflux : u.Quantity[u.electron / u.second]
            The electron flux of the sky background per pixel.
        exptime : u.Quantity[u.second]
            The exposure time.
        signal_throughput : float
            The throughput of the signal (ie, atmosphere + instrument).
        coadd_n : float, optional
            The number of coadded exposures. Default is 1. Assumes every
            exposure has the same noise (the same sky brightness).
        read_noise : float, optional
            The read noise of the system in electrons, if different from a
            single image or if coaddition logic has been handled externally.
            Default is None (uses instance read_noise).
        dark_current : float, optional
            The dark current of the system in electrons per second, if different
            from a single image or if coaddition logic has been handled
            externally. Default is None (uses instance dark_current).
        sharpness : array_like, optional
            sum(PSF^2) given directly (per row, e.g. from a PSF table over
            field angle and seeing); overrides ``psf``.

        Returns
        -------
        tuple[u.Quantity[u.electron / u.second], dict]
            A tuple containing the above-atmosphere equivalent electron count
            required to achieve the given SNR, and a dictionary with detailed
            noise components. The returned electrons represent the flux before
            atmospheric and instrumental losses; to recover detected electrons,
            multiply the returned value by signal_throughput.

        """
        if sharpness is not None:
            sharpness = np.asarray(sharpness, dtype=np.float64)
        elif psf is None:
            # no PSF: sharpness 1, for coadds whose exposures span many PSFs
            sharpness = 1
        else:
            sharpness = np.sum(psf**2)

        if dark_current is None:
            dark_current = self.dark_current
        if read_noise is None:
            read_noise = self.read_noise

        sky_electrons = eskyflux * exptime
        dark_electrons = dark_current * exptime  # dark current in e-/px/s

        # Derivation: CCD equation SNR = S / sqrt(S + B/s) where
        #   S = detected source electrons = F * t * eta  (F = above-atm flux,
        #       t = effective integration time = exptime * coadd_n, eta = throughput)
        #   B = read_noise^2 + dark_electrons + sky_electrons  (per-pixel background)
        #   s = sharpness = sum(PSF^2)
        #
        # Squaring both sides:  SNR^2 * (S + B/s) = S^2
        # Rearranging:          S^2 - SNR^2 * S - SNR^2 * B/s = 0
        #
        # This is a standard quadratic a*S^2 + b*S + c = 0 with
        #   a = 1,  b = -SNR^2,  c = -SNR^2 * B/s
        #
        # Positive root via the quadratic formula:
        #   S = (SNR^2 + sqrt(SNR^4 + 4 * SNR^2 * B/s)) / 2
        #
        # Substituting S = F * coadd_n * eta and solving for F gives the
        # above-atmosphere electron flux below.
        above_atm_electrons = (
            (snr**2 * signal_throughput)
            + (
                snr
                * signal_throughput
                * np.sqrt(
                    4 * sky_electrons.value * coadd_n
                    + 4 * dark_electrons.value * coadd_n
                    + sharpness * snr**2
                    + 4 * coadd_n * read_noise.value**2,
                )
            )
            / np.sqrt(sharpness)
        ) / (2 * coadd_n * signal_throughput**2)

        noise = np.sqrt(
            above_atm_electrons * signal_throughput
            + ((read_noise.value**2 + dark_electrons.value + sky_electrons.value) / sharpness),
        )

        # Single-frame noise breakdown; not divided by sqrt(coadd_n)
        noise_dict = {
            "above_atm_electrons": above_atm_electrons,
            "source_shot": np.sqrt(above_atm_electrons * signal_throughput),
            "readnoise": np.sqrt((read_noise.value**2) / sharpness),
            "dark_electrons": (dark_electrons.value / sharpness),
            "sky_electrons": (sky_electrons.value / sharpness),
            "noise": noise,
            "sharpness": sharpness,
        }

        noise /= np.sqrt(coadd_n)

        return above_atm_electrons * u.electron, noise_dict
