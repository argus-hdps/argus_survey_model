"""System throughput: QE, filter bandpasses, and first-order atmospheric extinction."""

from importlib.resources import files
from pathlib import Path
from typing import Optional, Dict

import astropy.constants as c
import astropy.io.ascii as asc
import astropy.units as u
import numpy as np
import scipy.interpolate as sint

from .utils import dotdict

# Efficiency of the telescope optics (the fraction of the light entering the aperture that the optics deliver to the
# sensor), one band average per filter.  Each value is the optics transmission averaged over the filter x QE x
# atmosphere photon-rate response of that filter at airmass 1.15, the typical survey airmass, with the weighting
# build_band_params uses for the signal.
# Scaled by throughput_loss.
OPTICS_EFFICIENCY: Dict[str, float] = {
    "A": 0.812,
    "V": 0.878,
    "sqm": 0.834,
    "g": 0.787,
    "r": 0.866,
    "i": 0.751,
    "r+i+z": 0.824,
    "r+i": 0.832,
    "i+z": 0.736,
    "rho": 0.837,
}


class SystemThroughput:
    """A class to handle the system throughput for a telescope system.

    Parameters
    ----------
    throughputs_dir : Optional[str], optional
        The directory containing throughput data files. Default is None.
    throughput_loss : float, optional
        Multiplicative degradation of the optics efficiency. Default is 1.0.
    filters : Dict[str, bool], optional
        Dictionary specifying which filters to include. Keys are filter names,
        values are booleans. Default enables all filters.
        Available filters: 'A', 'V', 'sqm', 'g', 'r', 'i', 'r+i+z', 'r+i', 'i+z', 'rho'

    Attributes
    ----------
    wav : astropy.units.Quantity
        The wavelength array.
    optics : dotdict
        Band-averaged optics efficiency per active filter (``OPTICS_EFFICIENCY`` x ``throughput_loss``).
    qe_wav : numpy.ndarray
        The quantum efficiency as a function of wavelength.
    filt_wav : dotdict
        The filter responses as a function of wavelength.
    freq : astropy.units.Quantity
        The frequency array.
    qe_freq : numpy.ndarray
        The quantum efficiency as a function of frequency.
    filt_freq : dotdict
        The filter responses as a function of frequency.

    """

    def __init__(
        self,
        throughputs_dir: Optional[str] = None,
        throughput_loss: float = 1.0,
        filters: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Initialize the SystemThroughput object."""
        if throughputs_dir is None:
            throughputs_dir = str(files("argus_sim").joinpath("data", "throughputs"))

        # Filter specs: file, x_col, y_col, x_scale, y_scale
        filter_files = {
            "A": {
                "file": "argus_A.csv",
                "x_col": "Wavelength (nm)",
                "y_col": "T",
                "x_scale": 1,
                "y_scale": 1 / 100,
            },
            "V": {"file": "GCPD_Johnson.V.dat", "x_col": "col1", "y_col": "col2", "x_scale": 1 / 10, "y_scale": 1},
            "sqm": {"file": "Cameras_SQM.CM500.dat", "x_col": "col1", "y_col": "col2", "x_scale": 1 / 10, "y_scale": 1},
            "g": {"file": "argus_g.csv", "x_col": "Wavelength(nm)", "y_col": "T", "x_scale": 1, "y_scale": 1 / 100},
            "r": {"file": "argus_r.csv", "x_col": "Wavelength", "y_col": "T", "x_scale": 1, "y_scale": 1 / 100},
            "i": {"file": "argus_i.csv", "x_col": "Wavelength", "y_col": "T", "x_scale": 1, "y_scale": 1 / 100},
            "r+i+z": {"file": "longpass_550nm.csv", "x_col": "wav_nm", "y_col": "T", "x_scale": 1, "y_scale": 1},
            "i+z": {"file": "longpass_700nm.csv", "x_col": "wav_nm", "y_col": "T", "x_scale": 1, "y_scale": 1},
            "r+i": {"file": "bandpass_560_850nm.csv", "x_col": "wav_nm", "y_col": "T", "x_scale": 1, "y_scale": 1},
            "rho": {"file": "bandpass_560_825nm.csv", "x_col": "wav_nm", "y_col": "T", "x_scale": 1, "y_scale": 1},
        }

        # If no filters specified, enable all. If filters specified, default to False
        if filters is None:
            self.active_filters = {name: True for name in filter_files}
        else:
            self.active_filters = {name: filters.get(name, False) for name in filter_files}

        # V filter is always needed for spectral sky V-band normalization
        self.active_filters["V"] = True

        tp_dir = Path(throughputs_dir)
        qe = asc.read(str(tp_dir / "imx455_qe.csv"))
        modtran = asc.read(str(tp_dir / "pachon_modtran_atm_aerosols.txt"))

        qe.sort("col1")
        modtran.sort("Wavelength(nm)")

        qe_interp = sint.CubicSpline(qe["col1"], qe["col2"])
        modtran_interp = sint.CubicSpline(modtran["Wavelength(nm)"], modtran["Throughput(0-1)"])

        # 300-1000 nm grid; the frequency grid runs the other way
        wavs = np.linspace(300, 1000, 10000) * u.nm
        freq = np.flip((c.c / wavs).to(u.Hertz))

        filter_responses = {}
        filter_freqs = {}

        for filt_name, spec in filter_files.items():
            if not self.active_filters.get(filt_name):
                continue

            data = asc.read(str(tp_dir / spec["file"]))
            data.sort(spec["x_col"])

            x = data[spec["x_col"]] * spec["x_scale"]
            y = data[spec["y_col"]] * spec["y_scale"]

            interp = sint.PchipInterpolator(x, y, extrapolate=False)
            response = np.nan_to_num(interp(wavs), nan=0.0)
            response[(wavs.value < x.min()) | (wavs.value > x.max())] = 0

            filter_responses[filt_name] = response

            # the frequency grid is the wavelength grid reversed
            freq_response = np.flip(response)
            filter_freqs[filt_name] = freq_response

        self.wav = wavs
        self.optics = dotdict({name: OPTICS_EFFICIENCY[name] * throughput_loss for name in filter_responses})
        self.qe_wav = qe_interp(wavs)

        self.filt_wav = dotdict({"atmosphere": modtran_interp(wavs), **filter_responses})

        self.freq = freq
        self.qe_freq = np.flip(self.qe_wav) * (u.electron / u.photon)

        self.filt_freq = dotdict(
            {
                "atmosphere": np.flip(modtran_interp(wavs)),
                "qe": self.qe_freq,
                **filter_freqs,
            }
        )

    @staticmethod
    def as_callable(x, y):
        """Create a cubic spline interpolation of the throughput function."""
        return sint.CubicSpline(x, y, extrapolate=False)
