"""ArgusSim: survey simulation for the Argus Optical Array."""

__all__ = [
    "get_config",
    "CradlePointing",
    "ABPhot",
    "NoiseBudget",
    "SystemThroughput",
    "GaussianPSF",
    "DeliveredPSF",
    "Survey",
    "SpectralSky",
]

import importlib.metadata
import logging

from . import config
from .config import get_config

__version__ = importlib.metadata.version("argus-sim")


def get_logger(name: str) -> logging.Logger:
    """Return the logger ``name``, with an INFO-level console handler added on first use.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger.

    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = False
        logger.setLevel(("INFO"))
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")
        console.setFormatter(formatter)
    return logger


log = get_logger(__name__)


c = config.get_config()
if c is None:
    log.info("No configuration file found, proceeding with defaults.")
    c = config.Config(sim_version=__version__)

from .ab_system import ABPhot  # noqa: E402
from .grid import CradlePointing  # noqa: E402
from .photon_budget import NoiseBudget  # noqa: E402
from .psf import DeliveredPSF, GaussianPSF  # noqa: E402
from .throughput import SystemThroughput  # noqa: E402
from .spectral_sky import SpectralSky  # noqa: E402
from .survey import Survey  # noqa: E402
