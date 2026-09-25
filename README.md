# ArgusSim

ArgusSim simulates the planned survey of the Argus Array, including telescope pointings, error budgets, accumulated sky coverage, and window functions. It models the sky background
from the atmosphere, Moon, airglow, zodiacal light and blended starlight, the throughput of the survey bands, the delivered
PSF with seeing, and the pointing pattern and sky coverage of the Array its planned 5-year survey.

The package has two parts:

- Survey simulation (`argus_sim.Survey`, `argus_sim.survey_driver`) runs the Array over a span of nights with
  weather and the Moon, and records every (sky tile, telescope) pair each exposure covered, with its background noise,
  time span, seeing and transparency.
- The visit store (`argus_sim.lightcurve`) is a compact, query-ordered copy of a survey run that can be read from a
  local directory or an S3 bucket. It returns every visit of a sky position, per-tile depth maps and nightly
  summaries, and draws simulated photometry for any light-curve model.

*Most science users will only need the second part and accompanying visit store (currently version v5.4).

## Installation

ArgusSim needs Python 3.10 or later. From a clone of this repository:

```
pip install .
```

or, for development or with a reproducible lock file, with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

The base install does not include CuPy or require a local GPU. Reading a visit store and light curve simulation runs on
the CPU, and so does the survey simulator, which uses NumPy when CuPy or a CUDA device is missing (though this will be extremely slow for reasonable length survey spans). Full survey
simulations are much faster on an NVIDIA GPU: install the optional CuPy dependency with `pip install ".[gpu]"` or
`uv sync --extra gpu` if you are generating a custom survey run.

## Quick start

Light curves from the v5.4 5-year run:

```python
import numpy as np
from argus_sim.lightcurve import SurveyRun, bin_lightcurve

run = SurveyRun("s3://some-public-bucket/argus/sim/argsim_5yr_beta_rho_uniform_v5.4/")
vis = run.visits(150.0, 2.2, band="g") # every 60-s slot and telescope that covered the position RA, Dec in decimal deg
print(len(vis), "visits")


def rr_lyrae(mjd, band, P=0.5672, mean=16.5, amp=0.8):
    ph = (mjd / P) % 1.0
    shape = np.where(ph < 0.15, ph / 0.15, 1.0 - (ph - 0.15) / 0.85)
    return mean + amp / 2 - amp * shape


lc = run.inject(vis, rr_lyrae, rng=np.random.default_rng(1))   # one simulated measurement per slot
nightly = bin_lightcurve(lc, by="night")
```

A model can be any function of `mjd` and `band` that returns expected AB magnitudes. `run.depth_map(band, period)` gives the depth
of every sky tile for the whole survey or each half-year, and `run.night_log(ra, dec)` says why a night has no data at
a position (weather, Moon, or outside the footprint). To pass S3 credentials or an endpoint, use
`SurveyRun(uri, storage_options={...})` with the keyword arguments of `pyarrow.fs.S3FileSystem`.


## Tutorials

The notebooks in `notebooks/` are written as tutorials:

- `tutorial_survey_data.ipynb`: reading the visit store, depth over time and across the sky, the exposure history of a
  position, and injecting variable-star, supernova and kilonova models (sncosmo, and redback, which is installed
  separately).
- `tutorial_depth_calculation.ipynb`: the 5-sigma depth of one exposure from throughput, sky, PSF and noise, checked
  against the v5.4 run.

## Command line

`asim defaults` writes the default settings to `argussim.toml` in the current directory. On import, ArgusSim reads
the first TOML file in the working directory that parses as a configuration, so editing that file changes the
instrument, site or survey parameters. `asim survey` runs the survey simulator over a span of nights, and
`asim survey --help` lists its options. To build a visit store from a survey run directory:

```
python -m argus_sim.lightcurve build RUN_DIR --out STORE
```

## License

MIT.
