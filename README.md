# ArgusSim

ArgusSim simulates the survey of the Argus Array, 1200 telescopes of 280 mm aperture: where the telescopes point,
what each exposure sees, how deep it goes, and what light curve a source would get. It models the sky background
from the atmosphere, Moon, airglow, zodiacal light and starlight; the throughput of the survey bands; the delivered
PSF with seeing; and the pointing pattern of the array over whole nights and years.

The package has two parts:

- Survey simulation (`argus_sim.Survey`, `argus_sim.survey_driver`) runs the array over a span of nights with
  weather and the Moon, and records every (sky tile, telescope) pair each exposure covered, with its background noise,
  time, seeing and transparency.
- The visit store (`argus_sim.lightcurve`) is a compact, query-ordered copy of a survey run that can be read from a
  local directory or an S3 bucket. It returns every visit of a sky position, per-tile depth maps and nightly
  summaries, and draws simulated photometry for any light-curve model.

Most science users only need the second part and the published visit store of the v5.4 5-year run.

## Installation

ArgusSim needs Python 3.10 or later. From a clone of this repository:

```
pip install .
```

or, for development, with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

The base install does not include CuPy and needs no GPU. Reading a visit store and simulating light curves run on
the CPU, and so does the survey simulator, which uses NumPy when CuPy or a CUDA device is missing. Full survey
simulations are much faster on an NVIDIA GPU: install the optional CuPy dependency with `pip install ".[gpu]"` or
`uv sync --extra gpu`.

## Quick start

Light curves from the v5.4 5-year run:

```python
import numpy as np
from argus_sim.lightcurve import SurveyRun, bin_lightcurve

run = SurveyRun("s3://schmidt-observatory-system/argus/sim/argsim_5yr_beta_rho_uniform_v5.4/")
vis = run.visits(150.0, 2.2, band="g")          # every 60-s slot and telescope that covered the position
print(len(vis), "visits")


def rr_lyrae(mjd, band, P=0.5672, mean=16.5, amp=0.8):
    ph = (mjd / P) % 1.0
    shape = np.where(ph < 0.15, ph / 0.15, 1.0 - (ph - 0.15) / 0.85)
    return mean + amp / 2 - amp * shape


lc = run.inject(vis, rr_lyrae, rng=np.random.default_rng(1))   # one simulated measurement per slot
nightly = bin_lightcurve(lc, by="night")
```

A model is any function of `mjd` and `band` that returns AB magnitudes. `run.depth_map(band, period)` gives the depth
of every sky tile for the whole survey or each half-year, and `run.night_log(ra, dec)` says why a night has no data at
a position (weather, Moon, or outside the footprint). To pass S3 credentials or an endpoint, use
`SurveyRun(uri, storage_options={...})` with the keyword arguments of `pyarrow.fs.S3FileSystem`.

The `g` band is the Argus beta band.

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

## Disclaimers

The PSF follows the delivered image-quality specification, an EE50 diameter of 3.0 pixels at the field corners, so
the simulated depths are conservative. The weather and seeing models draw from long-term statistics for a dark site
and do not reproduce any particular night. The results are intended for science modelling and survey planning and are
not performance guarantees for the instrument.

## Tests

```
uv run pytest
```

The light-curve tests need a small survey run, for example one night made with

```
asim survey --start 2026-06-14T12:00 --n-nights 1 --seed 1 --no-weather --outdir RUN_DIR
```

Set `ARGUS_LC_RUN` to that directory. These tests are skipped when the variable is unset, when it names no run,
or when the run was made with a different layout, site or throughputs from the installed package. The sky-tiling
parity test compares against an `hdps.skymap` source tree named by `HDPS_SKYMAP_REF` and is skipped when that
variable is unset. The provenance tests read the git commit, so they need a git checkout.

## License

MIT.
