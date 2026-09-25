************
Installation
************

ArgusSim needs Python 3.10 or later.

Using ``pip``
=============

From a clone of the repository::

    pip install .

The base install does not include CuPy and needs no GPU. Reading a visit store and simulating light curves run on
the CPU, and so does the survey simulator, which uses NumPy when CuPy or a CUDA device is missing. Full survey
simulations are much faster on an NVIDIA GPU. The optional CuPy dependency is installed with::

    pip install ".[gpu]"

Using ``uv``
============

For development, `uv <https://docs.astral.sh/uv/>`_ creates a virtual environment in ``.venv`` with the package,
its dependencies and the development tools::

    uv sync
    uv sync --extra gpu    # with CuPy

Commands then run in that environment with ``uv run``, for example ``uv run pytest``.
