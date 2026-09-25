"""Shared fixtures.

Tests marked ``needs_lc_run`` use the survey run named by the environment variable ARGUS_LC_RUN, for example the
output of ``asim survey --start 2026-06-14T12:00 --n-nights 1 --seed 1 --no-weather --outdir RUN``.  They skip when
the variable is unset, when it names no run, and when the run was made with a layout, site or throughputs other than
this package's.
"""

import json
import os

import pytest

LC_RUN = os.path.expanduser(os.environ.get("ARGUS_LC_RUN", ""))


def pytest_configure(config):
    """Register the marker."""
    config.addinivalue_line("markers", "needs_lc_run: needs the survey run named by ARGUS_LC_RUN")


def _lc_run_problem():
    """Why the ARGUS_LC_RUN run cannot be used, or None when it can."""
    if not LC_RUN:
        return "ARGUS_LC_RUN is not set"
    if not os.path.isdir(os.path.join(LC_RUN, "epochs")):
        return f"ARGUS_LC_RUN={LC_RUN} holds no survey run"
    prov_path = os.path.join(LC_RUN, "provenance.json")
    layout = json.load(open(prov_path)).get("ring_layout") if os.path.isfile(prov_path) else None
    if layout is not None:
        from argus_sim.ring_layout import LAYOUTS

        if layout not in LAYOUTS and not os.path.isfile(layout):
            return f"ARGUS_LC_RUN was made with layout {layout!r}, which this package does not define"
    return None


def pytest_collection_modifyitems(config, items):
    """Give every ``needs_lc_run`` test the run's visit store, so a run that does not match the package skips."""
    for item in items:
        if item.get_closest_marker("needs_lc_run") and "lc_index" not in item.fixturenames:
            item.fixturenames.append("lc_index")


def pytest_runtest_setup(item):
    """Skip ``needs_lc_run`` tests when ARGUS_LC_RUN is unusable."""
    if item.get_closest_marker("needs_lc_run"):
        problem = _lc_run_problem()
        if problem:
            pytest.skip(problem)


@pytest.fixture(scope="session")
def lc_index(tmp_path_factory):
    """Directory of the ARGUS_LC_RUN visit store: the run's own ``lightcurve/`` when built, else one built here.

    Skips when the store builder finds that the run was made with another layout, site or throughputs.
    """
    problem = _lc_run_problem()
    if problem:
        pytest.skip(problem)
    idx = os.path.join(LC_RUN, "lightcurve")
    if os.path.isfile(os.path.join(idx, "index.json")):
        return idx
    from argus_sim.lightcurve import SurveyRun

    idx = str(tmp_path_factory.mktemp("lcindex"))
    try:
        SurveyRun.build_index(LC_RUN, index_dir=idx)
    except RuntimeError as e:
        if "is not the run's" in str(e):
            pytest.skip(f"ARGUS_LC_RUN was made with another configuration than this package's: {e}")
        raise
    return idx
