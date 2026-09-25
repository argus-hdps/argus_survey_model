"""Record which ArgusSim commit produced a data product.

Every product directory gets a ``provenance.json`` carrying the package
version, the git commit of the source tree the process imported, whether
that tree had uncommitted changes to tracked files, and the command line.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """Commit hash and dirty flag of the source tree this package was imported from.

    ``git_dirty`` is True when any *tracked* file differs from HEAD; untracked
    files (outputs, scratch scripts) do not count.  All values are None when
    the package is not imported from the top of a git checkout (for example
    when it is installed from a wheel into an environment inside another
    repository).
    """
    top = _git("rev-parse", "--show-toplevel")
    head = _git("rev-parse", "HEAD") if top and Path(top).resolve() == REPO_ROOT else None
    if head is None:
        return {"git_hash": None, "git_hash_short": None, "git_dirty": None, "git_branch": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_hash": head,
        "git_hash_short": head[:7],
        "git_dirty": bool(status),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def layout_state() -> dict[str, Any]:
    """Tiling mode and ring layout the configuration will build, by registered name or path."""
    try:
        from . import c
        from .ring_layout import DEFAULT_LAYOUT
    except Exception:
        return {}
    from .ring_layout import LAYOUTS, table_applies

    name = c.packing_strategy.layout_file or DEFAULT_LAYOUT
    spec = LAYOUTS.get(name)
    table = (
        spec.band_table if spec is not None and spec.band_table and table_applies(c.filter_strategy.options) else None
    )
    return {
        "tiling": c.packing_strategy.tiling,
        "ring_layout": name,
        "band_assignment": f"table:{table}" if table else "strips",
        "coverage": getattr(c.survey, "coverage", "moc"),
    }


def provenance_record(product: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the provenance dictionary for ``product`` without writing it."""
    try:
        from . import __version__
    except Exception:
        __version__ = None
    record: dict[str, Any] = {
        "product": product,
        "argus_sim_version": __version__,
        **git_state(),
        **layout_state(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
    }
    if extra:
        record.update(extra)
    return record


def write_provenance(output_dir: str | Path, product: str, extra: dict[str, Any] | None = None) -> Path:
    """Write ``provenance.json`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "provenance.json"
    with open(path, "w") as f:
        json.dump(provenance_record(product, extra), f, indent=2)
    return path


def product_ratchet_len(product_dir: str | Path) -> float | None:
    """Ratchet length in minutes that a Survey product was run with, or None if it cannot be told.

    Looks, in order, at ``provenance.json`` (``ratchet_len``), ``combined_manifest.json``
    (``config.ratchet_len_min``), and the first per-ratchet parquet's metadata (``n_epochs`` times
    ``epoch_exptime_s``), which every Survey run with grid tables writes.
    """
    product_dir = Path(product_dir)
    prov = product_dir / "provenance.json"
    if prov.is_file():
        val = json.loads(prov.read_text()).get("ratchet_len")
        if val is not None:
            return float(val)
    comb = product_dir / "combined_manifest.json"
    if comb.is_file():
        val = json.loads(comb.read_text()).get("config", {}).get("ratchet_len_min")
        if val is not None:
            return float(val)
    first = next(iter(sorted(product_dir.glob("ratchets/*/ratchet_*.parquet"))), None)
    if first is not None:
        import pyarrow.parquet as pq

        md = {k.decode(): v.decode() for k, v in (pq.read_schema(first).metadata or {}).items()}
        if "n_epochs" in md and "epoch_exptime_s" in md:
            return float(md["n_epochs"]) * float(md["epoch_exptime_s"]) / 60.0
    return None
