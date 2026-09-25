"""Command line: ``python -m argus_sim.lightcurve build RUN_DIR --out STORE_DIR`` or ``... visits STORE RA DEC``."""

import argparse

from .run import SurveyRun
from .store import build_index


def main(argv=None) -> None:
    """Build a visit store, or print one position's visit table."""
    ap = argparse.ArgumentParser(prog="python -m argus_sim.lightcurve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build",
        help="build a run's visit store with its manifest, ready for upload, or resume an interrupted build",
    )
    b.add_argument("run_dir")
    b.add_argument("--out", "--index-dir", dest="out", help="store directory (default RUN_DIR/lightcurve)")
    b.add_argument("--workers", type=int)
    b.add_argument(
        "--max-memory-gb", type=float, default=4.0, help="memory for the rows sorted in one pass, GB (default 4)"
    )
    b.add_argument(
        "--check-every",
        type=int,
        default=500,
        help="compare the recomputed masks bit by bit with the run on every n-th ratchet (default 500)",
    )
    b.add_argument("--limit-ratchets", type=int, help="index only the first n ratchets (timing trials)")
    b.add_argument(
        "--tanseg-range",
        type=int,
        nargs=2,
        metavar=("LO", "HI"),
        help="index only the tiles with LO <= tanseg_id <= HI",
    )
    b.add_argument("--keep-sorted", action="store_true", help="keep the full-precision sorted rows in STORE/build")
    b.add_argument(
        "--sorted-from",
        metavar="V1_STORE",
        help="take the sorted rows from a format-1 store of the same run instead of sorting the epoch records",
    )
    v = sub.add_parser("visits", help="print the visit table of one position")
    v.add_argument("store", help="store directory, run directory or s3:// URI")
    v.add_argument("ra", type=float)
    v.add_argument("dec", type=float)
    v.add_argument("-o", "--output", help="write the table to this file (format from the extension)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build_index(
            a.run_dir,
            index_dir=a.out,
            workers=a.workers,
            max_memory_gb=a.max_memory_gb,
            check_every=a.check_every,
            limit_ratchets=a.limit_ratchets,
            tanseg_range=a.tanseg_range,
            keep_sorted=a.keep_sorted,
            sorted_from=a.sorted_from,
        )
    else:
        vis = SurveyRun(a.store).visits(a.ra, a.dec)
        if a.output:
            vis.write(a.output, overwrite=True)
        else:
            print(vis)


if __name__ == "__main__":
    main()
