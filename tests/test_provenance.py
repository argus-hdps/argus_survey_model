import json
import subprocess

from argus_sim.provenance import REPO_ROOT, git_state, provenance_record, write_provenance


def _git(*args):
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def test_repo_root_is_the_package_checkout():
    assert (REPO_ROOT / "src" / "argus_sim" / "provenance.py").is_file()


def test_git_hash_matches_head():
    state = git_state()
    assert state["git_hash"] == _git("rev-parse", "HEAD")
    assert state["git_hash_short"] == state["git_hash"][:7]


def test_dirty_flag_matches_tracked_changes():
    expected = bool(_git("status", "--porcelain", "--untracked-files=no"))
    assert git_state()["git_dirty"] is expected


def test_untracked_files_do_not_dirty(tmp_path, monkeypatch):
    scratch = REPO_ROOT / "tests" / "_untracked_provenance_probe.tmp"
    scratch.write_text("x")
    try:
        assert git_state()["git_dirty"] is bool(_git("status", "--porcelain", "--untracked-files=no"))
    finally:
        scratch.unlink()


def test_record_carries_identity_and_extras():
    rec = provenance_record("unit", {"n": 3})
    for key in ("product", "argus_sim_version", "git_hash", "git_dirty", "timestamp_utc", "argv", "python"):
        assert key in rec
    assert rec["product"] == "unit" and rec["n"] == 3


def test_write_provenance_round_trip(tmp_path):
    path = write_provenance(tmp_path / "product", "unit")
    assert path.name == "provenance.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["git_hash"] == git_state()["git_hash"]
    assert isinstance(on_disk["git_dirty"], bool)


def test_record_names_the_ring_layout():
    from argus_sim import c
    from argus_sim.ring_layout import DEFAULT_LAYOUT

    rec = provenance_record("unit")
    assert rec["tiling"] == c.packing_strategy.tiling
    assert rec["ring_layout"] == (c.packing_strategy.layout_file or DEFAULT_LAYOUT)


def test_product_ratchet_len_reads_parquet_metadata(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from argus_sim.provenance import product_ratchet_len

    assert product_ratchet_len(tmp_path) is None
    d = tmp_path / "ratchets" / "20260615"
    d.mkdir(parents=True)
    t = pa.Table.from_pandas(pd.DataFrame({"healpix": [1, 2]}))
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"n_epochs": b"15", b"epoch_exptime_s": b"60.0"})
    pq.write_table(t, d / "ratchet_0000001.parquet")
    assert product_ratchet_len(tmp_path) == 15.0
    (tmp_path / "provenance.json").write_text('{"ratchet_len": 30.0}')
    assert product_ratchet_len(tmp_path) == 30.0
