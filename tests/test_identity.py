"""``--identity`` must change whenever the code or weights that decide a tool's output change.

The flow fingerprints these tools by this answer (the launcher on PATH is a pip stub that never
changes), so an identity that misses a change means a cache silently serves stale predictions.
"""
import json
import shutil

import pytest

from timsim_predict import _identity


def _with_fake_packages(monkeypatch, tmp_path):
    """Point every package at a private copy of a tiny tree, returning those roots."""
    roots = {}
    for name in _identity.PACKAGES:
        root = tmp_path / name
        (root / "sub").mkdir(parents=True)
        (root / "__init__.py").write_text("x = 1\n")
        (root / "sub" / "m.py").write_text("y = 2\n")
        roots[name] = root
    hub = roots["pepdl"] / "pretrained"
    hub.mkdir()
    (hub / "hub.py").write_text(
        'MODEL_VERSION = "0.5.0"\n'
        'MODELS = {"ccs/best_model.pt": {"filename": "c.pt", "sha256": "aa"}}\n')
    monkeypatch.setattr(_identity, "_package_dir", lambda n: roots[n])
    return roots


def test_identity_is_deterministic_and_json(capsys):
    assert _identity.handle(["--identity"], "timsim-ccs")
    first = capsys.readouterr().out
    assert _identity.handle(["--out", "x", "--identity"], "timsim-ccs")
    assert capsys.readouterr().out == first
    got = json.loads(first)
    assert got["schema"] == _identity.SCHEMA and got["tool"] == "timsim-ccs"
    assert set(got["packages"]) == set(_identity.PACKAGES)
    assert "ccs/best_model.pt" in got["models"]["sha256"]


def test_no_flag_means_no_output(capsys):
    assert not _identity.handle(["--out", "x"], "timsim-ccs")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("package", _identity.PACKAGES)
def test_a_one_byte_change_in_any_package_changes_the_identity(monkeypatch, tmp_path, package):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    before = _identity.identity("timsim-ccs")
    (roots[package] / "sub" / "m.py").write_text("y = 3\n")
    assert _identity.identity("timsim-ccs") != before


def test_adding_or_renaming_a_file_changes_the_identity(monkeypatch, tmp_path):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    before = _identity.identity("timsim-ccs")
    shutil.move(roots["pepdl"] / "sub" / "m.py", roots["pepdl"] / "sub" / "n.py")
    renamed = _identity.identity("timsim-ccs")
    assert renamed != before
    (roots["pepdl"] / "sub" / "extra.bin").write_bytes(b"w")
    assert _identity.identity("timsim-ccs") != renamed


def test_bytecode_caches_do_not_change_the_identity(monkeypatch, tmp_path):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    before = _identity.identity("timsim-ccs")
    (roots["pepdl"] / "__pycache__").mkdir()
    (roots["pepdl"] / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"\0")
    assert _identity.identity("timsim-ccs") == before


def test_a_new_checkpoint_sha_changes_the_identity(monkeypatch, tmp_path):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    before = _identity.identity("timsim-ccs")
    hub = roots["pepdl"] / "pretrained" / "hub.py"
    hub.write_text(hub.read_text().replace('"aa"', '"bb"'))
    after = _identity.identity("timsim-ccs")
    assert after["models"]["sha256"]["ccs/best_model.pt"] == "bb" and after != before


def test_the_repo_checkpoints_fallback_is_covered(monkeypatch, tmp_path):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    ck = roots["pepdl"].parent.parent / "checkpoints"
    assert _identity.identity("timsim-ccs")["repo_checkpoints_sha256"] is None or ck.is_dir()
    ck.mkdir(exist_ok=True)
    (ck / "w.pt").write_bytes(b"1")
    one = _identity.identity("timsim-ccs")
    (ck / "w.pt").write_bytes(b"2")
    assert _identity.identity("timsim-ccs") != one


def test_a_missing_package_is_an_error_not_a_partial_identity(monkeypatch, tmp_path):
    roots = _with_fake_packages(monkeypatch, tmp_path)
    monkeypatch.setattr(_identity, "_package_dir", lambda n: None if n == "mscorepy" else roots[n])
    with pytest.raises(RuntimeError, match="mscorepy"):
        _identity.identity("timsim-ccs")


@pytest.mark.parametrize("module, tool", [("ccs", "timsim-ccs"), ("rt", "timsim-rt"),
                                          ("fragments", "timsim-fragments")])
def test_each_cli_answers_identity_without_its_required_args(capsys, module, tool):
    import importlib

    main = importlib.import_module(f"timsim_predict.{module}").main
    assert main(["--identity"]) == 0
    assert json.loads(capsys.readouterr().out)["tool"] == tool
