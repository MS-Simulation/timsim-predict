"""``--identity``: what a prediction tool would compute with, as deterministic JSON.

# Why

The flow (timsim-necro) fingerprints a node by its command string plus a hash of the invoked tool.
For a native binary that hash is the tool. For these tools it is not: ``timsim-ccs`` on PATH is a
six-line pip launcher, and the code and weights that decide the output live in this package,
``pepdl``, ``mscorepy`` (the tokenizer and its vocabulary, in a Rust extension) and the model
checkpoints. A fix in any of them left every fingerprint unchanged, so a cache silently served the
old output. That happened with pepdl's padding-mask fix, which moved 2+ CCS by ~35 Å².

So each tool answers ``--identity`` with everything that determines its output, and the flow hashes
that answer instead of the launcher.

# What is covered

* each package's version and a sha256 over every file in its installed directory (catches editable
  installs and wrapper-only changes; a bundled checkpoint inside pepdl is covered too). Required:
  timsim_predict, pepdl, mscorepy. Optional, when installed: chronologer (the default RT model's
  code) and koinapy (a Koina model's output also depends on its server, which no hash can cover);
* torch's version (numerics), read from metadata without importing torch;
* the model registry: pepdl's ``MODEL_VERSION`` and the pinned sha256 of every downloadable
  checkpoint. ``ensure_model`` verifies a download against that sha256, so the registry entry is
  the identity of the weights that get loaded, plus pepdl's source-checkout ``checkpoints/``
  fallback when it exists. Nothing is downloaded or imported to compute this.

All selectable models are covered, not just the one a command picks: the flow stamps a command
template before its ``--model`` placeholder is filled.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as md
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

SCHEMA = 1
PACKAGES = ("timsim_predict", "pepdl", "mscorepy")
# Code a selectable model runs that lives outside pepdl: Chronologer (the default RT model's
# network, tensorisation and vocabulary) and the Koina client. Hashed when installed, recorded as
# absent otherwise (installs without the local / koina extras).
OPTIONAL = ("chronologer", "koinapy")


def _package_dir(name: str) -> Optional[Path]:
    spec = importlib.util.find_spec(name)  # locates without executing the package
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _tree_sha256(root: Path) -> str:
    """sha256 over (relative path, content) of every file under root, in sorted order."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "absent"


def _model_registry(pepdl_dir: Path) -> dict:
    """``MODEL_VERSION`` and ``MODELS`` from pepdl's hub, read as literals (no import)."""
    tree = ast.parse((pepdl_dir / "pretrained" / "hub.py").read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("MODEL_VERSION", "MODELS"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    if set(found) != {"MODEL_VERSION", "MODELS"}:
        raise RuntimeError("pepdl.pretrained.hub no longer defines MODEL_VERSION and MODELS literally")
    return {"version": found["MODEL_VERSION"],
            "sha256": {k: v["sha256"] for k, v in sorted(found["MODELS"].items())}}


def identity(tool: str) -> dict:
    packages = {}
    for name in PACKAGES:
        d = _package_dir(name)
        if d is None:
            raise RuntimeError(f"{tool} --identity: package {name!r} is not importable")
        packages[name] = {"version": _version(name.replace("_", "-")), "files_sha256": _tree_sha256(d)}
    for name in OPTIONAL:
        d = _package_dir(name)
        packages[name] = None if d is None else {
            "version": _version(name), "files_sha256": _tree_sha256(d)}
    pepdl_dir = _package_dir("pepdl")
    # The intensity loader also looks for weights in a repo-root ``checkpoints/`` (two levels above
    # the package; it exists only in a source checkout), which the package tree does not include.
    repo_checkpoints = pepdl_dir.parent.parent / "checkpoints"
    return {
        "schema": SCHEMA,
        "tool": tool,
        "packages": packages,
        "torch": _version("torch"),
        "models": _model_registry(pepdl_dir),
        "repo_checkpoints_sha256": _tree_sha256(repo_checkpoints) if repo_checkpoints.is_dir() else None,
    }


def handle(argv, tool: str) -> bool:
    """Print the identity and return True if ``--identity`` was asked for; else return False."""
    argv = sys.argv[1:] if argv is None else argv
    if "--identity" not in argv:
        return False
    print(json.dumps(identity(tool), sort_keys=True, separators=(",", ":")))
    return True
