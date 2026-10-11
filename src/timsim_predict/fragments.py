"""``timsim-fragments`` — predicted fragment-ion intensities, per precursor. MEASUREMENT.

The fragment half of the feature space, as a standalone artifact rather than something buried in the
render: a predicted spectral library. It is a *measurement* (the intensities depend on the collision
energy, an instrument setting), but emitting it as its own table decouples "what fragments, and how
strongly" from "how a given acquisition assembles them into a .d".

# The indexing is the dangerous part, so it is not reasoned about — it is pinned

The model emits a Prosit-layout ``(29, 2, 3)`` tensor (fragment position × ion type × charge). One
transposed axis and every intensity pattern is silently, catastrophically wrong.

There are TWO different flat-174 serialisations in the codebase — ``flatten_prosit_array`` is
charge-major (``[y_c1(29), b_c1(29), y_c2(29)…]``) while a training-target helper is ordinal-major —
so a "flatten then decode a slot" approach is a trap: pick the wrong pair and every spectrum is wrong.
This module therefore does **not** flatten at all. It decodes the ``(29,2,3)`` tensor **directly**,
which is unambiguous, and takes the one axis fact it needs — that axis-2 index 0 is a *y* ion and
index 1 is a *b* ion — from ``flatten_prosit_array``'s own source (``array[:, 0, c]`` is y,
``array[:, 1, c]`` is b), not from reasoning. ``tests/test_fragment_decode.py`` pins that fact against
that function, so an axis mistake fails a test rather than corrupting a benchmark.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

import re

import numpy as np
import pandas as pd
import pyarrow as pa
from timsim_predict._models import provenance as _provenance
import pyarrow.parquet as pq

# Authoritative from flatten_prosit_array's source: axis-2 index 0 is a y ion, index 1 is a b ion.
# Pinned by tests/test_fragment_decode.py against that function.
_AXIS2_ION = {0: "y", 1: "b"}


def decode_tensor(pred_3d, floor: float):
    """Yield ``(ion_type, ordinal, charge, intensity)`` for the present, above-floor fragments of one
    precursor's ``(29, 2, 3)`` prediction. Prosit marks structurally-absent slots with -1, so the
    ``> floor`` test drops both those and the sub-floor peaks.
    """
    for k in range(pred_3d.shape[0]):          # position -> ordinal
        for t in range(pred_3d.shape[1]):      # 0 = y, 1 = b
            for c in range(pred_3d.shape[2]):  # charge - 1
                v = float(pred_3d[k, t, c])
                if v > floor:
                    yield _AXIS2_ION[t], k + 1, c + 1, v


_KOINA_ANN = re.compile(rb"([yb])(\d+)\+(\d+)")


def _predict_tensors_koina(sequences, charges, collision_energies, name: str):
    """Predict ``(n,29,2,3)`` fragment intensity tensors via a Koina model (e.g. Prosit_2020_intensity_HCD
    for Orbitrap-HCD Astral).

    Koina returns intensities in **long format** — one row per fragment with an ``annotation`` (e.g.
    ``b'y1+1'``, ``b'b2+1'``), an ``mz``, and a scalar ``intensities`` — NOT a flat 174-vector, so we
    parse each annotation into its ``(position, ion-type, charge)`` slot rather than assuming an order
    (axis-2: y→0, b→1; charge k→index k-1; position p→index p-1). The output DataFrame's index maps back
    to the input row. Absent fragments stay 0 (dropped downstream by the intensity floor).
    """
    import pandas as pd
    from pepdl.koina_models.access_models import ModelFromKoina

    n = len(sequences)
    df = pd.DataFrame(
        {
            "peptide_sequences": [str(s) for s in sequences],
            "precursor_charges": [int(c) for c in charges],
            "collision_energies": [float(e) for e in collision_energies],
        }
    )
    out = ModelFromKoina(model_name=name).predict(df)
    for col in ("annotation", "intensities"):
        if col not in out.columns:
            raise ValueError(f"Koina model {name!r} returned no {col!r} column (got {list(out.columns)})")

    pred = np.zeros((n, 29, 2, 3), dtype=np.float32)
    have_seq = "peptide_sequences" in out.columns
    have_chg = "precursor_charges" in out.columns
    matched = skipped = 0
    for idx, grp in out.groupby(level=0):
        ii = int(idx)
        if not (0 <= ii < n):
            raise ValueError(f"Koina output row index {ii} outside input range 0..{n - 1}")
        # Guard against Koina renumbering/reordering the index (which would silently misassign fragments
        # to the wrong peptide): verify this group's echoed sequence/charge match the input at row ii.
        if have_seq and str(grp["peptide_sequences"].iloc[0]) != str(sequences[ii]):
            raise ValueError(
                f"Koina row {ii} sequence {grp['peptide_sequences'].iloc[0]!r} != input "
                f"{sequences[ii]!r} — the output index does not map to the input row"
            )
        if have_chg and int(grp["precursor_charges"].iloc[0]) != int(charges[ii]):
            raise ValueError(f"Koina row {ii} charge does not match input — index does not map to input row")
        for ann, inten in zip(grp["annotation"].to_numpy(), grp["intensities"].to_numpy()):
            a = ann if isinstance(ann, (bytes, bytearray)) else str(ann).encode()
            # fullmatch: a neutral-loss annotation like b"y1+1-NH3" must NOT be placed in the plain y1,+1
            # slot. Prosit HCD emits only bare b/y here; anything else is skipped and counted.
            mo = _KOINA_ANN.fullmatch(a)
            if mo is None:
                skipped += 1
                continue
            it, pos, ch = mo.group(1), int(mo.group(2)), int(mo.group(3))
            if 1 <= pos <= 29 and 1 <= ch <= 3:
                pred[ii, pos - 1, 0 if it == b"y" else 1, ch - 1] = max(0.0, float(inten))
                matched += 1
            else:
                skipped += 1
    if matched == 0 and skipped > 0:
        raise ValueError(
            f"Koina model {name!r}: all {skipped} fragment annotations unparsed — output schema changed?"
        )
    return pred


#: The instrument the local model is told it is predicting for. The local model is the timsTOF
#: model (fine-tuned on timsTOF spectra labelled "timstof"); without an instrument it falls back to
#: "unknown". On held-out measured timsTOF spectra: median spectral angle 0.834 unknown, 0.857
#: timstof (pepdl#3). Recorded in the artifact as ``timsim.fragments.instrument``.
LOCAL_INSTRUMENT = "timstof"


@lru_cache(maxsize=1)
def _local_predictor():
    """The local intensity predictor, constructed ONCE.

    Predicting in slices calls this per slice, and constructing the predictor loads model weights —
    doing that per slice would trade the memory blow-up for a runtime one.
    """
    from pepdl.intensity.predictors import DeepPeptideIntensityPredictor
    return DeepPeptideIntensityPredictor(instrument=LOCAL_INSTRUMENT)


def predict_tensors(sequences, charges, collision_energies, model: Optional[str] = None):
    """Predict per-precursor ``(29,2,3)`` intensity tensors with the resolved intensity model.
    Returns ``(array[n,29,2,3], provenance)``."""
    from timsim_predict._models import resolve

    kind, name = resolve("fragments", model)
    if kind == "koina":
        return _predict_tensors_koina(sequences, charges, collision_energies, name), f"koina:{name}"

    predictor = _local_predictor()
    pred = predictor.predict_intensities(
        sequences=list(sequences),
        charges=[int(c) for c in charges],
        collision_energies=[float(e) for e in collision_energies],
    )
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 4 or pred.shape[1:] != (29, 2, 3):
        raise ValueError(f"expected model output (n, 29, 2, 3), got {pred.shape}")
    return pred, name


def fragment_schema(prov: str, collision_energy: float, ce_source: Optional[str] = None) -> pa.Schema:
    """The ``fragment_intensities`` (measurement) schema, stamped with model + CE provenance.

    ``ce_source`` is set only on the per-precursor CE path; leaving it ``None`` reproduces the
    single-CE metadata byte-for-byte, so a default run's artifact is unchanged.
    """
    meta = {
        "timsim.table": "fragment_intensities",
        "timsim.schema_version": "2.0",
        "timsim.axis": "measurement",
        "timsim.producer": "timsim-fragments",
        "timsim.fragments.model": prov,
        "timsim.fragments.collision_energy": repr(float(collision_energy)),
    }
    if not prov.startswith("koina:"):  # a Koina model has no instrument input
        meta["timsim.fragments.instrument"] = LOCAL_INSTRUMENT
    if ce_source is not None:
        meta["timsim.fragments.collision_energy"] = "per-precursor"
        meta["timsim.fragments.collision_energy_source"] = ce_source
    return pa.schema(
        [
            pa.field("precursor_id", pa.uint64(), nullable=False),
            pa.field("ion_type", pa.string(), nullable=False),
            pa.field("ordinal", pa.uint16(), nullable=False),
            pa.field("frag_charge", pa.uint8(), nullable=False),
            pa.field("intensity", pa.float32(), nullable=False),
        ],
        metadata=meta,
    )


def _per_key_collision_energies(precursors: pd.DataFrame, keys: pd.DataFrame, tol: float):
    """Collapse a per-PRECURSOR collision energy onto the per-``(sequence, charge)`` prediction keys.

    The predictor deduplicates on ``(sequence, charge)`` because two precursors that share both predict
    identically — but that is only true if they are also fragmented at the same energy. Both CE models
    make it true by construction: CE is a function of the ion's mobility scan (dda-PASEF ramp) or of the
    window that isolates it (dia-PASEF table), and both the scan and the m/z are themselves functions of
    ``(annotated sequence, charge)``. So the invariant is *checked*, not assumed: a key whose precursors
    disagree by more than ``tol`` eV means the assumption has broken somewhere upstream, and the run is
    refused rather than silently predicting one arbitrary member's spectrum for all of them.

    Returns the per-key CE array, aligned to ``keys``' row order.
    """
    g = precursors.groupby(["sequence", "charge"], observed=True)["collision_energy"]
    span = (g.max() - g.min())
    worst = float(span.max()) if len(span) else 0.0
    if worst > tol:
        bad = span[span > tol]
        ex = [(s, int(c), float(v)) for (s, c), v in bad.head(5).items()]
        raise ValueError(
            f"{len(bad)} of {len(span)} (sequence, charge) prediction keys carry MORE THAN ONE "
            f"collision energy (worst spread {worst:.6f} eV > --ce-tolerance {tol}). The predictor "
            f"deduplicates on (sequence, charge), so it can only honour one CE per key. Examples "
            f"(sequence, charge, eV spread): {ex}. Either raise --ce-tolerance if this spread is "
            f"acceptable, or fix the upstream CE table."
        )
    ce = g.mean()
    out = ce.reindex(pd.MultiIndex.from_arrays([keys["sequence"], keys["charge"]]))
    if out.isna().any():
        raise ValueError("internal: a prediction key has no collision energy")
    return out.to_numpy(dtype=float), worst


def predict_fragment_batches(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    chunk: int = 2_000_000,
    predict_chunk: int = 200_000,
    verbose: bool = True,
    ce_source: Optional[str] = None,
    ce_tolerance: float = 1e-6,
) -> tuple[str, pa.Schema, "Iterator[pa.RecordBatch]"]:
    """Streaming core: ``(provenance, schema, generator of RecordBatch)``.

    ``precursors`` must have ``precursor_id, sequence, charge`` (the sequence being the modform's
    [UNIMOD]-annotated sequence, so a modified peptide fragments as modified). Each distinct
    ``(sequence, charge)`` is decoded ONCE and fanned out over the precursors that share it, emitted in
    row-groups of ``chunk`` fragments. Peak memory is one chunk — not the full ~n_precursors×54-row
    frame, which at scale cost ~17 GB (and a slow per-row ``.iloc`` loop). Rows above ``floor``;
    structurally-absent slots (Prosit marks them -1) are dropped.
    """
    if predict_chunk <= 0:
        # A negative step makes range() yield nothing, so the function would return an EMPTY
        # fragment library and exit 0 -- a wrong answer that looks like a right one.
        raise ValueError(f"predict_chunk must be positive, got {predict_chunk}")

    need = {"precursor_id", "sequence", "charge"}
    missing = need - set(precursors.columns)
    if missing:
        raise ValueError(f"precursors is missing columns {sorted(missing)}")

    # Positional isomers with the same annotated sequence + charge predict identically; dedup.
    keys = precursors[["sequence", "charge"]].drop_duplicates().reset_index(drop=True)
    if verbose:
        print("  timsim-fragments")
        print(f"    precursors        : {len(precursors):,}")
        print(f"    distinct (seq,z)  : {len(keys):,}")

    # DEFAULT: a SINGLE collision energy for every precursor. This is correct for a no-IMS instrument
    # (Astral/Orbitrap DIA sets one NCE for the whole run), which is what the Koina HCD path serves. It
    # is NOT correct for the timsTOF, where CE is per-ion: dda-PASEF ramps it with the mobility scan,
    # and dia-PASEF reads it out of the method's own window table (20.00-58.12 eV on the benchmark
    # reference, against which a flat 25.0 is off by -33.1/+5.0 eV). `timsim-frag-ce` produces that
    # per-precursor table; pass it as --collision-energies and each prediction key is predicted at its
    # own energy. The prediction VOLUME is unchanged: CE is a function of (annotated sequence, charge)
    # through mz and CCS, so the dedup keys do not split (that invariant is checked, not assumed —
    # see _per_key_collision_energies).
    if "collision_energy" in precursors.columns:
        per_key_ce, worst_spread = _per_key_collision_energies(precursors, keys, ce_tolerance)
        if verbose:
            print(f"    collision energy  : per-precursor ({ce_source or 'supplied'}) — "
                  f"{per_key_ce.min():.3f}-{per_key_ce.max():.3f} eV over the keys, "
                  f"worst within-key spread {worst_spread:g} eV")
        key_ce = np.asarray(per_key_ce, dtype=float)
        schema_ce_source = ce_source or "per-precursor"
    else:
        if verbose:
            print(f"    collision energy  : {collision_energy}")
        key_ce = np.full(len(keys), float(collision_energy))
        schema_ce_source = None

    # Provenance WITHOUT calling the model, so the schema exists before the first slice is predicted.
    prov = _provenance("fragments", model)
    schema = (fragment_schema(prov, collision_energy, ce_source=schema_ce_source)
              if schema_ce_source else fragment_schema(prov, collision_energy))

    # key -> the precursor_ids sharing it, built with numpy .values (no pandas scalar indexing in the
    # hot path — that indexing, not the model, was the bulk of the old runtime).
    key2pids: dict = defaultdict(list)
    for pid, s, c in zip(
        precursors["precursor_id"].values, precursors["sequence"].values, precursors["charge"].values
    ):
        key2pids[(s, int(c))].append(int(pid))
    key_tuples = list(zip(keys["sequence"].values, keys["charge"].astype(int).values))

    def batches():
        bp, bi, bo, bf, bv = [], [], [], [], []
        total = 0
        # Predict in SLICES of keys. Predicting every key up front materialised an
        # (n_keys, 29, 2, 3) float32 tensor before a single batch was emitted -- 6.3 GB at 9.0M
        # keys -- and on the Koina path a second full copy in long format on top, which OOM-killed a
        # full human digest at 85% on a 31 GB box. The OUTPUT was already streamed; the INPUT was
        # not.
        #
        # This removes the DOMINANT term; it does not make the whole function slice-bounded.
        # `precursors`, `keys`, `key_ce`, `key_tuples` and above all `key2pids` (a dict holding a
        # Python list of ids for EVERY precursor) stay whole-dataset, and one very high-fanout key
        # can push the accumulators past `chunk`, since a flush only happens between keys.
        for start in range(0, len(key_tuples), predict_chunk):
            stop = min(start + predict_chunk, len(key_tuples))
            tensors, got_prov = predict_tensors(
                keys["sequence"].values[start:stop], keys["charge"].values[start:stop],
                key_ce[start:stop], model=model)
            if got_prov != prov:
                # The schema is already written with `prov`; a mismatch would mislabel every
                # artifact produced from here on.
                raise RuntimeError(
                    f"model provenance changed mid-prediction: schema says {prov!r}, "
                    f"slice starting at {start} reports {got_prov!r}")
            for i, key in enumerate(key_tuples[start:stop]):
                frags = list(decode_tensor(tensors[i], floor))  # decode this key once
                if not frags:
                    continue
                for pid in key2pids.get(key, ()):
                    for it, ordinal, fc, v in frags:
                        bp.append(pid); bi.append(it); bo.append(ordinal); bf.append(fc); bv.append(v)
                if len(bp) >= chunk:
                    yield pa.record_batch(
                        [pa.array(bp, pa.uint64()), pa.array(bi, pa.string()), pa.array(bo, pa.uint16()),
                         pa.array(bf, pa.uint8()), pa.array(bv, pa.float32())], schema=schema)
                    total += len(bp)
                    bp, bi, bo, bf, bv = [], [], [], [], []
            del tensors      # drop the slice before the next one is predicted, not within it
        if bp:
            yield pa.record_batch(
                [pa.array(bp, pa.uint64()), pa.array(bi, pa.string()), pa.array(bo, pa.uint16()),
                 pa.array(bf, pa.uint8()), pa.array(bv, pa.float32())], schema=schema)
            total += len(bp)
        if verbose:
            print(f"    model             : {prov}")
            print(f"    fragment rows      : {total:,}  (above floor {floor:g})")

    return prov, schema, batches()


def predict_fragments(
    precursors: pd.DataFrame,
    collision_energy: float,
    floor: float = 1e-3,
    model: Optional[str] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Return ``(fragment_intensities frame, provenance)`` — the in-memory convenience wrapper around
    :func:`predict_fragment_batches`. Materialises the whole frame, so for large inputs prefer the
    streaming batch generator (that is what the CLI uses)."""
    prov, schema, batches = predict_fragment_batches(
        precursors, collision_energy, floor=floor, model=model, verbose=verbose
    )
    table = pa.Table.from_batches(list(batches), schema=schema)
    return table.to_pandas(), prov


def main(argv=None) -> int:
    from timsim_predict import _identity

    if _identity.handle(argv, "timsim-fragments"):  # what the flow fingerprints; see _identity
        return 0
    ap = argparse.ArgumentParser(
        prog="timsim-fragments",
        description="precursors -> predicted fragment intensities (measurement; spectral library)",
    )
    ap.add_argument("--precursors", required=True, type=Path,
                    help="either a pre-joined (precursor_id, sequence, charge) table, OR the "
                         "timsim-precursors output (precursor_id, peptide_id, charge) when --peptides "
                         "is also given")
    ap.add_argument("--peptides", type=Path, default=None,
                    help="peptides.parquet (peptide_id -> sequence). Legacy convenience: builds the input "
                         "by joining --precursors to the BARE peptide sequence (a modified precursor then "
                         "fragments as unmodified). Prefer `timsim-frag-input`, which freezes the "
                         "[UNIMOD]-annotated modform sequence; then pass its output as --precursors.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--collision-energy", type=float, required=True,
                    help="RAW normalized collision energy (~20-45 NCE), NOT the /100-encoded value "
                         "the .d stores. This artifact is the pre-acquisition model prediction; the "
                         "render applies per-run CE calibration and down-sampling on top.")
    ap.add_argument("--collision-energies", type=Path, default=None,
                    help="OPTIONAL timsim-frag-ce artifact (precursor_id, collision_energy). Predicts "
                         "each precursor at ITS OWN energy instead of the flat --collision-energy — "
                         "which is what a timsTOF actually does (dda-PASEF ramps CE with the mobility "
                         "scan; dia-PASEF reads it from the method's window table). Omit and this tool "
                         "behaves byte-for-byte as before.")
    ap.add_argument("--ce-tolerance", type=float, default=1e-6,
                    help="--collision-energies only: the largest collision-energy spread (eV) tolerated "
                         "within one (sequence, charge) prediction key before the run is refused. The "
                         "predictor can honour exactly one CE per key, so a spread above this would be "
                         "silently discarded.")
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--predict-chunk", type=int, default=200_000,
                    help="distinct (sequence, charge) keys sent to the model per call (default "
                         "200000). The predictor runs in slices so peak memory is one slice rather "
                         "than the whole digest: predicting 9.0M keys at once needs a 6.3 GB tensor "
                         "before any output row is written, and on the Koina path a second full copy "
                         "in long format, which OOM-kills a full human digest on a 31 GB box.")
    ap.add_argument("--model", default=None,
                    help="intensity model spec: omit for our default, or 'koina:<name>' for a Koina "
                         "model. See timsim_predict._models.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.peptides is not None:
        # Build the (precursor_id, sequence, charge) fragment-prediction input by joining the precursors
        # table to peptide sequences. MEMORY: a PTM-heavy proteome can have 100M+ precursors, so we never
        # materialise one Python string per row — map peptide_id -> a categorical CODE (int32), then wrap
        # as a Categorical (codes + a small unique-sequence dictionary). 185M rows then cost ~0.7 GB of
        # codes, not ~9 GB of string objects. The predictor dedups (sequence, charge) internally.
        prec = pd.read_parquet(a.precursors, columns=["precursor_id", "peptide_id", "charge"])
        pep = pd.read_parquet(a.peptides, columns=["peptide_id", "sequence"])
        cat = pd.Categorical(pep["sequence"])
        code_of = dict(zip(pep["peptide_id"].to_numpy(), cat.codes.astype("int32")))
        codes = prec["peptide_id"].map(code_of)
        # Drop BOTH unmapped peptide_ids (map -> NaN) AND null sequences (Categorical encodes them as
        # code -1): notna() alone keeps the -1, which from_codes would turn back into a null sequence.
        keep = codes.notna() & (codes >= 0)
        prec = pd.DataFrame({
            "precursor_id": prec.loc[keep, "precursor_id"].to_numpy(),
            "sequence": pd.Categorical.from_codes(
                codes[keep].astype("int32").to_numpy(), categories=cat.categories
            ),
            "charge": prec.loc[keep, "charge"].to_numpy(),
        })
    else:
        prec = pd.read_parquet(a.precursors)

    # Per-precursor collision energy. Attached ONLY when asked for: any `collision_energy` column that
    # rode in on the precursors artifact is dropped otherwise, so the flat path cannot be switched on by
    # accident by an upstream schema change.
    ce_source = None
    if a.collision_energies is None:
        prec = prec.drop(columns=["collision_energy"], errors="ignore")
    else:
        ce = pd.read_parquet(a.collision_energies, columns=["precursor_id", "collision_energy"])
        ce_source = str(a.collision_energies)
        n_before = len(prec)
        prec = prec.merge(ce, on="precursor_id", how="left", validate="many_to_one")
        holes = int(prec["collision_energy"].isna().sum())
        if holes:
            raise SystemExit(
                f"--collision-energies covers only {n_before - holes:,} of {n_before:,} precursors "
                f"({holes:,} missing). A hole would be silently predicted at the flat "
                f"--collision-energy, mixing two CE models in one library. Regenerate the CE table "
                f"from the same precursors artifact."
            )
    a.out.parent.mkdir(parents=True, exist_ok=True)

    # Stream row-groups straight to the file: the full fragment frame is never resident.
    _prov, schema, batches = predict_fragment_batches(
        prec, a.collision_energy, floor=a.floor, model=a.model, verbose=not a.quiet,
        ce_source=ce_source, ce_tolerance=a.ce_tolerance,
        predict_chunk=a.predict_chunk,
    )
    # Write to a temp file and rename only on success. Prediction now happens WHILE the generator is
    # consumed, not before it is returned, so a model or network failure lands mid-write -- and the
    # writer's close() would leave a structurally valid Parquet file holding a PREFIX of the library.
    # A downstream step cannot tell that from a complete one. The rename is atomic within a
    # directory, so the output path only ever exists finished.
    tmp = a.out.with_name(a.out.name + ".partial")
    writer = pq.ParquetWriter(tmp, schema)
    try:
        for batch in batches:
            writer.write_table(pa.Table.from_batches([batch], schema=schema))
        writer.close()
    except BaseException:
        try:
            writer.close()
        finally:
            tmp.unlink(missing_ok=True)
        raise
    tmp.replace(a.out)
    if not a.quiet:
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
