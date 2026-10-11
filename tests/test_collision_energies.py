"""The per-precursor collision-energy path of ``timsim-fragments``.

The predictor deduplicates on ``(sequence, charge)`` and can honour exactly ONE energy per key. These
tests pin the three things that follow from that: the collapse is correct when the invariant holds, the
run is REFUSED when it does not (rather than silently predicting one arbitrary member's spectrum for
everyone), and the prediction volume does not grow. No model is loaded — this is the plumbing, not the
network.
"""

import pandas as pd
import pytest

from timsim_predict.fragments import _per_key_collision_energies, fragment_schema


def frame(rows):
    """rows = [(precursor_id, sequence, charge, collision_energy)]"""
    return pd.DataFrame(rows, columns=["precursor_id", "sequence", "charge", "collision_energy"])


def keys_of(df):
    return df[["sequence", "charge"]].drop_duplicates().reset_index(drop=True)


def test_one_ce_per_key_collapses_and_stays_aligned_to_the_key_order():
    df = frame([(1, "PEPTIDEK", 2, 50.05), (2, "PEPTIDEK", 3, 20.0), (3, "ELVISK", 2, 34.10)])
    keys = keys_of(df)
    ce, worst = _per_key_collision_energies(df, keys, tol=1e-6)
    assert worst == 0.0
    # Aligned to `keys`' ROW ORDER, not to a sorted groupby order — a mismatch here would predict every
    # spectrum at another key's energy, which no downstream check would catch.
    assert list(ce) == [50.05, 20.0, 34.10]


def test_positional_isomers_sharing_a_key_share_their_energy():
    # Two precursors, same annotated sequence and charge (the case the dedup exists for): one key, one CE.
    df = frame([(1, "PEPS[UNIMOD:21]TIDEK", 2, 45.19), (2, "PEPS[UNIMOD:21]TIDEK", 2, 45.19)])
    keys = keys_of(df)
    ce, worst = _per_key_collision_energies(df, keys, tol=1e-6)
    assert len(keys) == 1 and len(ce) == 1
    assert ce[0] == 45.19 and worst == 0.0


def test_a_key_with_two_energies_is_refused_not_averaged():
    df = frame([(1, "PEPTIDEK", 2, 50.05), (2, "PEPTIDEK", 2, 51.61)])
    with pytest.raises(ValueError) as e:
        _per_key_collision_energies(df, keys_of(df), tol=1e-6)
    msg = str(e.value)
    assert "MORE THAN ONE" in msg
    # The message must carry the measured spread and a concrete example, or the operator cannot act.
    assert "1.56" in msg and "PEPTIDEK" in msg


def test_the_tolerance_is_the_knob_that_decides():
    df = frame([(1, "PEPTIDEK", 2, 50.05), (2, "PEPTIDEK", 2, 50.05 + 1e-9)])
    # Below tolerance: accepted, and the key gets a single energy.
    ce, worst = _per_key_collision_energies(df, keys_of(df), tol=1e-6)
    assert worst < 1e-6 and len(ce) == 1
    # A tolerance tighter than the spread refuses the same input.
    with pytest.raises(ValueError):
        _per_key_collision_energies(df, keys_of(df), tol=1e-12)


def test_prediction_volume_does_not_grow():
    """Attaching a per-precursor CE must not split the dedup keys — that is the whole point of checking
    the invariant instead of predicting per (sequence, charge, CE)."""
    df = frame([(i, f"PEPTIDE{i % 7}K", 2 + i % 2, 20.0 + (i % 7)) for i in range(500)])
    keys = keys_of(df)
    ce, _ = _per_key_collision_energies(df, keys, tol=1e-6)
    assert len(ce) == len(keys) == len(df[["sequence", "charge"]].drop_duplicates())
    # Splitting on CE too would give the same count here — i.e. no extra model calls.
    assert len(df[["sequence", "charge", "collision_energy"]].drop_duplicates()) == len(keys)


def test_a_categorical_sequence_column_still_aligns():
    """`timsim-fragments --peptides` hands the predictor a pandas Categorical to keep 100M-row inputs in
    memory, so the groupby/reindex must survive that dtype."""
    df = frame([(1, "AAAK", 2, 30.0), (2, "BBBK", 2, 40.0), (3, "AAAK", 3, 50.0)])
    df["sequence"] = pd.Categorical(df["sequence"])
    keys = keys_of(df)
    ce, worst = _per_key_collision_energies(df, keys, tol=1e-6)
    assert worst == 0.0
    assert list(ce) == [30.0, 40.0, 50.0]


def test_schema_metadata_records_which_ce_model_produced_the_library():
    """A library predicted at per-precursor energies must not be indistinguishable from a flat one — the
    two are different measurements of the same peptides."""
    flat = fragment_schema("prospect-local", 25.0).metadata
    assert flat[b"timsim.fragments.collision_energy"] == b"25.0"
    assert b"timsim.fragments.collision_energy_source" not in flat

    per = fragment_schema("prospect-local", 25.0, ce_source="ce.parquet").metadata
    assert per[b"timsim.fragments.collision_energy"] == b"per-precursor"
    assert per[b"timsim.fragments.collision_energy_source"] == b"ce.parquet"


def test_local_model_records_its_instrument_and_koina_does_not():
    from timsim_predict.fragments import LOCAL_INSTRUMENT

    assert fragment_schema("prospect-local", 25.0).metadata[b"timsim.fragments.instrument"] == \
        LOCAL_INSTRUMENT.encode()
    koina = fragment_schema("koina:Prosit_2020_intensity_HCD", 25.0).metadata
    assert b"timsim.fragments.instrument" not in koina


def test_the_local_predictor_is_built_for_timstof(monkeypatch):
    """The instrument must reach pepdl: without it the model predicts for "unknown"."""
    import pepdl.intensity.predictors as ip
    from timsim_predict import fragments

    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(ip, "DeepPeptideIntensityPredictor", Fake)
    fragments._local_predictor.cache_clear()
    try:
        fragments._local_predictor()
    finally:
        fragments._local_predictor.cache_clear()
    assert seen == {"instrument": "timstof"}
