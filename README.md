# timsim-predict

The **prediction step** of the timsim v2 pipeline: three small CLI jobs that turn peptide/precursor
parquet tables into predicted-property parquet tables, using the
[`pepdl`](https://github.com/MS-Simulation/pepdl) deep predictors.

Everything else in timsim v2 is Rust; these three are Python because the properties come from trained
models and re-implementing inference in Rust buys nothing.

## The three jobs

| Script | In | Out (table) | What it stores |
| --- | --- | --- | --- |
| `timsim-ccs` | `--precursors` + `--peptides` | `precursor_ccs` | `precursor_id, ccs, ccs_std` |
| `timsim-rt` | `--peptides` | `peptide_rt` | `peptide_id, rt_index, rt_sigma_hat, rt_k_hat` |
| `timsim-fragments` | `--precursors` (+ optional `--peptides`) | `fragment_intensities` | `precursor_id, ion_type, ordinal, frag_charge, intensity` |

**`timsim-ccs`** stores CCS in **Å²**, read straight out of the model (`return_ccs=True`) — never
`1/K0`. CCS is a property of the ion and is instrument-independent; `1/K0` is what a *particular*
drift tube measures given a gas, temperature and pressure. Keeping CCS here (structure axis) and
deriving `1/K0` per run (measurement axis) is what makes cross-instrument simulation possible.
Positional isomers share `(sequence, charge, mz)`, so the table is deduplicated before the model call.

**`timsim-rt`** stores a **gradient-independent RT index**, not seconds — seconds are what a specific
LC gradient produces, whereas the peptide's property is its hydrophobicity. The index range over the
*whole* peptide space travels with the artifact (`timsim.rt.index_min` / `index_max`) so a peptide
lands at the same gradient fraction regardless of what else is in the sample. Alongside it, the
elution-peak shape parameters `rt_sigma_hat` / `rt_k_hat` are unit Beta draws (σ ~ Beta(4,4),
k ~ Beta(1,20)) that are **identity-keyed** — a blake2b hash of the sequence — so a peptide's peak
shape is reproducible across runs and stable under adding/removing other peptides. Rejected peptides
get a null index, not a fabricated one.

**`timsim-fragments`** decodes the Prosit **`(29, 2, 3)`** tensor (position × ion type × charge)
**directly** — it never flattens. Two different flat-174 serialisations exist upstream (charge-major
vs ordinal-major), so flatten-then-decode is a trap. The one axis fact it needs (axis-2 index 0 = *y*,
index 1 = *b*) comes from `flatten_prosit_array`'s source and is pinned by
`tests/test_fragment_decode.py`. Output streams in row-groups, so peak memory is one chunk.

## Model specs

Every job takes `--model`, a short spec resolved by `timsim_predict._models`: omitted / `None` /
`"default"` picks our default for that property; `"koina:<name>"` picks that model served via
[Koina](https://koina.wilhelmlab.org) (needs network); any other string is a named local backend
passed through (e.g. `"chronologer"`). Defaults are `rt=chronologer`, `ccs=deep-ccs`,
`charge=site-specific`, `fragments=prospect-local`; the resolved name is recorded as provenance.

**CCS via Koina is deliberately not implemented** and raises `NotImplementedError`: the Koina CCS
models return `1/K0` and need a `calcmass` column, so wiring them re-introduces exactly the
gas-dependent inversion this tool exists to avoid. It will be wired where a Koina server is reachable
to test it against, not blind.

## Output format

All three write **schema-v2 parquet** with an explicitly built Arrow schema, including
**nullability** (pyarrow defaults everything to nullable; the Rust reader validates it and refuses a
file whose required column arrives nullable). Each file carries `timsim.*` key-value metadata —
`timsim.table`, `timsim.schema_version`, `timsim.axis` (`structure` or `measurement`),
`timsim.producer`, plus per-property provenance (`timsim.ccs.model`, `timsim.rt.model`,
`timsim.fragments.model`, `timsim.fragments.collision_energy`, the RT index range). These artifacts
are consumed by the Rust `timsim-schema` / `timsim-cli` side, and the jobs are normally driven as
nodes of the **necroflow** DAG rather than invoked by hand.

## Install

```bash
pip install 'timsim-predict[local]'   # on-device torch models (maps onto pepdl[local])
pip install 'timsim-predict[koina]'   # remote Koina prediction (maps onto pepdl[koina])
```

The base install is torch-free and koina-free: both arrive only through the extras, which forward to
`pepdl`'s own `[local]` / `[koina]`. Model weights are **not** bundled — they download on first run
through pepdl's hub (GitHub Releases, SHA-256 verified) into `$IMSPY_CACHE_DIR`, default
`~/.cache/imspy/models/v0.5.0/`.

## Example

```bash
timsim-rt        --peptides peptides.parquet --out peptide_rt.parquet
timsim-ccs       --precursors precursors.parquet --peptides peptides.parquet \
                 --out precursor_ccs.parquet
timsim-fragments --precursors frag_input.parquet --collision-energy 30 \
                 --out fragment_intensities.parquet
```

`--collision-energy` takes the **raw** NCE (~20–45), not the `/100`-encoded value stored in a `.d`;
per-run CE calibration is applied later, at render. Tests: `python -m pytest tests -q`.
