# Changelog

All notable changes to `roust` are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Release infrastructure: tag-triggered PyPI (trusted publishing) + crates.io
  + GitHub Releases (`.github/workflows/release.yml`, `RELEASE.md`).

## [0.2.0] - UNRELEASED (not yet published to PyPI or crates.io) - Single Rust engine

- `pip install roust` (from a source checkout) now delivers the Rust binary
  directly (maturin `bindings = "bin"`), replacing the parallel Python
  implementation.
- **Behavior change - new region-packing defaults** (issue #4 campaign,
  PR #40): guarded span padding `--pad-lines` (default 5) and sub-linear
  length normalization `--len-exp` (default 0.85) are now the shipped
  defaults. Measured on SWE-bench Lite: FUNCTION 41.0 -> 53.3, LINE 35.7 ->
  42.7, FILE invariant (277/300); replicated on the 407-instance held-out
  SWE-bench Verified set. `--pad-lines 0 --len-exp 1.0` reproduces the
  pre-adoption packing byte-for-byte.
- **Low-confidence / exit-code contract**: `--json` `stats` now includes
  `top_score`, `matched_query_terms`/`total_query_terms`, and a calibrated
  `low_confidence: true` flag (also appended as `[low-confidence match]` to
  the stderr summary). Exit codes: `0` = results found (low-confidence
  matches included - roust still returns its best guess), `1` = no query
  term matched anything in the indexed corpus vocabulary, `2` = usage
  error.
- **Determinism fix** (issue #14): canonical (sorted) IDF summation in
  `pack_regions`' weight() - region packing is cross-process deterministic
  (`HashSet` iteration order previously leaked into region tie-breaks; see
  `roust-rs/PARITY_NOTES.md` item 15).
