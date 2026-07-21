# Paper eval wave — exact command sequence

Status: INFRASTRUCTURE ONLY (PR `paper-eval-infra`, refs #4). No long run in
this plan has been executed; every command below has been smoke-tested at
`--limit`/1-instance scale. Numbers marked DONE are real and committed.

## 0. Ground rules (read before any run)

- **Checkout discipline (issue #41).** `parity/region_eval_full.py` mutates
  the working trees under `--repos-dir` (`git checkout -f` + `git clean
  -fdq`). Every *concurrent* consumer gets its own private copy of the
  clones; never share a copy between two running evals. Adjacent contiguous
  shards share boundary repos, so shard number is NOT isolation. One full
  copy is ~2.1 GB (django 383 MB): 8 private copies ≈ 17 GB disk.

  ```bash
  # one private copy per concurrent shard (read-only against the source)
  for i in 1 2 3 4 5 6 7 8; do
    mkdir -p lab/shard_clones/$i
    for r in $(ls lab/swebench_repos); do
      git clone --quiet --no-hardlinks lab/swebench_repos/$r lab/shard_clones/$i/$r
    done
  done
  ```

- **Engine provenance.** `cargo build --release` (in `roust-rs/`) at the
  adopted HEAD before each arm; the blocking guard in every region-eval
  script refuses a binary whose embedded sha/dirty state does not match the
  checkout. Never pass `--allow-stale-engine` for paper numbers. All shards
  of one arm must report the same `engine_shas_seen` (the scorer asserts a
  single-element list is visible in its report — check it).

- **Driver guard.** The scripts refuse to start while a `swebench_driver`
  process is running. `BGREP_REGION_EVAL_SKIP_DRIVER_GUARD=1` only with an
  independently confirmed disjoint repo set.

- **Interpreter.** `.venv-pkg/bin/python` (pandas + pyarrow + requests).
  The stats layer (`lab/stats/paired_tests.py`) is stdlib-only.

## 1. Full SWE-bench test split (2,294 instances) — DONE except the run

```bash
# fetch (DONE, artifacts committed): 12.1 MB download, 2,294 rows, and the
# repo-coverage gate — the full split is exactly our 12 cloned repos.
.venv-pkg/bin/python scripts/fetch_swebench_full.py            # or --dry-run

# default-engine arm, 8 contiguous shards (repo-sorted; django=850 dominates
# the middle shards). Wall time basis: Lite 300 = 1,489 s on this machine
# (~5.0 s/instance) => 2,294 seq ≈ 3.2 h; 8-way ≈ 30–60 min wall.
for i in 1 2 3 4 5 6 7 8; do
  .venv-pkg/bin/python parity/region_eval_full.py --shard $i/8 \
    --repos-dir lab/shard_clones/$i \
    --report lab/results_regions/full2294_default_s${i}of8.jsonl \
    > lab/results_regions/full2294_default_s${i}of8.log 2>&1 &
done; wait

# score (read-only git show; may point at the shared clones)
.venv-pkg/bin/python lab/agentless_metric_full.py \
  --predictions lab/results_regions/full2294_default_s*of8.jsonl \
  --out lab/results_regions/agentless_metric_full_default.json
```

Sanity gates: 2,294 merged records (scorer default `--expect-n 2294`);
`n_region_eval_errors` ≈ 0; single `engine_shas_seen`.

## 2. BM25 same-harness arm

**Definition (the paper's "BM25-only"):** the engine invoked with every
channel-ablation flag it has —

```
--no-history --no-docs --no-anchors --no-testbridge
```

(via `region_eval_full.py --bm25-only`). This disables: commit-message BM25
field + co-change frontier (history), docs-page field + docs bridge, symbol
anchors, test bridge. **Remaining active** (no engine flag exists): the BM25F
lexical core — Okapi body field, path-token field, comment/NL field,
implementation-file prior — the import-graph/same-directory structural
expansion in `select_files` (its co-change input IS removed), and the
identical region packer. `personalized_pagerank` needs no flag: it is dead
code, never invoked by the CLI (roust-rs/src/core.rs). Smoke-verified
2026-07-20: on this repo the arm's regions differ from defaults (history
channel; anchors/bridges empty for the probe query); on
psf__requests-1142 the harness runs it end-to-end.

```bash
# Lite BM25 arm (~25 min; needs a private clone copy if anything else runs)
.venv-pkg/bin/python parity/region_eval_full.py --bm25-only \
  --gold-parquet lab/swebench_lite.parquet \
  --repos-dir lab/shard_clones/1 \
  --report lab/results_regions/lite300_bm25.jsonl
.venv-pkg/bin/python lab/agentless_metric_full.py \
  --predictions lab/results_regions/lite300_bm25.jsonl \
  --gold-parquet lab/swebench_lite.parquet --expect-n 300 \
  --out lab/results_regions/agentless_metric_lite_bm25.json

# Verified held-out BM25 arm (~35 min)
.venv-pkg/bin/python parity/region_eval_full.py --bm25-only \
  --gold-parquet lab/swebench_verified_heldout.parquet \
  --repos-dir lab/shard_clones/2 \
  --report lab/results_regions/verified407_bm25.jsonl
.venv-pkg/bin/python lab/agentless_metric_full.py \
  --predictions lab/results_regions/verified407_bm25.jsonl \
  --gold-parquet lab/swebench_verified_heldout.parquet --expect-n 407 \
  --out lab/results_regions/agentless_metric_verified_bm25.json

# (optional) full-split BM25 arm: section 1's shard loop + --bm25-only,
# reports named full2294_bm25_s${i}of8.jsonl. ≈ same wall time again.
```

## 3. ContextBench redo (adopted engine)

Per `lab/contextbench/README.md` (evaluator = a clone of EuniAI/ContextBench
with its own py3.11 venv). Rebuild the release binary at the adopted HEAD
first; the run records engine version per prediction row. 266 python tasks,
19 repos; budget ~1–2 h including evaluator repo checkouts (cache-dependent).

```bash
<evaluator>/.venv/bin/python lab/contextbench/run_contextbench.py \
  --evaluator <evaluator-clone> \
  --cache lab/contextbench_repos/cache \
  --tmp-root lab/contextbench_repos/tmp \
  --out-dir lab/contextbench_repos/run_python_adopted \
  --language python --evaluate
```

## 4. Multilingual redo (adopted engine)

```bash
# rebuild the JS/TS parquet (HF tree API, requests-only)
.venv-pkg/bin/python lab/mswe_adapter.py --out lab/mswe_jsts.parquet
# clones: the mswe repos are NOT the 12 SWE-bench clones; materialize a
# private clone dir first, then:
.venv-pkg/bin/python parity/region_eval_full.py \
  --gold-parquet lab/mswe_jsts.parquet \
  --repos-dir lab/mswe_repos_private \
  --report lab/results_regions/mswe_jsts_adopted.jsonl
.venv-pkg/bin/python lab/agentless_metric_full.py \
  --predictions lab/results_regions/mswe_jsts_adopted.jsonl \
  --gold-parquet lab/mswe_jsts.parquet --expect-n 580 \
  --out lab/results_regions/agentless_metric_mswe_adopted.json
```

CAVEAT: report FILE and LINE only. The FUNCTION metric's AST walk is
Python-only (`function_spans` skips non-.py files), so on JS/TS gold sets it
is vacuously "correct" — not a real number. 580 instances ≈ 50 min at the
Lite rate, but material-ui (10.4k files) indexes slower; budget 2 h.

## 5. Statistics (lab/stats/paired_tests.py)

```bash
# v10 vs v11 adoption deltas — DONE, artifact: lab/stats/v10_vs_v11_adoption.json
python3 lab/stats/paired_tests.py \
  --a-predictions lab/results_regions/full300_v10.jsonl \
  --a-metric lab/results_regions/agentless_metric_v4.json --label-a v10 \
  --b-predictions lab/results_regions/full300_v11.jsonl \
  --b-metric lab/results_regions/agentless_metric_v11.json --label-b v11 \
  --out lab/stats/v10_vs_v11_adoption.json
# result (n=300, 10k paired bootstrap, McNemar exact):
#   FILE     92.33 -> 92.33  +0.00pp [ +0.00,  +0.00]  p=1.0  (0 discordants)
#   FUNCTION 41.00 -> 53.33 +12.33pp [ +8.67, +16.33]  p=1.46e-10
#   LINE     35.67 -> 42.67  +7.00pp [ +3.67, +10.33]  p=1.04e-04
#   fraction  0.4564 -> 0.5168 +0.0605 [+0.0356, +0.0869]

# Verified held-out, old vs new formula (artifacts already committed)
python3 lab/stats/paired_tests.py \
  --a-predictions lab/results_regions/full407_verified_old.jsonl \
  --a-metric lab/results_regions/agentless_metric_verified_old.json --label-a old \
  --b-predictions lab/results_regions/full407_verified_new.jsonl \
  --b-metric lab/results_regions/agentless_metric_verified_new.json --label-b new \
  --out lab/stats/verified_old_vs_new.json

# full split: default vs BM25 arm (after sections 1–2; shard files merge)
python3 lab/stats/paired_tests.py \
  --a-predictions lab/results_regions/full2294_bm25_s*of8.jsonl \
  --a-metric lab/results_regions/agentless_metric_full_bm25.json --label-a bm25 \
  --b-predictions lab/results_regions/full2294_default_s*of8.jsonl \
  --b-metric lab/results_regions/agentless_metric_full_default.json --label-b default \
  --out lab/stats/full_bm25_vs_default.json
```

Conventions: paired bootstrap resamples instances (A/B move together),
percentile 95% CI, `--n-boot 10000 --seed 20260718` defaults; McNemar exact
two-sided on discordant pairs; errors count as wrong at every level with the
full-set denominator (matches the scorers' unified convention). Unit tests:
`tests/test_paired_stats.py`.

Paper note (2026-07-21, scoreboard metric composition): the paper's localization table must present roust's file-level result as a dual metric — depth-aligned File@10 first (all gold files within top 10: 82.7 Lite / 79.4 Verified held-out), then the all-gold-retrieved Agentless-metric FILE score (92.3 Lite / 92.1 Verified, ~35 files returned, range 22–38) — because Acc@10 (any-gold-in-top-10) and all-gold-retrieved are incommensurable in both directions; File@10 is the number to rank against Acc@10 rows, and on it roust sits below the trained retrievers (SweRankEmbed-Small 90.9).
