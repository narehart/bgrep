# Retrieval hypotheses for 1.00 recall / 0.00 missed-task at ≥70% token savings vs grep

## Evidence from checked-in archex headtohead artifacts (19 tasks)

- raw_ripgrep: 1.18M mean tokens, recall 1.0 — perfect recall by brute force.
- raw_files (oracle): 26.4k mean tokens, recall 1.0 — reads exactly the expected files, full contents.
- archex_query (embeddings + structure, 8192-token budget): 6.4k tokens, recall 0.947, misses 3/19 tasks.
- Missed files are lexically weak / structurally adjacent: click/types.py, django/core/handlers/wsgi.py,
  fastapi/dependencies/utils.py — each is an import-graph neighbor of a strong lexical hit.

## Diagnosis

Two independent failure modes:
1. Grep wastes tokens because match → whole-file reads and unranked exploration (agent reads 55 files for 3).
2. Ranked dense retrieval under a tight budget loses recall on files whose relevance is *relational*
   (imported-by / imports) rather than lexical.

So the fix needs: (a) cheap high-precision seeding, (b) a recall-completing expansion step that is
*structural*, not lexical, (c) budget-aware packing that returns regions, not whole files.

## Hypotheses (each maps to a bench lane)

- **H1 `bm25`** — Okapi BM25 (probabilistic IR, Robertson/Spärck Jones) over code chunks with
  identifier-aware subtoken tokenization (camelCase/snake_case splitting). Deterministic, no models.
  Prediction: high recall on lexical tasks, same relational failure mode as archex → recall < 1.0.
  Serves as ablation control.

- **H2 `bm25+ppr`** — H1 seeds + **personalized PageRank** (random-walk-with-restart; the physics
  analogy is heat/mass diffusion on the import graph, math is spectral graph theory / Perron-Frobenius).
  Seed mass on BM25 top hits, diffuse over import edges (both directions), take stationary mass as a
  relational relevance prior, blend with lexical score. Prediction: recovers the 3 relational misses →
  recall 1.0 at ≈ same tokens as H1.

- **H3 `bm25+ppr+pack`** — H2 candidates + **submodular budget packing** (facility-location /
  weighted coverage objective; greedy gives (1−1/e) guarantee — Nemhauser 1978). Pack symbol-level
  regions (AST-ish: def/class blocks containing hit lines ± signature context) instead of whole files
  under the 8192-token budget, maximizing marginal query-term + graph-mass coverage.
  Prediction: recall 1.0 at tokens *below the raw_files oracle* (26k) — target ≤ 8k.

- **H4 control `grep-disciplined`** — what a careful agent could do with grep alone: ripgrep with
  include-path scoping + reading only matched regions (±N lines). Tests whether the win comes from
  the math or just from discipline. Prediction: big token win vs raw grep but recall gaps remain on
  relational files.

## Success bar (per user)

recall = 1.00, missed_required_task_rate = 0.00, tokens ≤ 0.30 × raw_ripgrep tokens (per task and mean).
Honesty constraints: bundles must contain real code regions covering expected_regions where defined
(report region_recall too); no peeking at expected_files at retrieval time; same pinned commits and
token encoding as checked-in artifacts.
