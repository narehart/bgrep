#!/usr/bin/env python3
"""Agentless-style "% Correct Location" metric for roust on SWE-bench Lite.

Definition (Agentless, arXiv:2407.01489, Table 1): the fraction of instances
where the PREDICTED location set is a SUPERSET of the GOLD edit locations,
at three granularities (FILE / FUNCTION / LINE). Agentless's own numbers use
an LLM to narrow predicted files down to functions/lines; roust has no LLM
in the loop, so "predicted" here means "covered by roust's returned regions"
(the --json `regions` field: per-file line spans packed under an 8192-token
budget).

v2 (this version): all four numbers (FILE, FUNCTION, LINE, region PRECISION)
are computed EXACTLY from a single fresh run of the shipped Rust engine that
persists the actual returned region spans:

  - lab/results_regions/full300_v8.jsonl (parity/region_eval2.py's Part-A-only
    run over all 300 SWE-bench Lite instances, shipped roust-rs binary,
    --budget 8192, confidence-scheduled packing). Each line is one instance's
    record: hunk_file_covered, all_gold_files_retrieved, hunk_line_recall,
    hunk_touched, tokens (as parity/region_eval.py always computed), PLUS
    the raw `regions` dict and `engine_sha`/`engine_dirty` provenance, which
    the older `full300_final.json` never persisted.
  - lab/swebench_lite.parquet (gold patches -- parse_gold_hunks + AST
    function spans).
  - lab/swebench_repos/<owner>__<name>/ via `git show <base_commit>:<path>`
    ONLY (read-only, no checkout/mutation).

FUNCTION-level is now EXACT (not a proxy): predicted_functions_for_instance
mirrors gold_functions_for_instance exactly (same git_show, same AST
function_spans, same span identity `(path, start, end)`) but walks roust's
actual returned region spans instead of the gold hunk lines, and an instance
is correct iff every gold function span is contained in the predicted
function-span set.

CONVENTION NOTE (2026-07 hardening pass): errors now count as WRONG at every
level -- FILE, FUNCTION, and LINE all share the same denominator (n = all
loaded records). Previously FUNCTION excluded engine errors and git_show
failures from its denominator while FILE/LINE counted them as wrong; the two
conventions differ by <=0.12pp on every published roust run (and <=0.41pp on the archex baseline artifacts, where the old exclusion favored the baseline), so published artifacts
were deliberately NOT regenerated -- the report's `convention` field records
which convention an artifact was scored under (artifacts without the field
predate the unification and used the old FUNCTION exclusion). The excluded
classes are still counted and reported separately (`*_counted_wrong` keys).
"""

from __future__ import annotations

import ast
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/nicholasarehart/programming-projects/bgrep")
sys.path.insert(0, str(REPO_ROOT / "parity"))
from region_eval import parse_gold_hunks, line_in_spans  # noqa: E402

FULL300_V8 = REPO_ROOT / "lab" / "results_regions" / "full300_v8.jsonl"
LITE_PARQUET = REPO_ROOT / "lab" / "swebench_lite.parquet"
SWEBENCH_REPOS = REPO_ROOT / "lab" / "swebench_repos"
OUT_PATH = REPO_ROOT / "lab" / "results_regions" / "agentless_metric_v2.json"

# Agentless's own published GPT-4o SWE-bench Lite numbers (Table 1 of
# arXiv:2407.01489) -- for side-by-side reporting only, not reproduced here.
AGENTLESS_GPT4O = {"file": 69.7, "function": 52.0, "line": 35.3}


def mean_median(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), statistics.median(values)


def pct(x: float | None) -> float | None:
    return None if x is None else round(100.0 * x, 2)


# ---------------------------------------------------------------------------
# function spans (AST, via read-only `git show`) -- shared gold/predicted
# ---------------------------------------------------------------------------


def git_show(repo_slug: str, sha: str, path: str) -> str | None:
    repo_path = SWEBENCH_REPOS / repo_slug.replace("/", "__")
    if not repo_path.exists():
        return None
    r = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=repo_path,
                        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    return r.stdout


def function_spans(source: str) -> list[tuple[int, int]]:
    """Returns [(start, end_inclusive), ...] for every function/method
    definition in `source` (def and async def; NOT bare class bodies --
    "function/method" per the task's definition). Span start includes any
    decorator lines; end is ast's end_lineno (Python 3.8+)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            end = getattr(node, "end_lineno", None) or node.lineno
            spans.append((start, end))
    return spans


def gold_function_spans_for_instance(repo_slug: str, base_commit: str,
                                      gold_hunks: dict[str, list[tuple[int, int]]]
                                      ) -> tuple[set[tuple[str, int, int]], bool]:
    """Returns ({(path, start, end), ...}, ast_ok). A gold function span is
    any function/method AST span in a gold-hunk .py file that contains >=1
    gold-patch line. ast_ok is False if any gold .py file couldn't be
    fetched via `git show` (informational + exclusion trigger -- see
    compute_function_level_exact)."""
    spans_out: set[tuple[str, int, int]] = set()
    ast_ok = True
    for path, ranges in gold_hunks.items():
        if not path.endswith(".py"):
            continue
        src = git_show(repo_slug, base_commit, path)
        if src is None:
            ast_ok = False
            continue
        line_set: set[int] = set()
        for s, e in ranges:
            line_set.update(range(s, e + 1))
        for fs, fe in function_spans(src):
            if any(fs <= ln <= fe for ln in line_set):
                spans_out.add((path, fs, fe))
    return spans_out, ast_ok


def predicted_function_spans_for_instance(repo_slug: str, base_commit: str,
                                           regions: dict[str, list[list[int]]]
                                           ) -> tuple[set[tuple[str, int, int]], bool]:
    """Mirrors gold_function_spans_for_instance exactly (same git_show, same
    AST function_spans, same span identity `(path, start, end)`), but walks
    roust's actual RETURNED region spans instead of the gold hunk lines: a
    predicted function span is any function/method AST span in a
    RETURNED .py file that overlaps >=1 of that file's returned region
    spans. Non-.py returned files are skipped (function_spans is a Python
    AST parser only -- same convention the gold side already used)."""
    spans_out: set[tuple[str, int, int]] = set()
    ast_ok = True
    for path, region_spans in regions.items():
        if not path.endswith(".py"):
            continue
        src = git_show(repo_slug, base_commit, path)
        if src is None:
            ast_ok = False
            continue
        for fs, fe in function_spans(src):
            if any(fs <= e and s <= fe for s, e in region_spans):
                spans_out.add((path, fs, fe))
    return spans_out, ast_ok


# ---------------------------------------------------------------------------
# metric computation
# ---------------------------------------------------------------------------


def compute_file_level(records: list[dict]) -> dict:
    """Exact. Agentless's FILE metric is `predicted files ⊇ gold files`.
    The stored `all_gold_files_retrieved` is defined identically
    (`hunk_file_covered == 1.0`, i.e. every gold file is a key in the
    returned `regions` dict) -- so this is a direct read, not an
    approximation.

    An engine-error record (no `regions` ever obtained) has no stored
    `all_gold_files_retrieved` key; such records count as WRONG (0.0), not
    dropped -- same denominator as every other level (see the module
    docstring's convention note). `n_engine_errors` reports the count."""
    n_engine_errors = sum(1 for r in records if r["error"] is not None)
    vals = [1.0 if r.get("all_gold_files_retrieved", False) else 0.0 for r in records]
    m, _ = mean_median(vals)
    return {"pct_correct": pct(m), "n": len(records), "n_correct": sum(int(v) for v in vals),
            "n_engine_errors": n_engine_errors}


def compute_line_level(records: list[dict]) -> dict:
    """All-or-nothing part is EXACT: the stored `hunk_line_recall` is
    `covered_lines / total_lines` over the exact union of gold lines for
    the instance. Since this is a true fraction of a finite set,
    `hunk_line_recall == 1.0` is logically equivalent to "every gold line
    is covered by a returned region" -- exactly Agentless's LINE
    superset condition.

    Also reports the mean-fraction (prior "hunk_line_recall" metric) for
    continuity -- NOTE this is a materially easier number (partial credit)
    than the all-or-nothing metric to its left.

    An engine-error record has no stored `hunk_line_recall`; it counts as
    WRONG (0.0) in the all-or-nothing metric (same denominator as every
    other level -- see the module docstring's convention note) and is
    excluded from the mean/median fraction (there is no partial-credit
    fraction for an instance with zero returned regions)."""
    n_engine_errors = sum(1 for r in records if r["error"] is not None)
    all_or_nothing = [1.0 if r.get("hunk_line_recall") == 1.0 else 0.0 for r in records]
    fractions = [r["hunk_line_recall"] for r in records
                 if r.get("hunk_line_recall") is not None]
    aon_m, _ = mean_median(all_or_nothing)
    frac_m, frac_med = mean_median(fractions)
    return {
        "pct_correct_all_or_nothing": pct(aon_m),
        "n": len(records),
        "n_correct_all_or_nothing": sum(int(v) for v in all_or_nothing),
        "mean_fraction_covered": frac_m,
        "median_fraction_covered": frac_med,
        "n_engine_errors": n_engine_errors,
    }


def compute_function_level_exact(records: list[dict], patch_by_id: dict[str, str]) -> dict:
    """EXACT (superseding the old hunk_touched==1.0 proxy). An instance is
    correct iff its gold function-span set is a subset of its predicted
    function-span set (both sets computed via AST over `git show
    <base_commit>:<path>`, see gold_function_spans_for_instance /
    predicted_function_spans_for_instance).

    UNIFIED CONVENTION (2026-07): errors count as WRONG, same denominator
    as FILE/LINE (n = all records). Two classes are counted wrong here and
    reported separately (per the task's "count and report, don't silently
    drop" policy):
      - region_eval2 itself failed for the instance (`r["error"]` set --
        no regions were ever obtained, e.g. a checkout/engine failure):
        zero regions cannot cover a non-empty gold set, so wrong.
      - a `git_show` failure on either the gold or predicted side (ast_ok
        False) -- the true gold or predicted function set is then
        incomplete/unknown; counting the instance WRONG is the
        conservative, self-penalizing resolution of that ambiguity.
    (Old convention, used by artifacts that predate the report-level
    `convention` field: both classes were EXCLUDED from the FUNCTION
    denominator. Shift <=0.12pp on every published roust run, <=0.41pp on the archex baseline artifacts; none
    regenerated -- see module docstring.)

    Flagged assumption: an instance whose (complete, ast_ok) gold function
    set is EMPTY (e.g. every gold hunk lands in a non-.py file, or in
    module-level code outside any def) counts as correct -- the superset
    condition holds vacuously, same convention Agentless's own FILE/LINE
    supersets use for an empty gold set. This does not silently inflate the
    number: the per-instance detail list below records `n_gold_functions`
    for every counted instance so the case is auditable.
    """
    n_correct = 0
    n_judged = 0
    n_engine_errors = 0
    n_git_show_failures = 0
    detail = []
    for r in records:
        if r["error"] is not None:
            n_engine_errors += 1
            continue
        patch = patch_by_id.get(r["instance_id"])
        if patch is None:
            n_engine_errors += 1
            continue
        gold_hunks = parse_gold_hunks(patch)
        gold_spans, gold_ok = gold_function_spans_for_instance(r["repo"], r["base_commit"], gold_hunks)
        pred_spans, pred_ok = predicted_function_spans_for_instance(
            r["repo"], r["base_commit"], r["regions"])
        if not (gold_ok and pred_ok):
            n_git_show_failures += 1
            continue
        n_judged += 1
        correct = gold_spans.issubset(pred_spans)
        if correct:
            n_correct += 1
        detail.append({
            "instance_id": r["instance_id"],
            "n_gold_functions": len(gold_spans),
            "n_predicted_functions": len(pred_spans),
            "correct": correct,
        })
    n_total = len(records)
    pct_m = (n_correct / n_total) if n_total else None
    return {
        "pct_correct": pct(pct_m),
        "n": n_total,
        "n_judged": n_judged,
        "n_correct": n_correct,
        "n_engine_errors_counted_wrong": n_engine_errors,
        "n_git_show_failures_counted_wrong": n_git_show_failures,
        "detail": detail,
    }


def compute_region_precision(records: list[dict], patch_by_id: dict[str, str]) -> dict:
    """NEW (issue #4): "are we returning too much?" Per instance:
    (gold lines covered by returned regions) / (total lines spanned by
    ALL returned regions, across every returned file, not just gold
    files). Low precision means roust is packing a lot of non-gold
    context alongside the gold lines -- expected and by design (regions
    are read for surrounding context, not just the exact diff lines), but
    worth quantifying.

    Instances with `r["error"]` set (region_eval2 failure, no regions
    obtained) are excluded and counted. An instance whose returned
    `regions` dict is present but spans zero total lines (empty budget
    edge case) has undefined precision (0/0) and is excluded from the
    mean but still contributes to `total_region_lines_per_instance`...
    actually excluded there too since there's nothing to measure."""
    precisions = []
    total_region_lines_list = []
    n_excluded = 0
    for r in records:
        if r["error"] is not None:
            n_excluded += 1
            continue
        patch = patch_by_id.get(r["instance_id"])
        gold_hunks = parse_gold_hunks(patch) if patch is not None else {}
        gold_line_sets: dict[str, set[int]] = {}
        for f, ranges in gold_hunks.items():
            s: set[int] = set()
            for a, b in ranges:
                s.update(range(a, b + 1))
            gold_line_sets[f] = s

        regions: dict[str, list[list[int]]] = r["regions"]
        total_region_lines = 0
        covered_gold_lines = 0
        for f, spans in regions.items():
            total_region_lines += sum(e - s + 1 for s, e in spans)
            if f in gold_line_sets:
                covered_gold_lines += sum(
                    1 for ln in gold_line_sets[f] if line_in_spans(ln, spans))

        if total_region_lines == 0:
            n_excluded += 1
            continue
        precisions.append(covered_gold_lines / total_region_lines)
        total_region_lines_list.append(total_region_lines)

    prec_m, prec_med = mean_median(precisions)
    lines_m, lines_med = mean_median([float(x) for x in total_region_lines_list])
    return {
        "mean_precision": prec_m,
        "median_precision": prec_med,
        "n": len(precisions),
        "n_excluded": n_excluded,
        "mean_total_region_lines_per_instance": lines_m,
        "median_total_region_lines_per_instance": lines_med,
    }


def load_full300_v8() -> list[dict]:
    records = []
    with FULL300_V8.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    records = load_full300_v8()
    assert len(records) == 300, f"expected 300 stored records, got {len(records)}"

    engine_shas = {r["engine_sha"] for r in records if r["engine_sha"] is not None}
    engine_dirty = {r["engine_dirty"] for r in records if r["engine_dirty"] is not None}

    import pandas as pd
    df = pd.read_parquet(LITE_PARQUET)
    patch_by_id = {row["instance_id"]: row["patch"] for _, row in df.iterrows()}

    n_region_eval2_errors = sum(1 for r in records if r["error"] is not None)

    file_correct_subset = [r for r in records if r["all_gold_files_retrieved"]]

    print("Computing exact FUNCTION-level metric + region precision via read-only "
          "`git show` (this walks every gold + returned .py file's AST)...", file=sys.stderr)

    def block(recs: list[dict]) -> dict:
        return {
            "n": len(recs),
            "file": compute_file_level(recs),
            "function": compute_function_level_exact(recs, patch_by_id),
            "line": compute_line_level(recs),
            "region_precision": compute_region_precision(recs, patch_by_id),
        }

    all_block = block(records)
    subset_block = block(file_correct_subset)

    report: dict = {
        "convention": "errors-count-as-wrong-at-all-levels/v1: FILE, FUNCTION, and LINE share "
                      "the same denominator (n = all loaded records); engine errors and "
                      "git_show failures count as WRONG and are reported separately "
                      "(*_counted_wrong / n_engine_errors keys). Artifacts without this field "
                      "predate the unification and scored FUNCTION with errors EXCLUDED from "
                      "the denominator (<=0.12pp difference on every published roust run; <=0.41pp on the archex baseline artifacts).",
        "source": {
            "predictions": str(FULL300_V8.relative_to(REPO_ROOT)),
            "gold": str(LITE_PARQUET.relative_to(REPO_ROOT)),
            "n_instances": len(records),
            "n_region_eval2_errors": n_region_eval2_errors,
            "engine_shas_seen": sorted(engine_shas),
            "engine_dirty_seen": sorted(engine_dirty),
            "pipeline": "shipped roust-rs engine, --budget 8192, confidence-scheduled packing "
                        "(parity/region_eval2.py Part A, no-LLM)",
        },
        "all_instances": all_block,
        "file_correct_subset": {
            **subset_block,
            "note": "restricted to instances where FILE-level was already correct -- isolates "
                    "region/line/function quality from file recall (spec item 4)",
        },
        "agentless_gpt4o_published": AGENTLESS_GPT4O,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)

    print_table(report)


def print_table(report: dict) -> None:
    al = report["agentless_gpt4o_published"]
    a = report["all_instances"]
    fc = report["file_correct_subset"]

    def row(label, ours, theirs):
        ours_s = f"{ours:.1f}%" if ours is not None else "n/a"
        theirs_s = f"{theirs:.1f}%" if theirs is not None else "n/a"
        print(f"{label:38} {ours_s:>10}   {theirs_s:>10}")

    print("\n" + "=" * 78)
    print("Agentless-style %% Correct Location -- roust (no-LLM) vs GPT-4o Agentless")
    print("(FUNCTION is now EXACT, not a proxy -- v2)")
    print("=" * 78)
    print(f"{'':38} {'roust':>10}   {'GPT-4o Agentless':>10}")
    row("FILE   (predicted ⊇ gold files)", a["file"]["pct_correct"], al["file"])
    row("FUNCTION (EXACT)                ", a["function"]["pct_correct"], al["function"])
    row("LINE   (all-or-nothing)         ", a["line"]["pct_correct_all_or_nothing"], al["line"])
    print(f"\n{'LINE mean-fraction covered (continuity w/ prior reporting)':60} "
          f"{a['line']['mean_fraction_covered']:.4f}")
    rp = a["region_precision"]
    print(f"{'REGION PRECISION (gold lines / total returned lines), mean':60} "
          f"{rp['mean_precision']:.4f}  (n={rp['n']}, excluded={rp['n_excluded']})")
    print(f"{'  mean total region lines / instance':60} "
          f"{rp['mean_total_region_lines_per_instance']:.1f}")
    print(f"\nFUNCTION n={a['function']['n']} n_judged={a['function']['n_judged']} "
          f"n_correct={a['function']['n_correct']} "
          f"counted_wrong(engine_errors={a['function']['n_engine_errors_counted_wrong']}, "
          f"git_show_failures={a['function']['n_git_show_failures_counted_wrong']})")

    print(f"\n--- restricted to file-correct subset (n={fc['n']}/{a['n']}) ---")
    row("FUNCTION (EXACT)                ", fc["function"]["pct_correct"], None)
    row("LINE   (all-or-nothing)         ", fc["line"]["pct_correct_all_or_nothing"], None)
    print(f"{'LINE mean-fraction covered':60} {fc['line']['mean_fraction_covered']:.4f}")
    rpf = fc["region_precision"]
    print(f"{'REGION PRECISION, mean':60} {rpf['mean_precision']:.4f} (n={rpf['n']})")


if __name__ == "__main__":
    main()
