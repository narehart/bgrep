#!/usr/bin/env python3
"""
One-bit elicitation headroom diagnostic over abl_bridges_v7.jsonl.

Offline simulation: for each SWE-bench Lite instance, derive up to K
"subsystem" choices (top-2-level directory prefixes of returned_files,
ranked by best (lowest) file rank among returned_files), and ask whether
an oracle bit (agent picks the gold file's prefix, if offered) would help.

No repo access, no LLM calls — pure re-derivation from stored returned_files.
"""
import json
import sys
from collections import OrderedDict

PATH = "/Users/nicholasarehart/programming-projects/bgrep/lab/results_swebench/abl_bridges_v7.jsonl"


def top2_prefix(path):
    parts = path.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def candidate_prefixes(returned_files, k):
    """Top-2-level dir prefixes of returned_files, ranked by best (first) rank
    at which that prefix appears, top-k distinct prefixes."""
    best_rank = OrderedDict()
    for rank, f in enumerate(returned_files):
        p = top2_prefix(f)
        if p not in best_rank:
            best_rank[p] = rank
    # already insertion-ordered by first occurrence == best rank order
    prefixes = list(best_rank.keys())[:k]
    return prefixes


def gold_prefixes(gold_files):
    return {top2_prefix(g) for g in gold_files}


def rerank_by_prefix(returned_files, chosen_prefix):
    """Filter/boost returned_files to those under chosen_prefix first,
    preserving relative order, then append the rest (also preserving order)."""
    boosted = [f for f in returned_files if top2_prefix(f) == chosen_prefix]
    rest = [f for f in returned_files if top2_prefix(f) != chosen_prefix]
    return boosted + rest


def load_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_at10(gold_files, returned_files):
    return set(gold_files) <= set(returned_files[:10])


def evaluate_atall(gold_files, returned_files):
    return set(gold_files) <= set(returned_files)


def run_for_k(rows, k):
    """Full simulation for a given number of offered choices k."""
    at10_fail_rows = []
    atall_fail_rows = []

    for d in rows:
        gold = d["gold_files"]
        ret = d["returned_files"]
        if not evaluate_at10(gold, ret):
            at10_fail_rows.append(d)
        if not evaluate_atall(gold, ret):
            atall_fail_rows.append(d)

    n = len(rows)
    n_at10_fail = len(at10_fail_rows)
    n_atall_fail = len(atall_fail_rows)

    # --- (a) elicitation ceiling: how often gold's prefix is among the
    # top-k offered choices, for @10-failures and @all-failures separately.
    def ceiling_stats(fail_rows):
        hit = 0
        multi_gold_prefix_rows = 0
        detail = []
        for d in fail_rows:
            ret = d["returned_files"]
            choices = candidate_prefixes(ret, k)
            gprefs = gold_prefixes(d["gold_files"])
            if len(gprefs) > 1:
                multi_gold_prefix_rows += 1
            reachable = bool(gprefs & set(choices))
            if reachable:
                hit += 1
            detail.append({
                "instance_id": d["instance_id"],
                "choices": choices,
                "gold_prefixes": sorted(gprefs),
                "reachable": reachable,
            })
        return hit, multi_gold_prefix_rows, detail

    at10_hit, at10_multigold, at10_detail = ceiling_stats(at10_fail_rows)
    atall_hit, atall_multigold, atall_detail = ceiling_stats(atall_fail_rows)

    # --- (b) @10 after oracle-answer rerank vs baseline .827
    # Only meaningful/executable for @10-failures (rerank can only change
    # ORDER of already-returned files; if gold wasn't in returned_files at
    # all, no rerank of returned_files helps — that's the @all case, handled
    # as "second-shot reachable" instead, per spec).
    new_at10_pass = 0
    still_at10_fail_no_prefix_offered = 0
    still_at10_fail_prefix_offered_but_no_help = 0
    for d in at10_fail_rows:
        ret = d["returned_files"]
        gold = d["gold_files"]
        choices = candidate_prefixes(ret, k)
        gprefs = gold_prefixes(gold)
        offered = gprefs & set(choices)
        if not offered:
            still_at10_fail_no_prefix_offered += 1
            continue
        # oracle picks a matching offered prefix (if multiple gold prefixes,
        # picking any one is a single MC choice; simulate picking the
        # first-ranked offered gold prefix per spec's single-choice bit)
        chosen = next(p for p in choices if p in offered)
        reranked = rerank_by_prefix(ret, chosen)
        if evaluate_at10(gold, reranked):
            new_at10_pass += 1
        else:
            still_at10_fail_prefix_offered_but_no_help += 1

    at10_after_oracle = (n - n_at10_fail + new_at10_pass) / n

    # --- (c) degenerate offered-choice sets
    # (i) all k offered prefixes identical (only 1 distinct prefix present
    #     among returned_files, or fewer than k distinct prefixes exist and
    #     they collapse to 1) OR
    # (ii) gold prefix == rank-1 (first) offered prefix already (bit adds
    #     nothing because the top candidate is already correct)
    def degenerate_stats(fail_rows, tag):
        all_identical = 0
        gold_is_rank1 = 0
        both = 0
        either = 0
        for d in fail_rows:
            ret = d["returned_files"]
            gold = d["gold_files"]
            choices = candidate_prefixes(ret, k)
            gprefs = gold_prefixes(gold)
            distinct = set(choices)
            cond_identical = len(distinct) <= 1
            cond_rank1 = bool(choices) and (choices[0] in gprefs)
            if cond_identical:
                all_identical += 1
            if cond_rank1:
                gold_is_rank1 += 1
            if cond_identical and cond_rank1:
                both += 1
            if cond_identical or cond_rank1:
                either += 1
        return {
            "tag": tag,
            "n_fail": len(fail_rows),
            "all_prefixes_identical": all_identical,
            "gold_prefix_is_rank1_already": gold_is_rank1,
            "both": both,
            "either_degenerate": either,
        }

    at10_degen = degenerate_stats(at10_fail_rows, "@10-fail")
    atall_degen = degenerate_stats(atall_fail_rows, "@all-fail")

    return {
        "k": k,
        "n": n,
        "n_at10_fail": n_at10_fail,
        "n_atall_fail": n_atall_fail,
        "baseline_at10": (n - n_at10_fail) / n,
        "baseline_atall": (n - n_atall_fail) / n,
        "at10_ceiling_hit": at10_hit,
        "at10_ceiling_rate": at10_hit / n_at10_fail if n_at10_fail else float("nan"),
        "at10_multigold_rows": at10_multigold,
        "atall_ceiling_hit": atall_hit,
        "atall_ceiling_rate": atall_hit / n_atall_fail if n_atall_fail else float("nan"),
        "atall_multigold_rows": atall_multigold,
        "at10_after_oracle_rerank": at10_after_oracle,
        "at10_new_passes_from_rerank": new_at10_pass,
        "at10_fail_no_prefix_offered": still_at10_fail_no_prefix_offered,
        "at10_fail_prefix_offered_but_rerank_insufficient": still_at10_fail_prefix_offered_but_no_help,
        "at10_degen": at10_degen,
        "atall_degen": atall_degen,
        "at10_detail": at10_detail,
        "atall_detail": atall_detail,
    }


def print_report(res, verbose_detail=False):
    k = res["k"]
    print(f"\n{'='*70}")
    print(f"K = {k} offered choices")
    print(f"{'='*70}")
    print(f"n = {res['n']}")
    print(f"baseline @10  = {res['baseline_at10']:.4f}  ({res['n'] - res['n_at10_fail']}/{res['n']} pass, {res['n_at10_fail']} fail)")
    print(f"baseline @all = {res['baseline_atall']:.4f}  ({res['n'] - res['n_atall_fail']}/{res['n']} pass, {res['n_atall_fail']} fail)")

    print(f"\n(a) Elicitation ceiling (gold prefix among top-{k} offered):")
    print(f"  @10-failures:  {res['at10_ceiling_hit']}/{res['n_at10_fail']} = {res['at10_ceiling_rate']:.4f}"
          f"  (multi-gold-prefix instances: {res['at10_multigold_rows']})")
    print(f"  @all-failures: {res['atall_ceiling_hit']}/{res['n_atall_fail']} = {res['atall_ceiling_rate']:.4f}"
          f"  (multi-gold-prefix instances: {res['atall_multigold_rows']})  <- \"second-shot reachable\"")

    print(f"\n(b) @10 after oracle-answer rerank (of @10-failures only):")
    print(f"  baseline @10                 = {res['baseline_at10']:.4f}")
    print(f"  @10 after oracle rerank       = {res['at10_after_oracle_rerank']:.4f}"
          f"  (+{res['at10_new_passes_from_rerank']} instances converted, "
          f"delta = +{res['at10_after_oracle_rerank']-res['baseline_at10']:.4f})")
    print(f"  of the {res['n_at10_fail']} @10-failures:")
    print(f"    converted to pass by rerank         : {res['at10_new_passes_from_rerank']}")
    print(f"    gold prefix not offered (no bit help): {res['at10_fail_no_prefix_offered']}")
    print(f"    gold prefix offered but still >10 after boost: {res['at10_fail_prefix_offered_but_rerank_insufficient']}")

    print(f"\n(c) Degenerate offered-choice sets (bit adds nothing):")
    for tag, d in [("@10-fail", res["at10_degen"]), ("@all-fail", res["atall_degen"])]:
        print(f"  {tag} (n={d['n_fail']}):")
        print(f"    all {k} offered prefixes identical      : {d['all_prefixes_identical']}")
        print(f"    gold prefix already = rank-1 offered    : {d['gold_prefix_is_rank1_already']}")
        print(f"    both conditions                          : {d['both']}")
        print(f"    either (degenerate, union)               : {d['either_degenerate']}")

    if verbose_detail:
        print(f"\n  -- @10-failure detail (k={k}) --")
        for row in res["at10_detail"]:
            print(f"    {row['instance_id']:45s} choices={row['choices']} gold={row['gold_prefixes']} reachable={row['reachable']}")
        print(f"\n  -- @all-failure detail (k={k}) --")
        for row in res["atall_detail"]:
            print(f"    {row['instance_id']:45s} choices={row['choices']} gold={row['gold_prefixes']} reachable={row['reachable']}")


def main():
    rows = load_rows(PATH)
    print(f"Loaded {len(rows)} rows from {PATH}")

    # Primary run: k=4 (per spec)
    res4 = run_for_k(rows, 4)
    print_report(res4, verbose_detail=True)

    # Sensitivity: k=3 and k=6
    res3 = run_for_k(rows, 3)
    res6 = run_for_k(rows, 6)
    print_report(res3, verbose_detail=False)
    print_report(res6, verbose_detail=False)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY (sensitivity across k)")
    print(f"{'='*70}")
    header = f"{'k':>3} | {'@10 ceiling':>12} | {'@all ceiling':>13} | {'@10 after oracle':>17} | {'@10-fail degen(either)':>23} | {'@all-fail degen(either)':>24}"
    print(header)
    for res in (res3, res4, res6):
        print(f"{res['k']:>3} | {res['at10_ceiling_rate']:>12.4f} | {res['atall_ceiling_rate']:>13.4f} | "
              f"{res['at10_after_oracle_rerank']:>17.4f} | {res['at10_degen']['either_degenerate']:>10}/{res['n_at10_fail']:<10} | "
              f"{res['atall_degen']['either_degenerate']:>10}/{res['n_atall_fail']:<10}")


if __name__ == "__main__":
    main()
