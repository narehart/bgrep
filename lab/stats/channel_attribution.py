#!/usr/bin/env python3
"""Channel-attribution conditioning analysis (paper eval wave, final).

Question: the raw BM25-only ablation shows a REGION-level advantage over the
shipped defaults (all channels on). Is that advantage real (channels hurt
region quality), or a composition effect (the extra channels recover HARDER
files, whose regions are harder to localize, dragging the defaults' region
averages down)?

Method: for each pair (defaults vs bm25-only, on Lite-300 and Verified-407),
restrict BOTH arms to the INTERSECTION of instances that are FILE-correct in
BOTH arms, then recompute FUNCTION and LINE correct-rates on that identical
instance set. On identical instances, any remaining gap is channel-caused;
if the gap vanishes, the raw gap was composition.

Per-instance signals:
  FILE  correct  = all_gold_files_retrieved in the prediction JSONL
                   (errors / missing -> False; same convention as
                   agentless_metric_verified.compute_file_level).
  LINE  correct  = hunk_line_recall == 1.0 in the prediction JSONL
                   (exact all-or-nothing, same as compute_line_level).
  FUNCTION correct = per-instance `correct` flag in the metric JSON's
                   function.detail array (exact AST metric, precomputed).
                   Intersection instances absent from EITHER arm's detail
                   (old-convention exclusions, e.g. git_show failures) are
                   dropped from the FUNCTION denominator of BOTH arms and
                   counted in n_function_excluded.

Usage:
    python lab/stats/channel_attribution.py \
        --out lab/stats/channel_attribution.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RR = REPO_ROOT / "lab" / "results_regions"

PAIRS = {
    "lite300": {
        "defaults": {"pred": RR / "full300_v11.jsonl",
                     "metric": RR / "agentless_metric_v5.json"},
        "bm25": {"pred": RR / "lite300_bm25.jsonl",
                 "metric": RR / "agentless_metric_lite_bm25.json"},
    },
    "verified407": {
        "defaults": {"pred": RR / "full407_verified_new.jsonl",
                     "metric": RR / "agentless_metric_verified_new.json"},
        "bm25": {"pred": RR / "verified407_bm25.jsonl",
                 "metric": RR / "agentless_metric_verified_bm25.json"},
    },
}


def load_arm(pred_path: Path, metric_path: Path) -> dict[str, dict]:
    per: dict[str, dict] = {}
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            iid = r["instance_id"]
            per[iid] = {
                "file": bool(r.get("all_gold_files_retrieved", False)) and r.get("error") is None,
                "line": r.get("hunk_line_recall") == 1.0 and r.get("error") is None,
                "function": None,
            }
    metric = json.load(open(metric_path))
    for d in metric["all_instances"]["function"]["detail"]:
        if d["instance_id"] in per:
            per[d["instance_id"]]["function"] = bool(d["correct"])
    return per


def pct(x: float) -> float:
    return round(100.0 * x, 2)


def analyze_pair(name: str, cfg: dict) -> dict:
    arms = {a: load_arm(c["pred"], c["metric"]) for a, c in cfg.items()}
    ids = set(arms["defaults"]) & set(arms["bm25"])
    assert set(arms["defaults"]) == set(arms["bm25"]), f"{name}: instance sets differ"

    inter = sorted(i for i in ids
                   if arms["defaults"][i]["file"] and arms["bm25"][i]["file"])
    fn_ok = [i for i in inter
             if arms["defaults"][i]["function"] is not None
             and arms["bm25"][i]["function"] is not None]

    out = {
        "n_total": len(ids),
        "n_file_correct": {a: sum(v["file"] for v in arms[a].values()) for a in arms},
        "n_intersection_file_correct": len(inter),
        "n_function_excluded_in_intersection": len(inter) - len(fn_ok),
        "conditioned": {},
        "raw_full_set": {},
    }
    for a in ("defaults", "bm25"):
        arm = arms[a]
        out["conditioned"][a] = {
            "function_pct": pct(sum(arm[i]["function"] for i in fn_ok) / len(fn_ok)) if fn_ok else None,
            "function_n": len(fn_ok),
            "function_n_correct": sum(arm[i]["function"] for i in fn_ok),
            "line_pct": pct(sum(arm[i]["line"] for i in inter) / len(inter)) if inter else None,
            "line_n": len(inter),
            "line_n_correct": sum(arm[i]["line"] for i in inter),
        }
        out["raw_full_set"][a] = {
            "file_pct": pct(sum(v["file"] for v in arm.values()) / len(arm)),
            "function_pct": pct(sum(bool(v["function"]) for v in arm.values()) / len(arm)),
            "line_pct": pct(sum(v["line"] for v in arm.values()) / len(arm)),
        }
    c = out["conditioned"]
    out["delta_defaults_minus_bm25_conditioned"] = {
        "function_pp": round(c["defaults"]["function_pct"] - c["bm25"]["function_pct"], 2),
        "line_pp": round(c["defaults"]["line_pct"] - c["bm25"]["line_pct"], 2),
    }
    # paired discordant counts within the intersection (for the writeup)
    out["discordant_in_intersection"] = {
        "function": {
            "defaults_only": sum(arms["defaults"][i]["function"] and not arms["bm25"][i]["function"] for i in fn_ok),
            "bm25_only": sum(arms["bm25"][i]["function"] and not arms["defaults"][i]["function"] for i in fn_ok),
        },
        "line": {
            "defaults_only": sum(arms["defaults"][i]["line"] and not arms["bm25"][i]["line"] for i in inter),
            "bm25_only": sum(arms["bm25"][i]["line"] and not arms["defaults"][i]["line"] for i in inter),
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "lab" / "stats" / "channel_attribution.json")
    args = ap.parse_args()

    report = {
        "question": "is the raw BM25-only region advantage a composition effect "
                    "(channels recover harder files) or a real channel penalty?",
        "method": "FUNCTION/LINE correct-rates recomputed on the intersection of "
                  "file-correct instances in both arms of each pair",
        "pairs": {name: analyze_pair(name, cfg) for name, cfg in PAIRS.items()},
        "sources": {name: {a: {"pred": str(c["pred"].relative_to(REPO_ROOT)),
                               "metric": str(c["metric"].relative_to(REPO_ROOT))}
                           for a, c in cfg.items()}
                    for name, cfg in PAIRS.items()},
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pairs"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
