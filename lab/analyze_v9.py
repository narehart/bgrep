"""Compare v7-baseline vs v8 (gemspec/rake/erb added to EXTENDED_EXTENSIONS,
no impl_prior carve-out) vs v9 (v8 + gemspec/.rake -> impl_prior 0.5) on the
44 Ruby + 43 JS/TS multilingual instances (rubyjs_instances.txt).

v7-baseline is sliced out of the full 300-instance multilingual_v7.jsonl
(pre-dates the .gemspec/.rake/.erb extension addition -- extensions=extended
there means the v7-era EXTENDED_EXTENSIONS, without the Ruby-sibling exts).
v8/v9 are the dedicated 87-instance reruns.

Reports File@1/@5/@10/@all for Ruby rows (should show v9 recovering toward
v7-baseline while not regressing @all vs v8), gained/lost lists at @1 for
v7-vs-v8 and v8-vs-v9, and a JS/TS identity check (v9 must equal v8 exactly
-- the gemspec/rake prior change is unreachable for non-Ruby paths)."""
import json
from pathlib import Path

BASE = Path("/private/tmp/claude-501/-Users-nicholasarehart-programming-projects-bgrep/3ab12e71-fab2-4a81-bb2b-84700d211ef2/scratchpad/bgrep_lab")
RESULTS = BASE / "results_swebench"

RUBY_REPO_PREFIXES = (
    "faker-ruby__faker", "fastlane__fastlane", "fluent__fluentd",
    "jekyll__jekyll", "jordansissel__fpm", "rubocop__rubocop",
)


def load(path):
    recs = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                continue
            recs[d["instance_id"]] = d
    return recs


def is_ruby(instance_id):
    return any(instance_id.startswith(p) for p in RUBY_REPO_PREFIXES)


def at_k(rec, k):
    gold = set(rec["gold_files"])
    returned = rec["returned_files"][:k] if k is not None else rec["returned_files"]
    return gold.issubset(set(returned))


def main():
    wanted = {ln.strip() for ln in (BASE / "tmp_analysis" / "rubyjs_instances.txt").read_text().splitlines() if ln.strip()}

    v7_all = load(RESULTS / "multilingual_v7.jsonl")
    v8 = load(RESULTS / "multilingual_rubyjs_v8.jsonl")
    v9 = load(RESULTS / "multilingual_rubyjs_v9.jsonl")

    ids = sorted(wanted & set(v7_all) & set(v8) & set(v9))
    ruby_ids = [i for i in ids if is_ruby(i)]
    jsts_ids = [i for i in ids if not is_ruby(i)]
    print(f"total instances compared: {len(ids)} (ruby={len(ruby_ids)} js/ts={len(jsts_ids)})\n")

    data = {"v7-baseline": v7_all, "v8": v8, "v9": v9}

    for group_name, group_ids in (("RUBY", ruby_ids), ("JS/TS", jsts_ids)):
        print(f"=== {group_name} (n={len(group_ids)}) ===")
        print(f"{'lane':<14} {'@1':>10} {'@5':>10} {'@10':>10} {'@all':>10}")
        rows = {}
        for name, recs in data.items():
            row = {}
            for k, label in [(1, "@1"), (5, "@5"), (10, "@10"), (None, "@all")]:
                n = sum(1 for i in group_ids if at_k(recs[i], k))
                row[label] = n
            rows[name] = row
            print(f"{name:<14} " + " ".join(f"{row[l]:>4}/{len(group_ids)} ({row[l]/len(group_ids):.3f})" for l in ("@1", "@5", "@10", "@all")))
        print()

    # gained/lost @1 and @all for RUBY: v8 vs v7-baseline, v9 vs v8, v9 vs v7-baseline
    print("=== RUBY gained/lost ===")
    for k, label in [(1, "@1"), (None, "@all")]:
        b7 = {i: at_k(v7_all[i], k) for i in ruby_ids}
        b8 = {i: at_k(v8[i], k) for i in ruby_ids}
        b9 = {i: at_k(v9[i], k) for i in ruby_ids}
        for cmp_name, base, cur in (
            ("v8 vs v7-baseline", b7, b8),
            ("v9 vs v8", b8, b9),
            ("v9 vs v7-baseline", b7, b9),
        ):
            gained = [i for i in ruby_ids if cur[i] and not base[i]]
            lost = [i for i in ruby_ids if base[i] and not cur[i]]
            print(f"  {cmp_name} {label}: gained={len(gained)} lost={len(lost)} net={len(gained) - len(lost)}")
            print(f"    gained: {gained}")
            print(f"    lost:   {lost}")
        print()

    # JS/TS identity check: v9 must equal v8 exactly (returned_files, all keys)
    print("=== JS/TS v9 vs v8 identity check ===")
    mismatches = []
    for i in jsts_ids:
        if v9[i]["returned_files"] != v8[i]["returned_files"]:
            mismatches.append(i)
    print(f"returned_files identical: {len(jsts_ids) - len(mismatches)}/{len(jsts_ids)}")
    if mismatches:
        print(f"  mismatched: {mismatches}")
        for i in mismatches[:5]:
            print(f"    {i}:")
            print(f"      v8: {v8[i]['returned_files'][:10]}")
            print(f"      v9: {v9[i]['returned_files'][:10]}")

    # sanity: success criteria
    print()
    ruby_at1 = sum(1 for i in ruby_ids if at_k(v9[i], 1)) / len(ruby_ids)
    ruby_atall = sum(1 for i in ruby_ids if at_k(v9[i], None)) / len(ruby_ids)
    print(f"v9 Ruby @1 = {ruby_at1:.3f} (target >= 0.341)")
    print(f"v9 Ruby @all = {ruby_atall:.3f} (target >= 0.795)")


if __name__ == "__main__":
    main()
