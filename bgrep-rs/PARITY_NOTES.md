# PARITY_NOTES.md

Every Python-semantics trap the Rust port had to reproduce byte-for-byte
against `lab/lanes2.py` + `lab/history.py` (the frozen v7 pipeline), plus how
each was resolved in `src/`. Ordered roughly by where you'd hit them reading
the pipeline top to bottom: filesystem walk -> tokenization -> corpus/BM25 ->
import graph -> select_files -> history mining -> region packing.

Validated by:
- `parity/shim_reference.py` micro-parity on 5 hand-picked httpx queries
  (exact ranked-file-list match).
- `parity/harness.py --suite lite --limit 40` (SWE-bench Lite gate):
  **40/40 exact-match** against the frozen `abl_bridges_v7.jsonl` expectations
  (astropy + django instances).

## 1. `str.splitlines()` vs Rust `str::lines()` (`src/pyutil.rs::py_splitlines`)

Python's `str.splitlines()` recognizes a much broader set of line boundaries
than `\n`/`\r\n`: `\r`, `\v` (0x0B), `\f` (0x0C), `\x1c`-`\x1e`, `\x85` (NEL),
U+2028 (LS), U+2029 (PS). This matters for `history.py`'s `git log` stdout
parsing (commit messages can contain any of these) and for line-splitting in
`pack_regions`. `py_splitlines` hand-replicates the exact boundary set and the
"no trailing empty element" rule.

## 2. `Path(rel).parent` renders `"."` for a top-level path (`py_parent`)

Rust's `Path::parent()` on `"foo.py"` yields `Some("")`, but
`str(Path("foo.py").parent)` in Python is `"."`. Every same-package /
same-directory grouping in `select_files` (`same_dir[str(Path(rel).parent)]`)
and the JS/TS relative-import resolver depend on this exact string, so
`py_parent`/`py_parent_name` special-case it.

## 3. `os.path.normpath(str(Path(base) / spec))` for JS/TS import resolution
   (`normpath_join`)

`build_import_graph`'s relative-import resolver joins a base directory with a
`./` or `../` specifier and lexically normalizes it (no filesystem access).
`normpath_join` replicates pathlib's part-dropping (bare `"."` components
vanish) plus `normpath`'s `".."`-against-preceding-component resolution,
including the "unresolvable leading `..` is kept as-is" edge case.

## 4. `sorted(Path.rglob("*"))` orders by **parts tuple**, not the joined string
   (`path_sort_key`, `Corpus::build`'s file collection)

`("test", "a.py") < ("test.py",)` under component-wise tuple comparison,
because `"test" < "test.py"` as a standalone string, even though naive
string comparison of the *joined* forms would put `"test.py"` first (`'.' <
'/'` in byte/codepoint order). Every place the corpus file list's order
matters (BM25 iteration for stable tie-breaks, `corpus.files` traversal
order) sorts by `rel.split('/')` component vectors, not the raw string.

## 5. `str.lower()` (`py_lower`)

Rust's `to_lowercase()` uses full Unicode case folding; Python's `str.lower()`
uses the Unicode default mapping. For the ASCII-dominated identifier/path
text this pipeline tokenizes, the two are byte-identical, so `py_lower` is a
thin wrapper kept only so any future non-ASCII divergence has one place to
patch (documented risk, not an observed one).

## 6. `Counter.most_common()` tie-breaking = insertion order, not hash order
   (`src/history.rs::most_common`, `OrderedCounter = IndexMap<String, i64>`)

Python's `Counter` is a `dict` subclass: `.most_common()` is a stable sort by
count descending, so ties preserve first-insertion order. `history.rs` uses
`IndexMap` everywhere a Python `dict`/`Counter` is ported (never `HashMap`,
whose Rust iteration order is unspecified and, without a fixed hasher, is
per-process-random via SipHash), and `most_common` is a stable
`sort_by(|a,b| b.1.cmp(&a.1))` over a Vec built by iterating that IndexMap in
insertion order -- exactly reproducing Python's tie-break.

## 7. `dict`/`set` iteration order and hash-randomization exposure
   (`core.rs`, throughout)

Almost the entire pipeline is safe from Python's per-process hash-seed
randomization because `lanes2.py`/`history.py` only ever iterate `dict`s
(insertion-ordered, deterministic) or *pre-sorted* material derived from a
`set` (e.g. `combinations(sorted(set(code_files)), 2)`). The Rust port mirrors
this file-by-file: `IndexMap` wherever Python code relies on dict insertion
order (BM25 score accumulation, TF/DF counters, cochange maps, the `Explain`
diagnostic pool), plain `HashMap`/`HashSet` only where the corresponding
Python data structure is *never* iterated in an order-sensitive way (pure
membership/lookup use).

The **one genuine exception** is `build_import_graph`'s adjacency map: in
Python, `edges[s]` is a raw `set[str]`, and `select_files`'s
`neighbors = list(edges.get(s, ())) + same_dir.get(...)` does iterate it,
feeding candidate `pool`/`owner` assignment (`if w > pool.get(c, 0.0):
pool[c] = w`). A raw Python `set`'s iteration order depends on hash-seed
randomization and insertion history, so which of several *equal-weight*
neighbors "wins" a `pool[c] = w` tie in the reference implementation is
itself nondeterministic across separate interpreter runs of `lanes2.py`. The
Rust port makes the deliberate, documented choice to use `BTreeSet` (sorted
iteration) for `EdgeMap`'s adjacency sets -- a fixed, reproducible tie-break
standing in for genuinely-undefined Python behavior, not a deviation from a
well-defined one. In practice `w` only differs when `s` (the owning source)
differs, and sources are processed in `lex_picks` order, so this tie-break is
rarely if ever live; the 40/40 exact-match SWE-bench Lite gate (which
includes several large-fan-out import graphs on django/astropy) is the
empirical confirmation it doesn't matter in practice. See `core.rs`'s
`EdgeMap`/`build_import_graph` doc comments (referred to elsewhere in the
codebase as "PARITY_NOTES.md item 2").

## 8. Stable sort + matching pre-sort order reproduces Python's `sorted()` ties

Every `ranked.sort_by(...)` / `weighted.sort_by(...)` call in `select_files`
and its helpers (lex-pick ranking, RM3-lite feedback-term weighting, the
structural-expansion candidate pool, msg-field top-up, docsbridge page
ranking) omits an explicit tie-break, exactly like the Python
`sorted(items, key=lambda kv: -kv[1])` calls it ports. This is safe *only*
because (a) Rust's `sort_by`/`sort_unstable_by`-adjacent `Vec::sort_by` is a
stable sort (same guarantee as Python's Timsort), and (b) the pre-sort
iteration order of the source `IndexMap`/`Vec` is itself already
Python-insertion-order-identical (see items 6-7). Get either half wrong and
ties silently reorder on any repo with real score collisions -- which is
exactly the class of bug this parity pass found (item 9).

## 9. THE BUG: `git log --pretty <value>` (two argv tokens) is rejected by git
   (`src/history.rs::mine_history`) -- **fixed during this pass**

`Command::new("git").args(["log", ..., "--pretty", &pretty, "--name-only"])`
passes `--pretty` and its format string as two separate argv elements. Git
does **not** accept `--pretty` with a space-separated value the way most
GNU-style CLI options do -- it only accepts `--pretty=<value>` (or the
short-hand having no separate-token form at all for this particular option).
Passed as two tokens, git parses `--pretty` as a bare flag (default format)
and then treats the following token (`"format:__C__%at%x00%an%x00%s%n%b"`) as
a revision/pathspec argument, which is not a valid object name, so the
process exits 128 with `fatal: invalid object name 'format'.` on stderr.

`mine_history`'s error handling (`Ok(o) if o.status.success() => o, _ =>
return HistoryData::default()`) swallows this silently, so **every** call to
`mine_history` returned an empty `msgs`/`cochange`/`meta` -- not just on large
repos. This was invisible in the initial 5-query httpx micro-parity check
(step 3) purely by luck: httpx's git history doesn't happen to add anything
via the cochange/msg-bm25 top-up channels for those specific 5 queries, so
Python-with-working-history and Rust-with-empty-history produced identical
`files` lists. It became visible immediately on the 40-task SWE-bench Lite
gate (0/40 exact-match, but 40/40 top-10-match -- the bug only ever affects
the *tail* additions, never the top lexical picks, which is exactly what
`select_files`'s "history is monotone-append-only" design guarantees).

Fix: build the format string as a single `--pretty=format:...` argv token:

```rust
let pretty = format!("--pretty=format:{SENTINEL}%at%x00%an%x00%s%n%b");
Command::new("git").args(["log", "--no-merges", "-n", &max_commits.to_string(), &pretty, "--name-only"])
```

Lesson for future ports: verify every subprocess invocation directly (`git
log ... > out; echo $?`) rather than trusting that "it compiled and returned
some files" means the command actually ran the way the caller intended --
this class of bug produces a *plausible*, non-crashing, silently-degraded
result (empty history is a legal, common state for a fresh/shallow repo), so
nothing about the program's behavior signals the failure.

## 10. `Path.suffix` vs `str.endswith` for extension filtering

`Corpus`'s file-walk filter checks `p.suffix in CODE_EXTENSIONS` (Python
`pathlib.Path.suffix`: the last dotted component of the *final path segment*
only -- a file named `a.b.py` has suffix `.py`, and a dotfile like
`.gitignore` has suffix `""`). `has_code_suffix`/`suffix_of` in `core.rs`
replicate this exactly, distinct from `is_code_file`'s simpler
`rel.ends_with(ext)` check used by the (separate, `swebench_driver2.py`
-mirroring) `list_current_files` pre-filter in `main.rs`, which the Python
reference (`shim_reference.py::_list_current_files`) also implements via
`p.suffix not in L.CODE_EXTENSIONS` -- both call sites use the pathlib
`.suffix` semantics, and `main.rs::list_current_files` matches via
`ends_with`, which is equivalent for this specific extension set (none of
`CODE_EXTENSIONS` is a suffix of another) but is a documented "happens to be
equivalent, not deliberately identical logic" spot.

## 11. Python `round()` is banker's rounding (round-half-to-even)

Used only in the `--explain` diagnostic dump (`py_round` in `core.rs`), which
mirrors `lanes2.py`'s `LAST_EXPLAIN` pool-score rounding
(`round(add_score(c), 4)`). Not on any scoring hot path, but the Explain
output is a debugging/parity tool in its own right (used throughout this
pass to localize the git log bug), so it needs to itself be trustworthy.

## 12. Regex lookahead: `_CAMEL_RE`'s `(?=[A-Z][a-z])` (`camel_matches`)

Rust's `regex` crate doesn't support lookahead. `subtokens`' camelCase
splitter (`camel_matches` in `core.rs`) is a hand-rolled character scanner
replicating `_CAMEL_RE = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+"`'s
exact match boundaries (e.g. `"HTTPServer"` -> `["HTTP", "Server"]`, the
lookahead is what keeps the trailing `S` of an all-caps run with the
following word instead of the run). Covered by
`core::tests::camel_matches_edge_cases`.

## Not a trap, but worth flagging: `personalized_pagerank` is dead code

`core.rs::personalized_pagerank` is a structural port of
`lanes2.py::personalized_pagerank`, but `select_files(use_ppr=True)` never
actually calls it in either the Python or the Rust pipeline -- the
"structural expansion" block reimplements a different additive
pool/owner/add_score scheme directly. Kept `#[allow(dead_code)]` for
completeness/documentation parity, not exercised by any live code path.

## Verified working (not a trap, confirmed by the 40/40 gate)

- BM25F term/file iteration order (`Corpus::bm25_params`): drives every term
  outer-loop, file inner-loop, matching `lanes2.py`'s direct
  `for rel in self.files` nesting, so `scores.entry(rel).or_insert(0.0) +=`
  accumulation order -- and therefore the `IndexMap`'s resulting insertion
  order for BM25-score ties -- matches Python's dict-insertion order exactly.
- `select_files`'s guarantee-0/1/2 additions ordering, RM3-lite feedback-term
  selection, cochange-strength promotion, msg-field top-up cutoff (`0.35 *
  msg_max`), and anchor/testbridge/docsbridge tail-only promotions all
  reproduce byte-for-byte once the history-mining bug (item 9) was fixed.
- `pack_regions`' token-budget packing (tiktoken `cl100k_base`, same encoder
  as the Python `tiktoken` package) produces identical region spans and
  bundle text on every gate task.
