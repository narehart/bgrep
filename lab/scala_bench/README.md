# scala_bench — mined Scala file-localization benchmark

60 instances across 6 repos, mined 2026-07-12. First Scala localization
benchmark used in this lab (all prior work — archex, SWE-bench Lite — is
Python/JS/Java/Go/Rust). Same instance shape as SWE-bench:
`{instance_id, repo, base_commit, problem_statement, gold_files}`.

## Method

`mine_scala.py`, driven by the `gh` CLI:

1. Search each repo for merged PRs linked to an issue
   (`is:pr is:merged linked:issue`).
2. Recover the issue each PR closes by regex-matching
   `Fixes/Closes/Resolves #N` (or a full issue URL) in the PR body.
3. Keep the pair only if all of:
   - the linked issue body is **>= 200 chars** of real text (excludes
     template stubs / one-liners);
   - the PR is **merged** and has a resolvable **base commit sha** (the PR's
     `base.sha`, i.e. the tree state immediately before the fix — temporally
     consistent with the issue, since the PR is what closes it);
   - the PR touches **1-5 non-test `.scala` files**, with the file's own diff
     not being comment/blank-only. Test paths are excluded via a path/suffix
     regex (`test/`, `tests/`, `it/`, `src/test/`, `*Test.scala`,
     `*Tests.scala`, `*Spec.scala`, `*Specs.scala`, `*TestSuite.scala`).
4. `problem_statement` = issue title + issue body (verbatim, unedited).
   `gold_files` = the surviving non-test `.scala` paths from the PR diff.

10 repos were queried; the first 6 that could each fill a 10-instance cap
supplied the full 60-instance set, so `apache/pekko`, `circe/circe`,
`com-lihaoyi/mill`, and `sbt/sbt` were never reached (see `mine_scala.py`
`REPOS` list and `--target-total`).

## Repos (10 instances each, 60 total)

| repo | scanned PRs | accepted | main rejection reasons |
|---|---|---|---|
| akka/akka-core | 28 | 10 | no linked issue (11), file count out of 1-5 range (4) |
| typelevel/cats | 21 | 10 | file count out of range (5), no linked issue (4) |
| typelevel/cats-effect | 30 | 10 | file count out of range (9), issue body too short (8) |
| playframework/playframework | 13 | 10 | file count out of range (3) |
| scala/scala3 | 13 | 10 | file count out of range (3) |
| zio/zio | 20 | 10 | file count out of range (4), no linked issue (3), issue body too short (3) |

Full per-repo funnel in `mine_stats.json`; full mining trace in
`mine_run.log`.

## Date mined

2026-07-12, against each repo's current default branch history at mining
time. Re-running `mine_scala.py` later will surface a different, larger PR
population (more merged-and-linked PRs accumulate over time) and is not
expected to reproduce this exact file.

## Spot-check (this commit)

3 instances checked at seed indices 5, 25, 50 of `scala_loc.jsonl`
(`akka__akka-core-30974`, `typelevel__cats-effect-4343`,
`scala__scala3-26441`): each repo shallow-cloned and checked out at
`base_commit`, all gold files confirmed present at that sha and are non-test
`.scala` files, all three `problem_statement`s read as real, specific GitHub
issues (stack traces / repro snippets / expected-vs-actual, not stubs).

One finding worth flagging: `akka__akka-core-30974`'s single gold file is
`project/AkkaDisciplinePlugin.scala`, a build-config file, not application
code — because the PR's actual fix touched only doc/sample files under
`akka-docs/src/test/...`, which are excluded by the test-path filter (they
live under a `src/test` root even though they're compiled docs samples, not
unit tests). The build-config edit that's left (relaxing a Scala 3
warnings-as-errors setting) is a genuine part of the merged fix and does
correspond to the issue, but it illustrates a real limitation of test-path
exclusion on repos that put "doc sample" sources under `src/test`: gold can
end up thin or config-only rather than pointing at the code that actually
implements the fix. See caveats below.

## Caveats

- **No solvability validation.** Unlike SWE-bench, instances here were not
  filtered by "does an LLM actually reproduce/fix this from the issue text
  alone" — only by the mechanical issue/PR/file-count criteria above.
  Retrieval-difficulty and fix-difficulty are unmeasured.
- **Retrieval-eval only.** `gold_files` marks *localization* targets (which
  files to read/edit), not a patch to apply or tests to run. There is no
  environment/harness for execution-based evaluation, unlike SWE-bench's
  Docker images and FAIL_TO_PASS tests.
- **Gold = fix-touched files**, not necessarily the minimal or unique correct
  fix location. A PR can touch a file for reasons only loosely related to the
  issue (refactor swept up in the same commit) or, as seen in the spot-check,
  leave gold thin because the most relevant files were filtered out as
  "test" paths.
- **Test-path filter is heuristic** (regex on path/suffix) and can misclassify
  doc-sample or fixture code that happens to live under `src/test` or a
  `*Spec.scala`-named file that is not actually a unit test.
- **1-5 gold-file band is a mining artifact**, not a claim about realistic fix
  size; it exists to keep instances answerable by file-localization (excludes
  both trivial 0-file and repo-wide sweeping PRs).
- Repo set is skewed toward FP/effects-heavy Typelevel-adjacent ecosystems
  (cats, cats-effect, zio) plus akka, play, and the Scala 3 compiler itself —
  not a representative sample of the broader Scala ecosystem.
