# Release runbook

`roust` ships three artifacts from a single tag push:

| Artifact | What it is | Where it goes |
|---|---|---|
| **Wheel** (`roust-X.Y.Z-py3-none-<platform>.whl`) | The compiled `roust` binary embedded in a wheel (maturin `bindings = "bin"`, the ruff/uv pattern). No Python code runs; `pip install roust` just unpacks the binary onto `PATH`. | PyPI |
| **sdist** (`roust-X.Y.Z.tar.gz`) | The Rust source tree, buildable with `pip install roust` on a platform without a prebuilt wheel (falls back to `cargo`/`maturin` building from source). | PyPI |
| **Crate** (`roust` on crates.io) | The `roust-rs/` Rust source, for `cargo install roust` and use as a library dependency. | crates.io |
| **GitHub Release** | The same raw platform binaries extracted out of the wheels (no need to unzip a wheel to get the executable), plus auto-generated release notes. | GitHub Releases |

All of it is built and published by `.github/workflows/release.yml`, triggered
by pushing a tag matching `v*`.

---

## One-time setup (USER ACTIONS)

These cannot be done by an agent — they require your PyPI and crates.io
credentials. Do both before the first tag push.

┌─────────────────────────────────────────────────────────────────────────┐
│ ACTION A: Configure a PyPI Trusted Publisher (no token needed)          │
│                                                                           │
│ The project has never been published, so there's no existing PyPI       │
│ project to attach a publisher to. Use a *pending* publisher instead:     │
│                                                                           │
│ 1. Go to https://pypi.org/manage/account/publishing/ (this is your      │
│    account's publishing page, not a project page — the project doesn't  │
│    exist yet).                                                          │
│ 2. Under "Add a new pending publisher", fill in:                        │
│      PyPI Project Name:     roust                                       │
│      Owner:                 narehart                                    │
│      Repository name:       roust                                       │
│      Workflow name:         release.yml                                 │
│      Environment name:      release                                    │
│ 3. Submit. Nothing is published or reserved yet — the pending publisher │
│    only activates (and claims the `roust` name) the first time          │
│    `release.yml` actually runs a real (non-dry-run) publish job, i.e.   │
│    the first `git push origin vX.Y.Z`. Until then, someone could in     │
│    principle still register `roust` on PyPI out from under this config; │
│    see issue #13 on urgency.                                            │
│ 4. No secret to add. The `publish-pypi` job in release.yml requests an  │
│    OIDC token itself (`permissions: id-token: write`) and PyPI          │
│    verifies it against the pending publisher config above — that's the │
│    entire trust mechanism, no `PYPI_API_TOKEN` anywhere.                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ ACTION B: crates.io account + CARGO_REGISTRY_TOKEN secret                │
│                                                                           │
│ 1. Go to https://crates.io and log in via GitHub OAuth (creates the     │
│    account on first login).                                             │
│ 2. Go to https://crates.io/settings/tokens and create a new API token.  │
│    Scope it to "publish-new" + "publish-update" if crates.io offers      │
│    scoped tokens; otherwise a default full-access token is fine.        │
│ 3. Add it as a repo secret (Settings > Secrets and variables > Actions  │
│    > New repository secret, or via CLI):                                │
│      gh secret set CARGO_REGISTRY_TOKEN --repo narehart/roust           │
│    (paste the token when prompted).                                     │
│ 4. Optional, only if you want to reserve the crates.io name before the  │
│    first tag: run `cargo login <token>` locally, then                   │
│    `cargo publish --manifest-path roust-rs/Cargo.toml` by hand once.    │
│    Not required — the `publish-crates-io` job does this on every real   │
│    release once the secret exists.                                     │
└─────────────────────────────────────────────────────────────────────────┘

---

## Per-release steps

1. **Bump the version in both places** (they must match — CI checks this,
   see below):
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `roust-rs/Cargo.toml`: `version = "X.Y.Z"`

2. **Update `CHANGELOG.md`**: move the `[Unreleased]` entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading (or add one if there's nothing under
   `Unreleased` yet).

3. **Commit** the version bump + changelog update.

4. **Dry-run the release workflow first** (recommended for anything other
   than a routine patch): go to Actions > Release > "Run workflow", pick the
   branch/commit you just pushed, leave `dry_run` checked (it's the
   default). This builds the sdist, all five wheel targets, and the raw
   binaries — exactly what a real release builds — but skips PyPI,
   crates.io, and the GitHub Release. Confirm all jobs are green before
   tagging.

5. **Tag and push**:
   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   The tag push triggers `release.yml` for real (`dry_run` is only a
   `workflow_dispatch` input; a tag push always runs the full publish path).

6. **Watch the run**: `gh run watch` or the Actions tab. Expect, in order:
   `plan` -> `sdist` + `wheels` (5 targets, parallel) -> `publish-pypi` +
   `publish-crates-io` + `github-release` (parallel, after builds finish).

7. **If `publish-crates-io` fails** (e.g. missing/expired token) while
   `publish-pypi` and `github-release` succeed: that's by design — it's a
   separate job so a crates.io problem doesn't block the PyPI release or
   the GitHub Release. Fix the token, then re-run just that job from the
   Actions UI ("Re-run failed jobs"), or dispatch `release.yml` manually
   against the `vX.Y.Z` tag ref with `dry_run` unchecked.

---

## Wheel/binary matrix

| Target | Runner | Notes |
|---|---|---|
| `x86_64-apple-darwin` | `macos-15-intel` | `macos-13` (the previous Intel runner) was retired; `macos-15-intel` is the standard-tier replacement, supported through Fall 2027. |
| `aarch64-apple-darwin` | `macos-14` | Apple Silicon, native (no cross-compilation). |
| `x86_64-unknown-linux-gnu` | `ubuntu-latest` | manylinux 2_17 via `PyO3/maturin-action`'s Docker container. |
| `aarch64-unknown-linux-gnu` | `ubuntu-latest` | manylinux 2_17, cross-compiled via `maturin-action`'s `ghcr.io/rust-cross/manylinux2014-cross:aarch64` container — no QEMU needed (that's only required for the official single-arch pypa images on a foreign host arch). |
| `x86_64-pc-windows-msvc` | `windows-latest` | Included: `maturin-action` builds `bin`-bindings wheels natively on the Windows host runner with no extra cross-compilation setup, so there was no reason to leave it out. |

Windows arm64 (`aarch64-pc-windows-msvc`) is not built — no GitHub-hosted
Windows-on-arm runner exists yet. Add it if/when one does.

---

## Version consistency check

`scripts/check_versions.py` compares the `version` in `pyproject.toml`
against `roust-rs/Cargo.toml` and fails if they differ. It runs as the
`version-check` job in `.github/workflows/ci.yml` on every push/PR to `main`,
so a drift is caught long before a tag is ever pushed — the release workflow
itself doesn't re-check, it just builds whatever `pyproject.toml`/`Cargo.toml`
say at the tagged commit.

Run it locally any time with `python3 scripts/check_versions.py`.

---

## Testing the workflow before credentials exist

Use `workflow_dispatch` with `dry_run` checked (the default) — see step 4
above. This is the only supported way to exercise `release.yml` without
`CARGO_REGISTRY_TOKEN` or the PyPI trusted publisher configured: the build
jobs (`sdist`, `wheels`) don't need either, and the three publish jobs are
skipped outright (`if: needs.plan.outputs.dry_run != 'true'`), so a missing
secret or publisher config can't fail a dry run.
