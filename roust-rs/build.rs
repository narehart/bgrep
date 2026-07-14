//! Embeds engine provenance (short git SHA + a roust-rs-scoped dirty flag)
//! into the binary at compile time, so `roust --version` and `--json` stats
//! always identify the exact engine build that produced them -- see the
//! `uv run roust` stale-wheel incident this guards against (issue #8's
//! sibling: `lab/tokenbench` shells out to `uv run roust`, which does NOT
//! rebuild on `roust-rs/src` changes until `uv sync --reinstall-package
//! roust` is run).

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let repo_root = Path::new(&manifest_dir)
        .parent()
        .expect("roust-rs/ has a parent directory (the repo root)")
        .to_path_buf();

    // Re-run this build script whenever HEAD moves to a new commit. HEAD
    // itself only changes on checkout/detach, not on every commit to the
    // current branch, so also watch the ref it points at when it's a
    // symbolic ref (the common case).
    let git_head = repo_root.join(".git").join("HEAD");
    println!("cargo:rerun-if-changed={}", git_head.display());
    if let Ok(head_contents) = std::fs::read_to_string(&git_head) {
        if let Some(ref_path) = head_contents.trim().strip_prefix("ref: ") {
            let ref_file = repo_root.join(".git").join(ref_path);
            println!("cargo:rerun-if-changed={}", ref_file.display());
        }
    }

    let sha = run_git(&repo_root, &["rev-parse", "--short", "HEAD"])
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    // Scoped to roust-rs/ paths only: a dirty tree elsewhere in the repo
    // (e.g. lab/ scratch files) must not mark the engine build as stale.
    let dirty_label = match run_git(&repo_root, &["status", "--porcelain", "--", "roust-rs/"]) {
        Some(out) if !out.is_empty() => "dirty",
        Some(_) => "clean",
        // Non-git context (e.g. a PyPI sdist build): nothing to report as
        // dirty against, so don't claim a false "dirty".
        None => "clean",
    };

    println!("cargo:rustc-env=ROUST_GIT_SHA={sha}");
    println!("cargo:rustc-env=ROUST_GIT_DIRTY={dirty_label}");
}

/// Runs `git <args>` with cwd = repo_root, returning trimmed stdout on
/// success (git installed, repo present, command exits 0), None otherwise.
fn run_git(repo_root: &Path, args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}
