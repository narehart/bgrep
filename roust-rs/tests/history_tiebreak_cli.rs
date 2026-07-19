//! E8b CLI contract tests against the real binary: the
//! --history-boost/--history-tiebreak mutual-exclusion error, the
//! --history-tiebreak domain validation, and end-to-end 10x output
//! determinism with the tie-break live on a constructed git repo (same
//! fixture discipline as tests/history_assoc.rs).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn git(repo: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["-c", "user.email=e8b@test.invalid", "-c", "user.name=e8b", "-c", "commit.gpgsign=false"])
        .args(args)
        .status()
        .expect("failed to run git");
    assert!(status.success(), "git {args:?} failed in {repo:?}");
}

fn roust(repo: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_roust"))
        .args(extra)
        .arg("widget summation")
        .arg(repo)
        .output()
        .expect("failed to run roust binary")
}

/// Tiny but real git repo (the binary indexes it, mines history, and can
/// serve the query end-to-end).
fn make_fixture_repo(tag: &str) -> PathBuf {
    let repo = std::env::temp_dir().join(format!("roust_e8b_cli_{tag}_{}", std::process::id()));
    std::fs::remove_dir_all(&repo).ok();
    std::fs::create_dir_all(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    std::fs::write(
        repo.join("pack.py"),
        "def alpha_widget(values):\n    total = 0\n    for v in values:\n        total += v\n    return total\n\n\ndef beta_widget(values):\n    m = 1\n    result = [v / m for v in values]\n    return result\n",
    )
    .unwrap();
    git(&repo, &["add", "-A"]);
    git(&repo, &["commit", "-q", "-m", "test: initial commit"]);
    std::fs::write(
        repo.join("pack.py"),
        "def alpha_widget(values):\n    total = 0\n    for v in values:\n        total = total + v\n    return total\n\n\ndef beta_widget(values):\n    m = 1\n    result = [v / m for v in values]\n    return result\n",
    )
    .unwrap();
    git(&repo, &["commit", "-q", "-am", "fix: widget summation determinism"]);
    repo
}

/// The two doorways for the association signal are mutually exclusive:
/// both nonzero must be a hard usage error (exit 2), not a silent
/// preference -- an eval sweep that passed both would otherwise produce an
/// uninterpretable arm.
#[test]
fn history_boost_and_tiebreak_are_mutually_exclusive() {
    let repo = make_fixture_repo("mutex");
    let out = roust(&repo, &["--history-boost", "0.3", "--history-tiebreak", "0.05"]);
    assert_eq!(out.status.code(), Some(2), "both flags nonzero must exit 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "stderr must name the mutual-exclusion contract, got: {stderr}"
    );
    // either flag alone (the other at its 0.0 default) is accepted.
    for extra in [&["--history-tiebreak", "0.05"][..], &["--history-boost", "0.3"][..]] {
        let out = roust(&repo, extra);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{extra:?} alone must run: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    // explicit zeros together are still the OFF state, not a conflict.
    let out = roust(&repo, &["--history-boost", "0.0", "--history-tiebreak", "0.0"]);
    assert_eq!(out.status.code(), Some(0), "explicit 0.0 + 0.0 is OFF, not a conflict");
    std::fs::remove_dir_all(&repo).ok();
}

/// --history-tiebreak is a relative epsilon: negative and non-finite
/// values are domain errors (exit 2).
#[test]
fn history_tiebreak_rejects_negative_and_non_finite() {
    let repo = make_fixture_repo("domain");
    // the `=` form: a bare `-0.05` value would be eaten by clap's own
    // "unexpected argument" usage error (also exit 2) before roust's domain
    // validation ever saw it, which is not the contract under test here.
    for bad in ["--history-tiebreak=-0.05", "--history-tiebreak=nan", "--history-tiebreak=inf"] {
        let out = roust(&repo, &[bad]);
        assert_eq!(out.status.code(), Some(2), "{bad} must exit 2");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("--history-tiebreak"), "stderr must name the flag, got: {stderr}");
    }
    std::fs::remove_dir_all(&repo).ok();
}

/// End-to-end determinism: 10 repeated binary invocations with the
/// tie-break live (mining, cache, packing, tie-group reorder included)
/// must emit byte-identical stdout.
#[test]
fn history_tiebreak_binary_output_deterministic_10x() {
    let repo = make_fixture_repo("det");
    let first = roust(&repo, &["--history-tiebreak", "0.05"]);
    assert_eq!(first.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&first.stderr));
    assert!(!first.stdout.is_empty(), "fixture query must produce a bundle");
    for run in 1..10 {
        let again = roust(&repo, &["--history-tiebreak", "0.05"]);
        assert_eq!(again.status.code(), Some(0));
        assert_eq!(
            String::from_utf8_lossy(&again.stdout),
            String::from_utf8_lossy(&first.stdout),
            "run {run} diverged from run 0"
        );
    }
    std::fs::remove_dir_all(&repo).ok();
}
