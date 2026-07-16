# Changelog

All notable changes to `roust` are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Release infrastructure: tag-triggered PyPI (trusted publishing) + crates.io
  + GitHub Releases (`.github/workflows/release.yml`, `RELEASE.md`).

## [0.2.0] - Single Rust engine

- `pip install roust` now delivers the Rust binary directly (maturin
  `bindings = "bin"`), replacing the parallel Python implementation.
