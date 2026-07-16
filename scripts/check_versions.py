#!/usr/bin/env python3
"""Fail if pyproject.toml and roust-rs/Cargo.toml declare different versions.

Plain regex on purpose -- no toml dependency required, works on any Python 3.
Both files are expected to contain exactly one top-level `version = "X.Y.Z"`
line (dependency version pins use `name = "X.Y.Z"`, not a bare `version` key,
so a single regex pass across the whole file is unambiguous today). If either
file ever grows a second `version = "..."` line, tighten this to scope the
search to the relevant TOML table.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def read_version(path: pathlib.Path) -> str:
    text = path.read_text()
    match = VERSION_RE.search(text)
    if not match:
        print(f"error: no `version = \"...\"` line found in {path}", file=sys.stderr)
        sys.exit(1)
    return match.group(1)


def main() -> None:
    pyproject = ROOT / "pyproject.toml"
    cargo_toml = ROOT / "roust-rs" / "Cargo.toml"

    py_version = read_version(pyproject)
    rs_version = read_version(cargo_toml)

    if py_version != rs_version:
        print(
            f"error: version mismatch -- {pyproject} has {py_version!r}, "
            f"{cargo_toml} has {rs_version!r}. Bump both to the same version.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"ok: pyproject.toml and roust-rs/Cargo.toml both at {py_version}")


if __name__ == "__main__":
    main()
