#!/usr/bin/env python3
"""Extract the raw roust binary from a maturin `bin`-bindings wheel and
archive it for GitHub Release attachment.

The wheel already embeds the platform binary at
`<name>-<version>.data/scripts/roust[.exe]` (that's the whole point of
`bindings = "bin"`); this just pulls it back out so the GitHub Release can
ship the raw binary alongside the wheel, rather than making users unzip a
wheel to get it.

Usage: package_binary.py <wheel> <target-triple> <output-dir>
Writes <output-dir>/roust-<target-triple>.tar.gz, or .zip when
<target-triple> contains "windows".
"""

import pathlib
import sys
import tarfile
import zipfile


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: package_binary.py <wheel> <target-triple> <output-dir>", file=sys.stderr)
        sys.exit(1)

    wheel_path, target, out_dir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    is_windows = "windows" in target

    with zipfile.ZipFile(wheel_path) as whl:
        candidates = [
            n for n in whl.namelist()
            if n.endswith("/scripts/roust") or n.endswith("/scripts/roust.exe")
        ]
        if not candidates:
            print(f"error: no roust binary found in {wheel_path}: {whl.namelist()}", file=sys.stderr)
            sys.exit(1)
        member = candidates[0]
        binary_name = "roust.exe" if member.endswith(".exe") else "roust"
        data = whl.read(member)

    binary_path = out_dir / binary_name
    binary_path.write_bytes(data)
    if not is_windows:
        binary_path.chmod(0o755)

    if is_windows:
        archive_path = out_dir / f"roust-{target}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(binary_path, binary_name)
    else:
        archive_path = out_dir / f"roust-{target}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(binary_path, arcname=binary_name)

    binary_path.unlink()
    print(f"wrote {archive_path}")


if __name__ == "__main__":
    main()
