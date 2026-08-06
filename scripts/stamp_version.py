#!/usr/bin/env python3
"""Write the release version into the two files that carry it.

Run by `.github/workflows/release.yml` with `VERSION` set from the git tag, so a
release is always correctly versioned without a manual bump landing on main
first. Runnable by hand too — `VERSION=1.2.3 python3 scripts/stamp_version.py` —
which is the point: a one-line `sed` in a workflow is only ever tested by
cutting a release.

The two places:

- `src-tauri/tauri.conf.json` — the version in the installer and the app window.
- the **root** `Cargo.toml`, under `[workspace.package]`. Not
  `src-tauri/Cargo.toml`, which only says `version.workspace = true`.

The Cargo edit is deliberately scoped to the `[workspace.package]` section
rather than "the first version line in the file". An earlier attempt used
sed's `0,/re/` range, which is a GNU extension: it worked on the Windows runner
and silently did nothing on a Mac, leaving a release stamped with the old
version and no error to notice.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARGO = ROOT / "Cargo.toml"
TAURI_CONF = ROOT / "src-tauri" / "tauri.conf.json"

# Cargo and Tauri both want a plain semver triple; a tag like `v1.2` or
# `v1.2.3-beta` would build and then fail deep inside the bundler.
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def stamp_cargo(version: str) -> None:
    text = CARGO.read_text()
    section = re.search(r"^\[workspace\.package\]$", text, flags=re.M)
    if not section:
        raise SystemExit(f"{CARGO}: no [workspace.package] section to stamp")

    head, tail = text[: section.end()], text[section.end() :]
    # Stop at the next section header so a later `version` cannot be hit.
    next_section = re.search(r"^\[", tail, flags=re.M)
    body_end = next_section.start() if next_section else len(tail)
    body, rest = tail[:body_end], tail[body_end:]

    body, count = re.subn(
        r'^version\s*=\s*".*"$', f'version = "{version}"', body, count=1, flags=re.M
    )
    if count != 1:
        raise SystemExit(f"{CARGO}: no version key under [workspace.package]")

    CARGO.write_text(head + body + rest)


def stamp_tauri(version: str) -> None:
    config = json.loads(TAURI_CONF.read_text())
    config["version"] = version
    TAURI_CONF.write_text(json.dumps(config, indent=2) + "\n")


def main() -> int:
    version = os.environ.get("VERSION", "").strip()
    if not version:
        raise SystemExit("VERSION is not set")
    if not SEMVER.match(version):
        raise SystemExit(f"VERSION {version!r} is not a plain x.y.z semver")

    stamp_cargo(version)
    stamp_tauri(version)
    print(f"stamped {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
