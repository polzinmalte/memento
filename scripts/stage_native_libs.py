#!/usr/bin/env python3
"""Collect the native DLLs the app needs at startup into `src-tauri/bundled/`.

memento links sherpa-onnx as a *shared* library, so `sherpa-onnx-c-api.dll` and
its dependencies have to sit next to `memento-app.exe` or Windows refuses to
start the process at all — before `main`, with a system dialog rather than
anything the app could report:

    Die Ausführung des Codes kann nicht fortgesetzt werden, da
    sherpa-onnx-c-api.dll nicht gefunden wurde.

Tauri only bundles the executable and whatever `bundle.resources` names, and
nothing named these. They are not in the repository either: `sherpa-rs-sys`
downloads them per platform. So they are staged into one directory that
`tauri.conf.json` can point at with a path that does not depend on the build
profile.

Run by `.github/workflows/release.yml` after the release build. Runnable by hand
— `python3 scripts/stage_native_libs.py` — which is the point: a copy step that
only ever runs while cutting a release is a copy step nobody has tested.

It fails when a DLL is missing rather than bundling a partial set, because the
symptom of a partial set is the dialog above on a user's machine, long after CI
went green.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "src-tauri" / "bundled"

# The shared sherpa-onnx distribution, as of sherpa-rs-sys 0.6.8 / sherpa-onnx
# v1.12.9. `cargs` and `sherpa-onnx-cxx-api` are not called from memento, but
# `sherpa-onnx-c-api.dll` imports them, and an import is resolved at load time
# whether or not anything ever calls through it.
REQUIRED = [
    "sherpa-onnx-c-api.dll",
    "sherpa-onnx-cxx-api.dll",
    "cargs.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
]


def search_roots(profile: str) -> list[Path]:
    """Where the DLLs may be, best first.

    The build script copies them beside the build output, which is the normal
    case. The download cache is the fallback for a restored CI cache, where the
    build script is fingerprint-fresh and never re-ran to do that copying.
    """
    roots = [ROOT / "target" / profile]
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        roots.append(Path(local_appdata) / "sherpa-rs")
    return roots


def find(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.is_dir():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        # The cache nests the distribution under a target triple and a checksum.
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


def main() -> int:
    profile = os.environ.get("PROFILE", "release")
    roots = search_roots(profile)

    STAGING.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for name in REQUIRED:
        source = find(name, roots)
        if source is None:
            missing.append(name)
            continue
        shutil.copy2(source, STAGING / name)
        print(f"staged {name} from {source.parent}")

    if missing:
        print(
            f"error: could not find {', '.join(missing)} in "
            f"{', '.join(str(r) for r in roots)}",
            file=sys.stderr,
        )
        print(
            "The app will not start without them. Build first so the "
            "sherpa-rs-sys build script downloads and copies them.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(REQUIRED)} DLL(s) staged in {STAGING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
