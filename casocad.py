#!/usr/bin/env python3
"""Single-file launcher for casoCAD with an interactive GPU + backend menu.

Pick whichever you have:

    uv run casocad.py        # easiest: uv resolves the deps automatically
    python casocad.py        # if you already have the project venv active
    ./casocad.py             # it is executable

By default it shows a menu so you can pick which GPU and graphics backend to use,
then starts the CAD with that choice. To skip the menu:

    casocad.py --gpu nvidia --backend opengl
    casocad.py --default          # platform default, no questions asked
    casocad.py --list             # just print what was detected and exit

Why a launcher and not ``python app/main.py``: ``app/main.py`` uses a
package-relative import, so it must be imported as a package (done here via an
absolute ``from app.main import main``). And the GPU/backend choice is a set of
environment variables the driver only reads at startup — so once chosen, we
re-exec this script with them in place before any rendering context exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the project packages (app, core) importable regardless of the cwd.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from app.launcher import (  # noqa: E402
    _LAUNCHED_ENV,
    Backend,
    available_backends,
    build_env,
    describe,
    detect_gpus,
    interactive_select,
)


def _match(items, key: str, attr: str):
    key = key.lower()
    for item in items:
        if getattr(item, attr).lower() == key or getattr(item, attr).lower().startswith(key):
            return item
    return None


def _relaunch_with(env: dict[str, str]) -> "NoReturn":  # type: ignore[name-defined]
    """Set the chosen env vars and re-exec this script so the driver sees them."""
    child_env = dict(os.environ)
    child_env.update(env)
    child_env[_LAUNCHED_ENV] = "1"
    os.execve(sys.executable, [sys.executable, str(_ROOT / "casocad.py"), *sys.argv[1:]], child_env)


def _start_app() -> int:
    from app.main import main

    return main()


def main() -> int:
    parser = argparse.ArgumentParser(prog="casocad", description="Launch casoCAD.")
    parser.add_argument("--gpu", help="vendor or name fragment (e.g. nvidia, intel)")
    parser.add_argument("--backend", help="opengl | vulkan | metal | d3d11")
    parser.add_argument("--default", action="store_true", help="platform default, no menu")
    parser.add_argument("--list", action="store_true", help="list detected GPUs/backends and exit")
    args = parser.parse_args()

    # Second pass (already configured): just start the app.
    if os.environ.get(_LAUNCHED_ENV) == "1":
        return _start_app()

    gpus = detect_gpus()
    backends = available_backends()

    if args.list:
        if gpus:
            print("Detected GPUs:")
            for g in gpus:
                print(f"  - {g.name} [{g.vendor}, {'discrete' if g.discrete else 'integrated'}]")
        else:
            print("GPU selection is not available on this platform (the OS picks the GPU).")
        print("Available backends:")
        for b in backends:
            print(f"  - {b.label} ({b.key})")
        return 0

    # --default: skip selection entirely, let the platform decide.
    if args.default:
        os.environ[_LAUNCHED_ENV] = "1"
        return _start_app()

    # Flag-driven (non-interactive) selection.
    if args.gpu or args.backend:
        if args.gpu and not gpus:
            print("casocad: GPU selection isn't available on this platform.", file=sys.stderr)
            return 2
        if args.gpu:
            gpu = _match(gpus, args.gpu, "vendor") or _match(gpus, args.gpu, "name")
            if gpu is None:
                print(f"casocad: no GPU matching '{args.gpu}'. Try --list.", file=sys.stderr)
                return 2
        else:
            gpu = gpus[0] if gpus else None
        backend = _match(backends, args.backend, "key") if args.backend else backends[0]
        if backend is None:
            print(f"casocad: no backend matching '{args.backend}'. Try --list.", file=sys.stderr)
            return 2
        print(f"casocad: starting with {describe(gpu, backend)}")
        _relaunch_with(build_env(gpu, backend))

    # Interactive menu (only when attached to a terminal).
    if not sys.stdin.isatty():
        os.environ[_LAUNCHED_ENV] = "1"
        return _start_app()

    gpu, backend = interactive_select(gpus, backends)
    print(f"\ncasocad: starting with {describe(gpu, backend)}\n")
    _relaunch_with(build_env(gpu, backend))


if __name__ == "__main__":
    raise SystemExit(main())
