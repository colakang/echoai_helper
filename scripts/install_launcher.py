#!/usr/bin/env python3
"""
scripts/install_launcher.py — put EchoAI Helper in Launchpad and the Dock.

A command-line install is cheap to ship but leaves you opening a terminal
every time you want to take notes in a meeting. This writes a small .app
bundle in ~/Applications that launches the install you already have.

    .venv/bin/python scripts/install_launcher.py
    .venv/bin/python scripts/install_launcher.py --uninstall

No signing, no notarization, no administrator rights. Gatekeeper only
enforces on files carrying com.apple.quarantine, which is applied to things
downloaded from the internet -- a bundle created here does not have it.

The bundle contains a shell script, not a copy of the program, so updating
through your package manager updates what the icon launches.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform != "darwin":
    print("This creates a macOS application bundle; nothing to do here.")
    raise SystemExit(0)

from src import launcher_macos as launcher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        installed = launcher.is_installed()
        print(f"  {launcher.bundle_path()}: "
              f"{'installed' if installed else 'not installed'}")
        return 0 if installed else 1

    if args.uninstall:
        return 0 if launcher.uninstall() else 1

    launcher.install()
    return 0


if __name__ == "__main__":
    sys.exit(main())
