#!/usr/bin/env python3
"""
scripts/setup_audio.py — prepare this Mac to record a meeting.

Installs the virtual audio device if it is missing, builds the Multi-Output
device that feeds both it and your speakers, and selects it. Replaces the
manual walk through Audio MIDI Setup described in docs/macos-audio-setup.md.

    .venv/bin/python scripts/setup_audio.py            # set everything up
    .venv/bin/python scripts/setup_audio.py --status   # just report
    .venv/bin/python scripts/setup_audio.py --restore  # undo the output switch

One administrator prompt, only if BlackHole still has to be installed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform != "darwin":
    print("This script configures macOS audio routing; nothing to do here.")
    raise SystemExit(0)

from src.audio import setup_macos as setup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true",
                        help="report what is set up, change nothing")
    parser.add_argument("--restore", action="store_true",
                        help="point system audio back at your speakers")
    parser.add_argument("--no-activate", action="store_true",
                        help="build the device but do not switch to it")
    args = parser.parse_args()

    if args.restore:
        return 0 if setup.restore() else 1

    if args.status:
        state = setup.inspect()
        print(state.describe())
        print()
        print("  ready" if state.ready
              else "  not ready — run this script without --status")
        return 0 if state.ready else 1

    print("Preparing audio capture...\n")
    state = setup.run(auto_activate=not args.no_activate)
    print()
    print(state.describe())
    print()
    if state.ready:
        print("  Ready. Meeting audio will be both heard and recorded.")
        print("  Run with --restore afterwards to put the output back.")
        return 0
    print("  Not ready. See the notes above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
