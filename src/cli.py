"""
src/cli.py

Command-line entry point, so the app can be installed with a package manager
rather than cloned.

    uv tool install echoai-helper
    echoai-helper                    # run it
    echoai-helper setup              # prepare audio routing
    echoai-helper install-launcher   # put an icon in Launchpad

Subcommands rather than flags because these are different jobs, and two of
them are things you do once and forget.
"""

import argparse
import sys


def _sessions(args) -> int:
    """List, export or delete past recordings."""
    from src.session import list_sessions, to_conversation, delete_session
    from pathlib import Path

    sessions = list_sessions()
    if not sessions:
        print("  No recordings yet.")
        return 0

    if args.delete is not None:
        target = sessions[args.delete]
        if delete_session(target.path):
            print(f"  Deleted {target.name}")
            return 0
        return 1

    if args.export is not None:
        target = sessions[args.export]
        conversation = to_conversation(target.path)
        out = Path(args.output or f"conversation_{target.path.stem}.md")
        if out.suffix.lower() == ".json":
            import json
            out.write_text(json.dumps(conversation, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        else:
            from src.export_markdown import render
            out.write_text(render(conversation), encoding="utf-8")
        print(f"  Wrote {out}")
        return 0

    for index, info in enumerate(sessions):
        print(f"  [{index}] {info.describe()}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="echoai-helper",
        description="Real-time meeting transcription, on-device.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="start the app (default)")

    setup_cmd = sub.add_parser(
        "setup", help="prepare macOS audio routing for recording")
    setup_cmd.add_argument("--status", action="store_true",
                           help="report what is configured, change nothing")
    setup_cmd.add_argument("--restore", action="store_true",
                           help="point system audio back at your speakers")

    launcher_cmd = sub.add_parser(
        "install-launcher", help="add a double-clickable icon")
    launcher_cmd.add_argument("--uninstall", action="store_true")

    sub.add_parser("check-audio", help="list devices and diagnose capture")

    sessions_cmd = sub.add_parser(
        "sessions", help="list, export or delete past recordings")
    sessions_cmd.add_argument("--export", type=int, metavar="N",
                              help="export recording N")
    sessions_cmd.add_argument("--delete", type=int, metavar="N",
                              help="delete recording N")
    sessions_cmd.add_argument("-o", "--output", help="where to write the export")
    sub.add_parser("version", help="print the version")

    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "version":
        from importlib.metadata import version, PackageNotFoundError
        try:
            print(version("echoai-helper"))
        except PackageNotFoundError:
            print("development")
        return 0

    if command == "setup":
        return _setup(args)

    if command == "install-launcher":
        return _launcher(args)

    if command == "sessions":
        return _sessions(args)

    if command == "check-audio":
        from scripts import check_audio          # noqa: F401
        return check_audio.main()

    from main import main as run_app
    run_app()
    return 0


def _setup(args) -> int:
    if sys.platform != "darwin":
        print("Audio routing setup is macOS-only.")
        return 0

    from src.audio import setup_macos as setup

    if args.restore:
        return 0 if setup.restore() else 1
    if args.status:
        state = setup.inspect()
        print(state.describe())
        return 0 if state.ready else 1

    state = setup.run(auto_activate=True)
    print()
    print(state.describe())
    return 0 if state.ready else 1


def _launcher(args) -> int:
    if sys.platform != "darwin":
        print("The launcher is a macOS application bundle.")
        return 0

    from src import launcher_macos as launcher
    if args.uninstall:
        return 0 if launcher.uninstall() else 1
    launcher.install()
    return 0


if __name__ == "__main__":
    sys.exit(main())
