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
        "install-launcher", help="add a double-clickable icon, and open it")
    launcher_cmd.add_argument("--uninstall", action="store_true")
    launcher_cmd.add_argument(
        "--no-open", action="store_true",
        help="install the icon without starting the app")

    # Registered so it appears in --help. Its arguments are handled before
    # argparse ever sees them -- see the note at the top of main().
    key_cmd = sub.add_parser(
        "key", help="store an OpenAI key for replies and export cleanup")
    key_cmd.add_argument("value", nargs="?", help="the key; prompted if omitted")

    sub.add_parser("check-audio",
                   help="list devices and diagnose capture "
                        "(--record, --selftest, --source)")

    sub.add_parser(
        "config", help="put an editable conf.yaml where you can reach it")

    sessions_cmd = sub.add_parser(
        "sessions", help="list, export or delete past recordings")
    sessions_cmd.add_argument("--export", type=int, metavar="N",
                              help="export recording N")
    sessions_cmd.add_argument("--delete", type=int, metavar="N",
                              help="delete recording N")
    sessions_cmd.add_argument("-o", "--output", help="where to write the export")
    sub.add_parser("version", help="print the version")

    # check-audio owns its own flags (--record, --selftest, --source), and
    # forwarding them through argparse does not work: REMAINDER drops leading
    # options, and redeclaring them here would let the two definitions drift
    # apart. Hand the tail over untouched instead.
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "check-audio":
        from scripts import check_audio
        return check_audio.main(raw[1:])

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

    if command == "key":
        return _key(args)

    if command == "config":
        return _config()

    from .app import main as run_app
    run_app()
    return 0


def _key(args) -> int:
    """
    Store the API key where the app will find it.

    Written to the user config directory. A key put beside the app is in
    site-packages once this is installed from a wheel -- not a place to keep a
    secret, and a reinstall would delete it.
    """
    import getpass
    from src.config import EnvConfig

    value = args.value or getpass.getpass("OpenAI key (sk-...): ")
    problem = EnvConfig.save_key(value)
    if problem:
        print(problem)
        return 1
    print(f"  Saved to {EnvConfig.key_file()}")
    return 0


def _config() -> int:
    """
    Copy the shipped conf.yaml somewhere the user can actually edit it.

    Installed from a wheel, the default sits in site-packages -- not a path
    anyone should be editing, and a reinstall would overwrite it anyway. The
    copy in the user config directory takes precedence once it exists.
    """
    import shutil
    from pathlib import Path
    from src.config import PathConfig

    target = Path(PathConfig.get_user_config_path()) / "conf.yaml"
    if target.exists():
        print(f"  Already yours to edit: {target}")
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(PathConfig.get_conf_file(), target)
    print(f"  Wrote {target}")
    print("  Edit it there; it overrides the shipped defaults.")
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

    # Said here because this is where somebody is being walked through the
    # install, and it is the one thing setup cannot arrange for them. Not a
    # requirement: transcription is local and needs no account.
    from src.config import EnvConfig
    if not EnvConfig.ensure_api_key():
        print()
        print("No OpenAI key is configured. Transcription does not need one --")
        print("it runs on this machine. The reply suggestions and the cleanup")
        print("pass at export do; the app will ask when you use them, or:")
        print(f"    echoai-helper key")
    return 0 if state.ready else 1


def _launcher(args) -> int:
    if sys.platform != "darwin":
        print("The launcher is a macOS application bundle.")
        return 0

    from src import launcher_macos as launcher
    if args.uninstall:
        return 0 if launcher.uninstall() else 1

    app = launcher.install()

    # Open it, unless asked not to. Installing an icon is not the goal --
    # having the app running is, and this is the last step of the three the
    # README gives you. Leaving the user to go and find the icon they have
    # just been told about is a step that exists only because nobody removed
    # it.
    if not args.no_open:
        import subprocess
        print("Starting it now...")
        try:
            subprocess.run(["open", str(app)], check=True, timeout=30)
        except Exception as e:
            print(f"Could not open it automatically: {e}")
            print(f"It is in Launchpad, or: open {str(app)!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
