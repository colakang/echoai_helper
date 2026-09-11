"""
src/llm/cli_provider.py

Answer through a locally installed coding-agent CLI (`claude`, `codex`,
`gemini`) instead of an API key.

The draw is billing: those tools authenticate against a subscription you may
already pay for, so a long meeting costs nothing extra per token. The cost is
latency. Each answer spawns a process and pays its startup: measured on this
machine, `claude -p` returns in 3.8-5.5s against roughly 1s for the same
question over the OpenAI API.

That number decides where this belongs:

  meeting notes      fine. The polish pass runs after a sentence is already
                     finished and nobody is waiting on it.
  live interview     unusable. The point is a prompt while the other person
                     is still talking, and 4s plus transcription latency
                     lands well after the moment has passed.

Two further limits worth knowing before relying on it in a real meeting:
subscription rate limits can throttle mid-session in a way a metered API key
will not, and the CLI must be installed and already logged in -- there is no
key to hand it.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Generator, List, Optional

from .llm_provider import LLMProvider

# Generous: a cold CLI start plus a long answer. Better to wait than to
# truncate a reply that was nearly finished. A 25-line polish batch measured
# 20s here, so this leaves ample headroom for a slow response.
DEFAULT_TIMEOUT_S = 300

# Tools these agents would otherwise be free to reach for. They are here to
# answer a question about a transcript, not to touch the machine, and every
# tool they consider costs a round trip: an 8-line batch took 70s with them
# available and 20s for 25 lines without.
NO_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebFetch", "WebSearch", "Task", "TodoWrite", "NotebookEdit"]


class CLIProvider(LLMProvider):
    """Runs a local agent CLI once per request."""

    COMMANDS = {
        # `claude -p` is the documented non-interactive mode.
        "claude": {
            # --strict-mcp-config skips loading the user's MCP servers,
            # which are irrelevant here and cost 1.5s of the ~2.9s startup.
            "argv": ["claude", "-p", "--strict-mcp-config",
                     "--disallowed-tools", ",".join(NO_TOOLS)],
            "stdin": True,
        },
        # codex refuses to run outside a trusted directory unless told not to
        # care, which it has no reason to here -- it is answering a question,
        # not touching a repo.
        "codex": {
            "argv": ["codex", "exec", "--skip-git-repo-check"],
            "stdin": True,
            # codex prints a session banner to stdout -- provider, sandbox,
            # reasoning effort, session id -- and then the answer, so reading
            # stdout gives a reply with several lines of preamble glued to the
            # front. Every polish batch failed on it, because the cleanup
            # expects exactly as many lines back as it sent.
            #
            # -o writes just the final message to a file. Nothing to parse.
            "answer_file_flag": "--output-last-message",
        },
        "gemini": {
            "argv": ["gemini", "-p"],
            "stdin": True,
        },
    }

    def __init__(self, command: str = "claude", model: Optional[str] = None,
                 timeout: int = DEFAULT_TIMEOUT_S, extra_args=None):
        if command not in self.COMMANDS:
            raise ValueError(
                "unknown CLI {!r}; known: {}".format(
                    command, ", ".join(sorted(self.COMMANDS))))
        self.command = command
        self.model = model
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    # -- LLMProvider -------------------------------------------------------

    def get_model_name(self) -> str:
        return "{}{}".format(self.command,
                             ":" + self.model if self.model else "")

    # Where these CLIs actually install, for when PATH does not say.
    #
    # A double-clicked .app inherits /usr/bin:/bin:/usr/sbin:/sbin and nothing
    # else -- no shell profile is read, so every one of these is invisible to
    # shutil.which. That is not a corner case: `claude` installs to
    # ~/.local/bin by default, which no GUI app has ever had on its PATH. The
    # symptom was that the cleanup backend offered only the API, with the CLI
    # greyed out, on a machine where the CLI was installed and working.
    EXTRA_BIN_DIRS = (
        "~/.local/bin",          # claude's own installer
        "/opt/homebrew/bin",     # Homebrew, Apple silicon
        "/usr/local/bin",        # Homebrew, Intel; npm -g default
        "~/.npm-global/bin",     # npm with a user prefix
        "~/bin",
        "~/.bun/bin",
        "~/.volta/bin",
    )
    # nvm puts one bin directory per installed node version.
    NVM_GLOB = "~/.nvm/versions/node/*/bin"

    @classmethod
    def locate(cls, command: str) -> Optional[str]:
        """Absolute path to `command`, or None. PATH first, then the usual places."""
        found = shutil.which(command)
        if found:
            return found

        directories = [os.path.expanduser(d) for d in cls.EXTRA_BIN_DIRS]
        directories += sorted(glob.glob(os.path.expanduser(cls.NVM_GLOB)),
                              reverse=True)          # newest node first
        for directory in directories:
            candidate = os.path.join(directory, command)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    @classmethod
    def available(cls) -> List[str]:
        """Which of the known CLIs are installed, in COMMANDS order."""
        return [name for name in cls.COMMANDS if cls.locate(name)]

    def executable(self) -> Optional[str]:
        return self.locate(self.command)

    def validate_config(self) -> bool:
        if self.locate(self.command) is None:
            print("[CLIProvider] {!r} was not found on PATH or in {}".format(
                self.command, ", ".join(self.EXTRA_BIN_DIRS)))
            return False
        return True

    def generate_response(self, messages: List[Dict[str, str]],
                          temperature: float = 0.6, stream: bool = True,
                          **kwargs) -> Generator[str, None, None]:
        """
        Yield the answer.

        Yields once, not progressively: these CLIs emit a whole reply, and
        temperature is not exposed by any of them. The caller's streaming
        contract is honoured by yielding a single chunk.
        """
        argv = list(self.COMMANDS[self.command]["argv"])
        # argv[0] is the bare name; replace it with the resolved path, or the
        # subprocess would fail for the same reason the lookup did.
        resolved = self.locate(self.command)
        if resolved:
            argv[0] = resolved
        if self.model:
            argv += ["--model", self.model]
        argv += self.extra_args

        answer_flag = self.COMMANDS[self.command].get("answer_file_flag")

        try:
            # Run from an empty directory. These CLIs read the working
            # directory as project context -- CLAUDE.md, the repo, whatever is
            # around -- and none of it is relevant to answering a question
            # about a transcript. Left in the project directory, an 8-line
            # batch took over 90s and timed out.
            with tempfile.TemporaryDirectory(prefix="echoai-llm-") as workdir:
                answer_path = os.path.join(workdir, "answer.txt")
                if answer_flag:
                    argv += [answer_flag, answer_path]

                completed = subprocess.run(
                    argv,
                    input=flatten(messages),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                )

                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    raise RuntimeError("{} failed: {}".format(
                        self.command, detail[:400] or "no output"))

                if answer_flag:
                    try:
                        with open(answer_path, encoding="utf-8") as f:
                            raw = f.read()
                    except OSError:
                        # It ran and wrote nothing. stdout is the banner plus
                        # the answer, which is worse than nothing here: a
                        # batch built from it comes back the wrong length and
                        # is discarded anyway.
                        raise RuntimeError(
                            "{} wrote no answer".format(self.command))
                else:
                    raw = completed.stdout
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "{} did not answer within {}s".format(self.command, self.timeout))
        except FileNotFoundError:
            raise RuntimeError(
                "{} is not installed or not on PATH".format(self.command))

        answer = _clean(raw)
        if answer:
            yield answer


def flatten(messages: List[Dict[str, str]]) -> str:
    """
    Render a messages list as one prompt.

    These CLIs take a single prompt on stdin, with no way to pass a role
    structure, so the turns are labelled in text. Losing the real role
    boundaries is the price of using them.
    """
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(content)
        elif role == "assistant":
            parts.append("You previously replied: {}".format(content))
        else:
            parts.append("Speaker: {}".format(content))
    parts.append("Reply now, following the rules above.")
    return "\n\n".join(parts)


def _clean(output: str) -> str:
    """
    Strip the wrappers a CLI may add.

    `claude -p` prints the reply as-is, but with --output-format json it is
    wrapped; be tolerant of both so a config change does not put a JSON blob
    on screen as the answer.
    """
    text = (output or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            return text
        for key in ("result", "text", "response", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return text
    return text
