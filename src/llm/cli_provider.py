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
            "argv": ["claude", "-p", "--disallowed-tools", ",".join(NO_TOOLS)],
            "stdin": True,
        },
        # codex refuses to run outside a trusted directory unless told not to
        # care, which it has no reason to here -- it is answering a question,
        # not touching a repo.
        "codex": {
            "argv": ["codex", "exec", "--skip-git-repo-check"],
            "stdin": True,
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

    def validate_config(self) -> bool:
        if shutil.which(self.command) is None:
            print("[CLIProvider] {!r} is not on PATH".format(self.command))
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
        if self.model:
            argv += ["--model", self.model]
        argv += self.extra_args

        try:
            # Run from an empty directory. These CLIs read the working
            # directory as project context -- CLAUDE.md, the repo, whatever is
            # around -- and none of it is relevant to answering a question
            # about a transcript. Left in the project directory, an 8-line
            # batch took over 90s and timed out.
            with tempfile.TemporaryDirectory(prefix="echoai-llm-") as workdir:
                completed = subprocess.run(
                    argv,
                    input=flatten(messages),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "{} did not answer within {}s".format(self.command, self.timeout))
        except FileNotFoundError:
            raise RuntimeError(
                "{} is not installed or not on PATH".format(self.command))

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError("{} failed: {}".format(
                self.command, detail[:400] or "no output"))

        answer = _clean(completed.stdout)
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
