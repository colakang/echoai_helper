"""
src/session.py

Write the transcript to disk as it happens.

Until now it lived only in memory until someone remembered to export. A real
meeting ran 84 minutes and produced 1311 lines, all of it riding on one
process staying alive -- and during that same session the microphone stream
died silently and the obvious fix was to restart, which would have thrown the
lot away.

So every settled utterance is appended to a JSONL file as it is produced. A
crash costs the last sentence rather than the meeting. Closing the window
stops being a destructive act. And a past meeting can be opened and exported
again -- a different format, another pass of cleanup, a different number of
speakers -- instead of the single chance the old flow allowed.

Sessions are created automatically, one per launch. Asking someone to make
one before recording is handing them an implementation detail they will
sooner or later forget, and the cost of forgetting is the whole meeting.
JSONL because appending a line is atomic enough to survive being killed
mid-write: at worst the final line is truncated and is skipped on read.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

SESSION_SUFFIX = ".jsonl"
# Written when a session ends cleanly. Its absence is what marks a crash.
CLOSED_MARKER = "closed"


def sessions_dir() -> Path:
    from .config import PathConfig
    path = Path(PathConfig.get_user_config_path()) / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class SessionInfo:
    path: Path
    started: datetime
    line_count: int
    closed: bool
    duration_s: float = 0.0
    speakers: int = 0

    @property
    def name(self) -> str:
        return self.started.strftime("%Y-%m-%d %H:%M")

    @property
    def crashed(self) -> bool:
        """Ended without a closing marker, and had something in it."""
        return not self.closed and self.line_count > 0

    def describe(self) -> str:
        minutes = self.duration_s / 60
        parts = [self.name, f"{self.line_count} lines"]
        if minutes >= 1:
            parts.append(f"{minutes:.0f} min")
        if self.speakers:
            parts.append(f"{self.speakers} speakers")
        if self.crashed:
            parts.append("not exported")
        return "  ·  ".join(parts)


class SessionWriter:
    """
    Appends utterances to one session file.

    Every write is flushed. Buffering would trade the one property this exists
    for -- surviving a process that dies without warning -- against a saving
    of no consequence at a few lines a minute.
    """

    def __init__(self, directory: Optional[Path] = None):
        self.started = datetime.now()
        base = directory or sessions_dir()
        # Second resolution alone collides -- two sessions started in the same
        # second would append to one file and read back as a single meeting.
        # Reachable in practice: starting a new session from Clear Transcript.
        stamp = self.started.strftime("%Y%m%d_%H%M%S")
        self.path = base / f"session_{stamp}{SESSION_SUFFIX}"
        suffix = 1
        while self.path.exists():
            self.path = base / f"session_{stamp}_{suffix}{SESSION_SUFFIX}"
            suffix += 1
        self.line_count = 0
        self._file = None
        self._open()

    def _open(self) -> None:
        try:
            self._file = open(self.path, "a", encoding="utf-8")
            self._write({"type": "session", "started": self.started.isoformat()})
        except OSError as e:
            # Never fatal. A meeting that is recorded but not saved is worth
            # more than one that refuses to start.
            print(f"[WARN] Could not open the session file: {e}")
            self._file = None

    def _write(self, payload: dict) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._file.flush()
        except (OSError, TypeError) as e:
            print(f"[WARN] Could not write to the session file: {e}")

    def append(self, track: str, text: str, timestamp: datetime,
               speaker: Optional[str] = None, response_id: Optional[str] = None,
               embedding=None) -> None:
        """Record one settled utterance."""
        if not (text or "").strip():
            return
        payload = {
            "type": "utterance",
            "track": track,
            "text": text,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        if speaker:
            payload["speaker"] = speaker
        if response_id:
            payload["response_id"] = response_id
        if embedding is not None:
            # The voice print, so speakers can be re-grouped on a later
            # export. Four decimals: cosine similarities agree with full
            # precision to about 1e-3, well below any threshold that matters.
            payload["embedding"] = [round(float(v), 4) for v in embedding]
        self._write(payload)
        self.line_count += 1

    def append_response(self, response_id: str, question: str, answer: str) -> None:
        self._write({"type": "response", "response_id": response_id,
                     "question": question, "answer": answer})

    def close(self) -> None:
        """Mark the session as finished, so the next launch knows it did not crash."""
        self._write({"type": CLOSED_MARKER,
                     "ended": datetime.now().isoformat(),
                     "lines": self.line_count})
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


def read_session(path: Path) -> List[dict]:
    """
    Every record in a session file.

    A truncated final line -- the process was killed mid-write -- is skipped
    rather than raising. Losing the last utterance is the expected cost of a
    crash; losing the file because of it would not be.
    """
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        print(f"[WARN] Could not read {path}: {e}")
    return records


def inspect_session(path: Path) -> Optional[SessionInfo]:
    records = read_session(path)
    if not records:
        return None

    utterances = [r for r in records if r.get("type") == "utterance"]
    closed = any(r.get("type") == CLOSED_MARKER for r in records)

    started = None
    for record in records:
        if record.get("type") == "session":
            started = _parse(record.get("started"))
            break
    if started is None:
        started = datetime.fromtimestamp(path.stat().st_mtime)

    duration = 0.0
    if len(utterances) >= 2:
        first = _parse(utterances[0].get("timestamp"))
        last = _parse(utterances[-1].get("timestamp"))
        if first and last:
            duration = max(0.0, (last - first).total_seconds())

    speakers = {u.get("speaker", "").rstrip("?")
                for u in utterances if u.get("speaker")}
    return SessionInfo(path=path, started=started, line_count=len(utterances),
                       closed=closed, duration_s=duration,
                       speakers=len(speakers))


def list_sessions(directory: Optional[Path] = None) -> List[SessionInfo]:
    """Past sessions, newest first."""
    directory = directory or sessions_dir()
    found = []
    for path in sorted(directory.glob(f"*{SESSION_SUFFIX}"), reverse=True):
        info = inspect_session(path)
        if info is not None:
            found.append(info)
    return found


def find_recoverable(directory: Optional[Path] = None,
                     within_hours: float = 12.0) -> Optional[SessionInfo]:
    """
    The most recent session that ended without being closed.

    Bounded by age deliberately: a meeting from last week is something to
    export from the history, not to resume into. Offering to continue it would
    splice two unrelated conversations into one transcript.
    """
    cutoff = time.time() - within_hours * 3600
    for info in list_sessions(directory):
        if not info.crashed:
            continue
        if info.path.stat().st_mtime < cutoff:
            return None          # newest first, so anything older is too old
        return info
    return None


def to_conversation(path: Path) -> dict:
    """Rebuild the export payload from a session file."""
    records = read_session(path)
    utterances = [r for r in records if r.get("type") == "utterance"]
    answers = {r["response_id"]: r for r in records
               if r.get("type") == "response" and r.get("response_id")}

    messages = []
    for index, record in enumerate(utterances):
        message = {
            "role": (record.get("track") or "speaker").lower(),
            "text": record.get("text", ""),
            "timestamp": record.get("timestamp"),
            "index": index,
        }
        for key in ("speaker", "embedding", "response_id"):
            if record.get(key) is not None:
                message[key] = record[key]

        answer = answers.get(record.get("response_id"))
        if answer and answer.get("answer"):
            message["response"] = {"response_text": answer["answer"],
                                   "question_text": answer.get("question", "")}
        messages.append(message)

    started = None
    for record in records:
        if record.get("type") == "session":
            started = record.get("started")
            break

    return {
        "metadata": {
            "export_time": datetime.now().astimezone().isoformat(),
            "session_started": started,
            "version": "2.1",
            "total_messages": len(messages),
            "source": str(path),
        },
        "conversation": {"messages": messages},
    }


def delete_session(path: Path) -> bool:
    try:
        os.remove(path)
        return True
    except OSError as e:
        print(f"[WARN] Could not delete {path}: {e}")
        return False


def _parse(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
