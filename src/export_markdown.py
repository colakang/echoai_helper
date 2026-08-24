"""
src/export_markdown.py

Render a finished conversation as Markdown.

The JSON export is a data format: complete, machine-readable, and nobody's
idea of meeting notes. This is the human-facing one -- what actually gets
pasted into a doc or a ticket.

Which means a different default. The JSON keeps every original line because
it is a record and something has to be auditable. Markdown is read, so it
carries the cleaned text and drops the raw version unless asked: showing both
for every line doubles the length and buries the content in bookkeeping.

`include_original=True` is for when the transcript is evidence rather than
notes -- a disputed call, an interview being assessed -- and it renders the
raw line beneath the cleaned one so the two can be compared.
"""

from datetime import datetime
from typing import Dict, List, Optional

# Speaker labels the diarizer emits. A trailing "?" means it was unsure.
UNCERTAIN = "?"


def render(conversation: dict, include_original: bool = False,
           title: Optional[str] = None) -> str:
    """
    Turn the export payload into Markdown.

    Args:
        conversation: what export_structured_conversation returned.
        include_original: also show the pre-cleanup text of any line that
            was changed.
        title: heading; defaults to the export date.
    """
    metadata = conversation.get("metadata", {}) or {}
    messages = (conversation.get("conversation", {}) or {}).get("messages", []) or []

    lines: List[str] = []
    lines.append(f"# {title or _default_title(metadata)}")
    lines.append("")
    lines.extend(_front_matter(metadata, messages))
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    if not messages:
        lines.append("_No conversation was recorded._")
        return "\n".join(lines) + "\n"

    previous_speaker = None
    for message in messages:
        speaker = _speaker_of(message)
        if speaker != previous_speaker:
            lines.append("")
            lines.append(f"**{speaker}**")
            lines.append("")
            previous_speaker = speaker

        lines.extend(_render_message(message, include_original))

    answers = [m for m in messages if (m.get("response") or {}).get("response_text")]
    if answers:
        lines.append("")
        lines.append("## Suggested replies")
        lines.append("")
        lines.append("_Generated during the conversation; not spoken._")
        lines.append("")
        for message in answers:
            question = _text_of(message)
            answer = message["response"]["response_text"].strip()
            lines.append(f"- **{_timestamp(message)}** {question}")
            lines.append(f"  - {answer}")

    return "\n".join(lines).rstrip() + "\n"


def _render_message(message: dict, include_original: bool) -> List[str]:
    text = _text_of(message)
    if not text:
        return []

    stamp = _timestamp(message)
    prefix = f"`{stamp}` " if stamp else ""
    out = [f"{prefix}{text}", ""]

    original = message.get("text", "")
    if include_original and message.get("polished") and original.strip() != text:
        out.insert(1, f"> _as heard:_ {original.strip()}")
        out.insert(2, "")
    return out


def _speaker_of(message: dict) -> str:
    """
    Who is talking, as a reader would want to see it.

    The transcriber prefixes far-end lines with a diarization label ("S2: ...")
    when speaker labelling is on. Lift it out so turns group under a heading
    instead of repeating the tag on every line.
    """
    label = message.get("speaker")
    if not label:
        raw = message.get("polished") or message.get("text") or ""
        head, _, _ = raw.partition(":")
        if head and len(head) <= 4 and head.lstrip("S").rstrip(UNCERTAIN).isdigit():
            label = head

    role = (message.get("role") or "").lower()
    if label:
        uncertain = label.endswith(UNCERTAIN)
        name = f"Speaker {label.rstrip(UNCERTAIN).lstrip('S')}"
        return name + (" (uncertain)" if uncertain else "")
    return "You" if role == "you" else "Speaker"


def _text_of(message: dict) -> str:
    """Cleaned text if there is one, with any speaker tag removed."""
    text = (message.get("polished") or message.get("text") or "").strip()
    head, sep, rest = text.partition(":")
    if sep and len(head) <= 4 and head.lstrip("S").rstrip(UNCERTAIN).isdigit():
        return rest.strip()
    return text


def _timestamp(message: dict) -> str:
    """Clock time only -- the date is already in the heading."""
    raw = message.get("timestamp")
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return str(raw)[:8]


def _default_title(metadata: Dict) -> str:
    raw = metadata.get("export_time")
    if raw:
        try:
            return datetime.fromisoformat(raw).strftime("Conversation — %Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            pass
    return "Conversation"


def _front_matter(metadata: Dict, messages: List[dict]) -> List[str]:
    speakers = sorted({_speaker_of(m) for m in messages if _text_of(m)})
    lines = [
        f"- **Lines:** {len([m for m in messages if _text_of(m)])}",
        f"- **Speakers:** {', '.join(speakers) if speakers else '—'}",
    ]

    cleanup = metadata.get("cleanup")
    if cleanup:
        note = f"{cleanup.get('lines_cleaned', 0)} lines corrected by " \
               f"{cleanup.get('model', 'a language model')}"
        if cleanup.get("batches_failed"):
            note += f" ({cleanup['batches_failed']} batches failed)"
        lines.append(f"- **Cleanup:** {note}")
    return lines
