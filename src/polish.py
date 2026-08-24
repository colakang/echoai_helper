"""
src/polish.py

Clean up a finished transcript with an LLM.

Runs on export rather than during the meeting, which is what makes it
affordable and what makes it good:

  - one call per batch instead of one per utterance. An hour of speech is
    roughly 700 segments; polishing each as it arrived would be 700 requests.
  - latency stops mattering, so the CLI provider becomes usable and the whole
    pass can run on a subscription instead of per-token billing.
  - the model sees the surrounding conversation, so it can fix a mis-heard
    word from context that was not available when that utterance was live.

The pass is *additive*. Every segment keeps its original text; the cleaned
version is stored beside it. A transcript is a record, and a model that
"improves" it is also capable of quietly inventing something that was never
said -- so the raw version has to survive for anyone who wants to check.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# Segments per request. Large enough that the model sees real context, small
# enough that one bad batch spoils little and that the line-count check stays
# meaningful.
DEFAULT_BATCH_SIZE = 25

# Segments of surrounding context sent with each batch but not re-polished.
# Without this the first line of a batch is corrected with no idea what
# preceded it, which is exactly where a mis-heard word needs context most.
DEFAULT_OVERLAP = 3

INSTRUCTIONS = """You are cleaning up a speech-recognition transcript.

The text came from an ASR model and contains its characteristic errors:
mis-heard words that sound like the right one, missing or wrong punctuation,
merged or split sentences, and wrong homophones. Names, numbers and technical
terms are the most frequently mangled.

Correct those, using the surrounding conversation as evidence. Specifically:

- Fix words the recogniser clearly got wrong, where the context makes the
  intended word obvious.
- Fix punctuation, capitalisation and sentence boundaries.
- Keep the speaker's own words, register and language. Do not translate,
  summarise, formalise, or make anyone sound more articulate than they were.
- Keep meaningful hesitation where it carries something; drop pure filler.
- If a line is genuinely unintelligible, leave it exactly as it is.
- Never add information that is not in the transcript. If you are unsure what
  a word was, leave it alone rather than guessing at something plausible.

Return every line back, numbered exactly as given, one per line, in the same
order, in the form:

    12| cleaned text

Return the same number of lines you were given. No commentary, no headings,
nothing else."""

_LINE = re.compile(r"^\s*(\d+)\s*\|\s?(.*)$")


@dataclass
class PolishResult:
    segments: List[dict]
    polished_count: int = 0
    batches_attempted: int = 0
    batches_failed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.batches_failed == 0

    def summary(self) -> str:
        parts = [f"{self.polished_count}/{len(self.segments)} lines cleaned"]
        if self.batches_failed:
            parts.append(f"{self.batches_failed}/{self.batches_attempted} "
                         f"batches failed (originals kept)")
        return ", ".join(parts)


def polish_transcript(messages: List[dict], provider,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      overlap: int = DEFAULT_OVERLAP,
                      progress: Optional[Callable[[int, int], None]] = None
                      ) -> PolishResult:
    """
    Add a `polished` field to each message that has text.

    `messages` is the conversation list from
    ResponseManager.export_structured_conversation. It is not modified;
    copies are returned.

    A batch that fails for any reason -- provider error, wrong number of
    lines back, unparseable output -- leaves its segments with their original
    text and is counted. Partial success is the expected outcome on a long
    transcript and is reported rather than raised.
    """
    segments = [dict(m) for m in messages]
    indexed = [(i, s) for i, s in enumerate(segments)
               if (s.get("text") or "").strip()]

    result = PolishResult(segments=segments)
    if not indexed or provider is None:
        return result

    batches = [indexed[i:i + batch_size]
               for i in range(0, len(indexed), batch_size)]

    for batch_number, batch in enumerate(batches):
        result.batches_attempted += 1
        if progress:
            progress(batch_number, len(batches))

        start = indexed.index(batch[0])
        context = indexed[max(0, start - overlap):start]

        try:
            cleaned = _polish_batch(batch, context, provider)
        except Exception as e:
            result.batches_failed += 1
            result.errors.append(str(e))
            continue

        if len(cleaned) != len(batch):
            # A mismatched count means the lines can no longer be matched to
            # their segments with confidence, and applying them anyway would
            # attribute one speaker's words to another. Drop the batch.
            result.batches_failed += 1
            result.errors.append(
                f"batch {batch_number + 1}: expected {len(batch)} lines, "
                f"got {len(cleaned)}")
            continue

        for (index, segment), text in zip(batch, cleaned):
            if text and text != segment.get("text"):
                segments[index]["polished"] = text
                result.polished_count += 1

    if progress:
        progress(len(batches), len(batches))
    return result


def _polish_batch(batch, context, provider) -> List[str]:
    prompt = [INSTRUCTIONS, ""]

    if context:
        prompt.append("Earlier in the conversation, for context only "
                      "(do not return these):")
        for _, segment in context:
            prompt.append(f"    {_speaker(segment)}{segment.get('text', '')}")
        prompt.append("")

    prompt.append("Clean these lines:")
    for number, (_, segment) in enumerate(batch, start=1):
        prompt.append(f"{number}| {_speaker(segment)}{segment.get('text', '')}")

    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": "\n".join(prompt[1:]).strip()},
    ]

    output = "".join(provider.generate_response(
        messages=messages, temperature=0.2, stream=False))
    return _parse(output, len(batch))


def _speaker(segment: dict) -> str:
    """Speaker label as a prefix, so the model can tell turns apart."""
    label = segment.get("speaker") or segment.get("role")
    return f"[{label}] " if label else ""


def _parse(output: str, expected: int) -> List[str]:
    """
    Pull `12| text` lines back out.

    Keyed by the model's own numbering rather than by position, so a stray
    blank line or a dropped one does not shift every subsequent correction
    onto the wrong segment.
    """
    found: Dict[int, str] = {}
    for raw in (output or "").splitlines():
        match = _LINE.match(raw)
        if not match:
            continue
        number = int(match.group(1))
        if 1 <= number <= expected:
            found[number] = _strip_speaker(match.group(2).strip())

    if len(found) != expected:
        return []
    return [found[i] for i in range(1, expected + 1)]


def _strip_speaker(text: str) -> str:
    """Drop the [S1] prefix if the model echoed it back."""
    return re.sub(r"^\[[^\]]{1,24}\]\s*", "", text)
