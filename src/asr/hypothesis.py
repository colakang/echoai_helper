"""
src/asr/hypothesis.py

LocalAgreement: turn a stream of unstable transcripts into monotonic text.

The transcriber re-runs the model over a growing audio buffer, so successive
results disagree about their own tail -- "good morning, everyone." becomes
"good morning. Everyone." on the next pass, and the UI rewrites text the user
has already read.

LocalAgreement fixes that by only committing what two consecutive hypotheses
agree on. Words in the common prefix are settled and never change; the
disagreeing tail stays provisional. Output becomes append-only.

The idea is from WhisperLiveKit's HypothesisBuffer.flush()
(https://github.com/QuentinFuxa/WhisperLiveKit, MIT). Their implementation
commits on `current_new.text == self.buffer[0].text` -- a pure text
comparison. Timestamps appear elsewhere in their buffer (to offset tokens
between overlapping windows and to trim committed audio), but the commit rule
itself needs none, which is what makes this usable with SenseVoice: it returns
text only, no word timings.

We can drop their offset handling entirely because every hypothesis here is
transcribed from the start of the current segment, so all of them are already
aligned to the same origin.
"""

import re
from typing import List, Tuple

# Split into words while keeping CJK characters as individual tokens: Chinese
# and Cantonese have no spaces, so whitespace splitting would make one phrase a
# single token and defeat prefix matching entirely.
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")


def tokenize(text: str) -> List[str]:
    """Split a transcript into comparison units."""
    return _TOKEN_RE.findall(text or "")


# Trailing/leading punctuation and case are the least stable part of the
# model's output: a re-run turns "good morning, everyone." into "good
# morning. Everyone." Comparing raw tokens makes "morning," and "morning."
# disagree, which stalls the commit prefix on punctuation churn alone --
# measured on a real capture, 19 hypotheses committed only 2 words.
# Compare on a normalised key; display the original text.
_STRIP_RE = re.compile(r"^[^\w" + _CJK + r"]+|[^\w" + _CJK + r"]+$")


def normalize(token: str) -> str:
    """The form used to decide whether two tokens are 'the same word'."""
    return _STRIP_RE.sub("", token).casefold() or token


def detokenize(tokens: List[str]) -> str:
    """
    Rejoin tokens. A space is omitted only between two CJK characters; mixed
    CJK/Latin runs keep theirs, which is both the typographic convention and
    what makes tokenize/detokenize round-trip.
    """
    out = []
    for token in tokens:
        if not out:
            out.append(token)
            continue
        cjk_here = bool(re.match(rf"^[{_CJK}]$", token))
        cjk_prev = bool(re.search(rf"[{_CJK}]$", out[-1]))
        out.append(token if (cjk_here and cjk_prev) else " " + token)
    return "".join(out)


class HypothesisTracker:
    """
    Feed it successive hypotheses for one growing audio segment; it tells you
    which prefix has settled.

        tracker = HypothesisTracker()
        tracker.update("good morning")            # -> committed ""
        tracker.update("good morning everyone")   # -> committed "good morning"

    `committed` only ever grows. `pending` is the unsettled tail and may change
    on every update.
    """

    def __init__(self, agreement: int = 2):
        if agreement < 2:
            raise ValueError("agreement must be at least 2 hypotheses")
        self.agreement = agreement
        self._committed: List[str] = []
        self._previous: List[str] = []

    @property
    def committed(self) -> str:
        return detokenize(self._committed)

    @property
    def pending(self) -> str:
        return detokenize(self._previous[len(self._committed):])

    @property
    def text(self) -> str:
        """Everything known so far: settled prefix plus provisional tail."""
        return detokenize(self._previous if self._previous else self._committed)

    def update(self, hypothesis: str) -> str:
        """
        Absorb one hypothesis. Returns the text newly committed by this call
        (empty when nothing settled).
        """
        current = tokenize(hypothesis)

        # Longest common prefix of this hypothesis and the previous one,
        # compared on normalised tokens so punctuation and capitalisation
        # churn does not stall the commit.
        agreed = 0
        limit = min(len(current), len(self._previous))
        while agreed < limit and normalize(current[agreed]) == normalize(self._previous[agreed]):
            agreed += 1

        newly: List[str] = []
        if agreed > len(self._committed):
            newly = current[len(self._committed):agreed]
            # Append only. Re-slicing from `current` would adopt its spelling
            # of already-committed tokens, so a later pass re-punctuating
            # "morning," as "morning." would rewrite text the user has read --
            # agreement is decided on normalised tokens precisely so that
            # churn does not stall, and it must not leak into the display.
            self._committed = self._committed + newly
        elif not _is_prefix(self._committed, current):
            # The model revised something already committed. We cannot unsay
            # it without the rewriting this class exists to prevent, so keep
            # the commitment and re-anchor the hypothesis onto it.
            current = self._committed + current[len(self._committed):]

        # Anchor the hypothesis on the committed display forms so `pending`
        # never re-shows a token that has already been settled.
        self._previous = self._committed + current[len(self._committed):]
        return detokenize(newly)

    def flush(self) -> str:
        """
        Commit everything, settled or not, and reset.

        Called at a segment boundary: the audio is over, so the last
        hypothesis is final by definition and there will be no further
        agreement to wait for.
        """
        tail = self._previous[len(self._committed):]
        self._committed = list(self._previous)
        text = detokenize(tail)
        self.reset()
        return text

    def reset(self) -> None:
        self._committed = []
        self._previous = []


def _is_prefix(prefix: List[str], tokens: List[str]) -> bool:
    if len(prefix) > len(tokens):
        return False
    return all(normalize(a) == normalize(b) for a, b in zip(prefix, tokens))
