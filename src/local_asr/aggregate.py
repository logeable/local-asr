from __future__ import annotations

import time
from dataclasses import dataclass, field

from .events import TranscriptEvent


@dataclass(slots=True)
class TranscriptAggregator:
    stable_repeat_threshold: int = 2
    silence_timeout_s: float = 0.8
    stable_text: str = ""
    partial_text: str = ""
    last_text: str = ""
    last_nonempty_ts: float = field(default_factory=time.time)
    pending_stable_candidate: str = ""
    pending_stable_repeats: int = 0

    def feed(self, text: str, *, ts: float | None = None, is_final_chunk: bool = False) -> list[TranscriptEvent]:
        now = time.time() if ts is None else ts
        events: list[TranscriptEvent] = []
        text = text.strip()

        if not text:
            if self.has_pending_text() and (is_final_chunk or now - self.last_nonempty_ts >= self.silence_timeout_s):
                events.extend(self.flush(ts=now))
            return events

        self.last_nonempty_ts = now

        common_prefix = shared_prefix(self.last_text, text)
        if common_prefix and common_prefix != self.pending_stable_candidate:
            self.pending_stable_candidate = common_prefix
            self.pending_stable_repeats = 1
        elif common_prefix:
            self.pending_stable_repeats += 1

        if (
            self.pending_stable_candidate
            and self.pending_stable_repeats >= self.stable_repeat_threshold
            and len(self.pending_stable_candidate) > len(self.stable_text)
        ):
            self.stable_text = self.pending_stable_candidate
            events.append(TranscriptEvent(level="stable", text=self.stable_text, ts=now))

        partial = text[len(self.stable_text) :] if text.startswith(self.stable_text) else text
        if partial != self.partial_text:
            self.partial_text = partial
            if partial:
                events.append(TranscriptEvent(level="partial", text=partial, ts=now))

        self.last_text = text

        if is_final_chunk:
            events.extend(self.flush(ts=now))
        return events

    def flush(self, *, ts: float | None = None) -> list[TranscriptEvent]:
        now = time.time() if ts is None else ts
        final_text = (self.stable_text + self.partial_text).strip()
        if not final_text and self.last_text:
            final_text = self.last_text.strip()
        self.reset_runtime()
        if not final_text:
            return []
        return [TranscriptEvent(level="final", text=final_text, ts=now)]

    def has_pending_text(self) -> bool:
        return bool(self.stable_text or self.partial_text or self.last_text)

    def reset_runtime(self) -> None:
        self.stable_text = ""
        self.partial_text = ""
        self.last_text = ""
        self.pending_stable_candidate = ""
        self.pending_stable_repeats = 0


def shared_prefix(left: str, right: str) -> str:
    chars: list[str] = []
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        chars.append(left_char)
    return "".join(chars)
