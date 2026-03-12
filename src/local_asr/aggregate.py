from __future__ import annotations

import time
from dataclasses import dataclass, field

from .events import TranscriptEvent


@dataclass(slots=True)
class TranscriptAggregator:
    silence_timeout_s: float = 0.8
    stable_text: str = ""
    partial_text: str = ""
    last_text: str = ""
    last_nonempty_ts: float = field(default_factory=time.time)

    def feed(self, text: str, *, ts: float | None = None, is_final_chunk: bool = False) -> list[TranscriptEvent]:
        now = time.time() if ts is None else ts
        events: list[TranscriptEvent] = []
        text = text.strip()

        if not text:
            if self.has_pending_text() and (is_final_chunk or now - self.last_nonempty_ts >= self.silence_timeout_s):
                events.extend(self.flush(ts=now))
            return events

        self.last_nonempty_ts = now

        if not self.last_text:
            self.partial_text = text
            self.last_text = text
            events.append(TranscriptEvent(level="partial", text=text, ts=now))
        elif text.startswith(self.last_text):
            events.extend(self._update_cumulative_text(text, now))
        elif self.last_text.startswith(text):
            if text != self.partial_text:
                self.partial_text = text
                events.append(TranscriptEvent(level="partial", text=text, ts=now))
            self.last_text = text
        else:
            events.extend(self._append_new_segment(text, now))

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

    def _update_cumulative_text(self, text: str, ts: float) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        if self.last_text and len(self.last_text) > len(self.stable_text):
            self.stable_text = self.last_text
            events.append(TranscriptEvent(level="stable", text=self.stable_text, ts=ts))
        partial = text[len(self.stable_text) :] if text.startswith(self.stable_text) else text
        if partial != self.partial_text:
            self.partial_text = partial
            if partial:
                events.append(TranscriptEvent(level="partial", text=partial, ts=ts))
        return events

    def _append_new_segment(self, text: str, ts: float) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        if self.partial_text:
            self.stable_text += self.partial_text
            events.append(TranscriptEvent(level="stable", text=self.stable_text, ts=ts))
        self.partial_text = text
        events.append(TranscriptEvent(level="partial", text=text, ts=ts))
        return events


def shared_prefix(left: str, right: str) -> str:
    chars: list[str] = []
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        chars.append(left_char)
    return "".join(chars)
