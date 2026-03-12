from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class SilenceEndpointDetector:
    threshold: float
    duration_s: float
    silence_started_at: float | None = None

    def observe(self, rms: float, *, has_pending_text: bool, now: float | None = None) -> bool:
        current_time = time.time() if now is None else now
        if rms >= self.threshold:
            self.silence_started_at = None
            return False
        self.silence_started_at = current_time if self.silence_started_at is None else self.silence_started_at
        if not has_pending_text:
            return False
        return current_time - self.silence_started_at >= self.duration_s

    def reset(self) -> None:
        self.silence_started_at = None
